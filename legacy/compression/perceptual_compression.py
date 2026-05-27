"""
Perceptual compression - optimize for network behavior, not raw weights.

Key insight: A 50% error in a small weight matters less than 5% error in a large weight.
We should compress based on IMPACT ON OUTPUT, not raw reconstruction error.

This is like JPEG for images - optimize for what humans perceive, not pixel-perfect accuracy.
For neural networks, we optimize for prediction accuracy, not weight accuracy.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
import pickle


class PerceptualCompressor:
    """
    Compress weights based on their impact on network output.
    
    Strategy:
    1. Measure importance of each frequency by its effect on predictions
    2. Keep frequencies that most affect output
    3. Discard frequencies that have minimal behavioral impact
    
    Expected: <10% error on actual network performance
    """
    
    def __init__(self, compression_ratio: int = 100):
        """
        Initialize perceptual compressor.
        
        Args:
            compression_ratio: Target compression ratio
        """
        self.compression_ratio = compression_ratio
        self.freq_coefficients = None
        self.freq_indices = None
        self.original_shape = None
        self.importance_scores = None
        
    def compress(self, 
                 weights: np.ndarray,
                 forward_fn: Optional[Callable] = None,
                 sample_inputs: Optional[np.ndarray] = None) -> Dict:
        """
        Compress weights based on perceptual importance.
        
        Args:
            weights: Weight matrix to compress
            forward_fn: Function that computes network output given weights
                       Signature: forward_fn(weights, inputs) -> outputs
            sample_inputs: Sample inputs for measuring importance
            
        Returns:
            Compression metadata
        """
        self.original_shape = weights.shape
        
        # Compute FFT
        freq_spectrum = np.fft.fft2(weights)
        flat_spectrum = freq_spectrum.flatten()
        
        # Calculate how many coefficients to keep
        n_keep = max(1, weights.size // self.compression_ratio)
        n_keep = min(n_keep, len(flat_spectrum))
        
        if forward_fn is not None and sample_inputs is not None:
            # Perceptual importance: measure impact on outputs
            importance = self._compute_perceptual_importance(
                weights, freq_spectrum, forward_fn, sample_inputs
            )
        else:
            # Fallback to magnitude-based importance
            importance = self._compute_magnitude_importance(flat_spectrum)
        
        self.importance_scores = importance
        
        # Keep top-k by importance
        if n_keep >= len(importance):
            top_indices = np.arange(len(importance))
        else:
            top_indices = np.argpartition(importance, -n_keep)[-n_keep:]
        
        self.freq_coefficients = flat_spectrum[top_indices]
        self.freq_indices = top_indices
        
        # Metadata
        actual_ratio = weights.size / len(self.freq_coefficients)
        
        return {
            'original_size': weights.size,
            'compressed_size': len(self.freq_coefficients),
            'compression_ratio': actual_ratio,
            'avg_importance': np.mean(importance[top_indices])
        }
    
    def _compute_perceptual_importance(self,
                                      weights: np.ndarray,
                                      freq_spectrum: np.ndarray,
                                      forward_fn: Callable,
                                      sample_inputs: np.ndarray) -> np.ndarray:
        """
        Compute importance of each frequency based on output impact.
        
        This is expensive but gives best compression quality.
        """
        # Get baseline outputs
        baseline_outputs = forward_fn(weights, sample_inputs)
        
        # Test each frequency's importance
        flat_spectrum = freq_spectrum.flatten()
        importance = np.zeros(len(flat_spectrum))
        
        # Sample a subset of frequencies to test (too expensive to test all)
        n_test = min(1000, len(flat_spectrum))
        test_indices = np.random.choice(len(flat_spectrum), n_test, replace=False)
        
        for idx in test_indices:
            # Create spectrum with this frequency zeroed out
            test_spectrum = flat_spectrum.copy()
            test_spectrum[idx] = 0
            
            # Reconstruct weights
            test_spectrum_2d = test_spectrum.reshape(freq_spectrum.shape)
            test_weights = np.fft.ifft2(test_spectrum_2d).real
            
            # Measure output change
            test_outputs = forward_fn(test_weights, sample_inputs)
            output_error = np.mean((baseline_outputs - test_outputs) ** 2)
            
            importance[idx] = output_error
        
        # For untested frequencies, use magnitude as proxy
        untested_mask = np.ones(len(flat_spectrum), dtype=bool)
        untested_mask[test_indices] = False
        importance[untested_mask] = np.abs(flat_spectrum[untested_mask])
        
        return importance
    
    def _compute_magnitude_importance(self, flat_spectrum: np.ndarray) -> np.ndarray:
        """
        Fallback: use magnitude as importance proxy.
        
        Weighted by position (low frequencies more important).
        """
        magnitudes = np.abs(flat_spectrum)
        
        # Weight by frequency position (low freq = more important)
        n = len(flat_spectrum)
        freq_weights = np.exp(-np.arange(n) / (n / 4))  # Exponential decay
        
        importance = magnitudes * freq_weights
        
        return importance
    
    def decompress(self) -> np.ndarray:
        """Reconstruct weights from compressed representation."""
        if self.freq_coefficients is None:
            raise ValueError("No compressed data available")
        
        # Reconstruct full spectrum
        full_spectrum = np.zeros(self.original_shape, dtype=complex)
        flat_spectrum = full_spectrum.flatten()
        flat_spectrum[self.freq_indices] = self.freq_coefficients
        full_spectrum = flat_spectrum.reshape(self.original_shape)
        
        # Inverse FFT
        reconstructed = np.fft.ifft2(full_spectrum).real
        
        return reconstructed
    
    def save(self, filepath: str):
        """Save compressed representation."""
        data = {
            'freq_coefficients': self.freq_coefficients,
            'freq_indices': self.freq_indices,
            'original_shape': self.original_shape,
            'importance_scores': self.importance_scores,
            'compression_ratio': self.compression_ratio
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load compressed representation."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.freq_coefficients = data['freq_coefficients']
        self.freq_indices = data['freq_indices']
        self.original_shape = data['original_shape']
        self.importance_scores = data['importance_scores']
        self.compression_ratio = data['compression_ratio']


class WeightedPerceptualCompressor:
    """
    Enhanced perceptual compressor with weight-magnitude weighting.
    
    Key insight: Errors in large weights matter more than errors in small weights.
    Weight the importance by the magnitude of the original weight.
    """
    
    def __init__(self, compression_ratio: int = 100):
        self.compression_ratio = compression_ratio
        self.freq_coefficients = None
        self.freq_indices = None
        self.original_shape = None
        
    def compress(self, weights: np.ndarray) -> Dict:
        """
        Compress with magnitude-weighted importance.
        
        Fast version: use frequency magnitude × spatial contribution estimate
        
        Args:
            weights: Weight matrix
            
        Returns:
            Compression metadata
        """
        self.original_shape = weights.shape
        
        # FFT
        freq_spectrum = np.fft.fft2(weights)
        flat_spectrum = freq_spectrum.flatten()
        magnitudes = np.abs(flat_spectrum)
        
        # Fast importance estimation:
        # 1. Frequency magnitude (how strong is this component)
        # 2. Position weighting (low frequencies more important)
        # 3. Phase coherence (how aligned with other frequencies)
        
        h, w = weights.shape
        
        # Create frequency position map
        freq_y = np.fft.fftfreq(h).reshape(-1, 1)
        freq_x = np.fft.fftfreq(w).reshape(1, -1)
        freq_dist = np.sqrt(freq_y**2 + freq_x**2).flatten()
        
        # Low frequency = more important (exponential decay)
        position_weight = np.exp(-freq_dist * 5)
        
        # Combine: magnitude × position weight
        importance = magnitudes * position_weight
        
        # Boost importance of frequencies with high magnitude
        # (errors in strong components matter more)
        importance *= (1 + magnitudes / (magnitudes.max() + 1e-10))
        
        # Keep top-k
        n_keep = max(1, weights.size // self.compression_ratio)
        n_keep = min(n_keep, len(importance))
        
        if n_keep >= len(importance):
            top_indices = np.arange(len(importance))
        else:
            top_indices = np.argpartition(importance, -n_keep)[-n_keep:]
        
        self.freq_coefficients = flat_spectrum[top_indices]
        self.freq_indices = top_indices
        
        return {
            'original_size': weights.size,
            'compressed_size': len(self.freq_coefficients),
            'compression_ratio': weights.size / len(self.freq_coefficients)
        }
    
    def decompress(self) -> np.ndarray:
        """Reconstruct weights."""
        full_spectrum = np.zeros(self.original_shape, dtype=complex)
        flat_spectrum = full_spectrum.flatten()
        flat_spectrum[self.freq_indices] = self.freq_coefficients
        full_spectrum = flat_spectrum.reshape(self.original_shape)
        
        return np.fft.ifft2(full_spectrum).real


def benchmark_perceptual_compression(weights: np.ndarray,
                                     forward_fn: Optional[Callable] = None,
                                     sample_inputs: Optional[np.ndarray] = None,
                                     ratios: list = None) -> Dict:
    """
    Benchmark perceptual compression.
    
    Args:
        weights: Weight matrix
        forward_fn: Forward function for perceptual importance
        sample_inputs: Sample inputs
        ratios: Compression ratios to test
        
    Returns:
        Benchmark results
    """
    if ratios is None:
        ratios = [10, 50, 100, 500, 1000]
    
    results = {}
    
    for ratio in ratios:
        # Perceptual compression
        compressor = PerceptualCompressor(compression_ratio=ratio)
        metadata = compressor.compress(weights, forward_fn, sample_inputs)
        reconstructed = compressor.decompress()
        
        # Calculate errors
        weight_error = np.mean(np.abs(weights - reconstructed)) / (np.abs(weights).mean() + 1e-10)
        
        # If we have forward function, measure output error too
        if forward_fn is not None and sample_inputs is not None:
            original_outputs = forward_fn(weights, sample_inputs)
            reconstructed_outputs = forward_fn(reconstructed, sample_inputs)
            output_error = np.mean(np.abs(original_outputs - reconstructed_outputs)) / (np.abs(original_outputs).mean() + 1e-10)
        else:
            output_error = None
        
        results[ratio] = {
            'compression_ratio': metadata['compression_ratio'],
            'weight_error': weight_error,
            'output_error': output_error,
            'compressed_size': metadata['compressed_size']
        }
    
    return results


def compare_all_methods(weights: np.ndarray,
                       forward_fn: Optional[Callable] = None,
                       sample_inputs: Optional[np.ndarray] = None) -> Dict:
    """
    Compare all compression methods.
    
    Args:
        weights: Weight matrix
        forward_fn: Forward function
        sample_inputs: Sample inputs
        
    Returns:
        Comparison results
    """
    from .fourier_compression import FourierCompressor
    from .hybrid_compression import HybridCompressor
    
    results = {}
    
    # Pure Fourier
    fourier = FourierCompressor(compression_ratio=100)
    fourier.compress(weights)
    fourier_recon = fourier.decompress()
    
    results['fourier'] = {
        'weight_error': np.mean(np.abs(weights - fourier_recon)) / (np.abs(weights).mean() + 1e-10),
        'compression_ratio': 100
    }
    
    # Hybrid
    hybrid = HybridCompressor(sparsity_threshold=0.01, fourier_ratio=50)
    hybrid.compress(weights)
    hybrid_recon = hybrid.decompress()
    
    results['hybrid'] = {
        'weight_error': np.mean(np.abs(weights - hybrid_recon)) / (np.abs(weights).mean() + 1e-10),
        'compression_ratio': weights.size / (len(hybrid.sparse_values) + len(hybrid.dense_freq_coeffs))
    }
    
    # Perceptual
    perceptual = PerceptualCompressor(compression_ratio=100)
    perceptual.compress(weights, forward_fn, sample_inputs)
    perceptual_recon = perceptual.decompress()
    
    results['perceptual'] = {
        'weight_error': np.mean(np.abs(weights - perceptual_recon)) / (np.abs(weights).mean() + 1e-10),
        'compression_ratio': 100
    }
    
    # Weighted perceptual
    weighted = WeightedPerceptualCompressor(compression_ratio=100)
    weighted.compress(weights)
    weighted_recon = weighted.decompress()
    
    results['weighted_perceptual'] = {
        'weight_error': np.mean(np.abs(weights - weighted_recon)) / (np.abs(weights).mean() + 1e-10),
        'compression_ratio': 100
    }
    
    # If we have forward function, measure output errors
    if forward_fn is not None and sample_inputs is not None:
        original_outputs = forward_fn(weights, sample_inputs)
        
        for method, recon in [('fourier', fourier_recon), 
                              ('hybrid', hybrid_recon),
                              ('perceptual', perceptual_recon),
                              ('weighted_perceptual', weighted_recon)]:
            recon_outputs = forward_fn(recon, sample_inputs)
            output_error = np.mean(np.abs(original_outputs - recon_outputs)) / (np.abs(original_outputs).mean() + 1e-10)
            results[method]['output_error'] = output_error
    
    return results
