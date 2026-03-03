"""
Adaptive compression that automatically selects the best method.

Analyzes weight characteristics and chooses:
- Fourier: for dense, structured patterns
- Statistical: for sparse, random distributions  
- Hybrid: for mixed characteristics

This is the "smart" compressor that should be used in production.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle

from .fourier_compression import FourierCompressor
from .hybrid_compression import HybridCompressor
from .statistical_compression import GaussianStatisticalCompressor, SimpleGaussianCompressor


class AdaptiveCompressor:
    """
    Automatically selects best compression method based on weight characteristics.
    
    Decision criteria:
    1. Sparsity: >90% → Statistical
    2. Low-frequency dominance: >70% → Fourier
    3. Mixed: → Hybrid
    """
    
    def __init__(self, target_ratio: int = 100):
        """
        Initialize adaptive compressor.
        
        Args:
            target_ratio: Target compression ratio
        """
        self.target_ratio = target_ratio
        self.selected_method = None
        self.compressor = None
        self.original_shape = None
        
    def analyze_weights(self, weights: np.ndarray) -> Dict:
        """
        Analyze weight characteristics to determine best method.
        
        Returns:
            Analysis results with recommended method
        """
        # 1. Sparsity analysis
        sparsity = np.sum(weights == 0) / weights.size
        near_zero_sparsity = np.sum(np.abs(weights) < 0.01) / weights.size
        
        # 2. Frequency analysis
        freq_spectrum = np.fft.fft2(weights)
        freq_power = np.abs(freq_spectrum) ** 2
        
        # Calculate low-frequency power (center 10% of spectrum)
        h, w = freq_power.shape
        center_h, center_w = h // 2, w // 2
        low_freq_region = freq_power[
            center_h - h//10:center_h + h//10,
            center_w - w//10:center_w + w//10
        ]
        low_freq_power = low_freq_region.sum() / freq_power.sum()
        
        # 3. Distribution analysis
        non_zero_weights = weights[weights != 0]
        if len(non_zero_weights) > 0:
            weight_std = np.std(non_zero_weights)
            weight_mean = np.mean(np.abs(non_zero_weights))
            coefficient_of_variation = weight_std / (weight_mean + 1e-10)
        else:
            coefficient_of_variation = 0
        
        # 4. Structure analysis (autocorrelation)
        if weights.size > 100:
            # Sample autocorrelation
            sample_size = min(100, weights.shape[0])
            sample = weights[:sample_size, :sample_size]
            autocorr = np.correlate(sample.flatten(), sample.flatten(), mode='same')
            structure_score = np.max(autocorr[1:]) / (autocorr[0] + 1e-10)
        else:
            structure_score = 0
        
        # Decision logic
        if sparsity > 0.90:
            recommended = 'statistical'
            reason = f'High sparsity ({sparsity:.1%})'
        elif low_freq_power > 0.70:
            recommended = 'fourier'
            reason = f'Low-frequency dominant ({low_freq_power:.1%})'
        elif structure_score > 0.5:
            recommended = 'fourier'
            reason = f'High structure score ({structure_score:.2f})'
        elif near_zero_sparsity > 0.50:
            recommended = 'hybrid'
            reason = f'Mixed sparse/dense ({near_zero_sparsity:.1%} near-zero)'
        else:
            recommended = 'hybrid'
            reason = 'Mixed characteristics'
        
        return {
            'sparsity': sparsity,
            'near_zero_sparsity': near_zero_sparsity,
            'low_freq_power': low_freq_power,
            'coefficient_of_variation': coefficient_of_variation,
            'structure_score': structure_score,
            'recommended_method': recommended,
            'reason': reason
        }
    
    def compress(self, weights: np.ndarray, force_method: Optional[str] = None) -> Dict:
        """
        Compress weights using best method.
        
        Args:
            weights: Weight matrix
            force_method: Force specific method ('fourier', 'statistical', 'hybrid')
            
        Returns:
            Compression metadata
        """
        self.original_shape = weights.shape
        
        # Analyze weights
        analysis = self.analyze_weights(weights)
        
        # Select method
        if force_method:
            self.selected_method = force_method
        else:
            self.selected_method = analysis['recommended_method']
        
        # Compress with selected method
        if self.selected_method == 'fourier':
            self.compressor = FourierCompressor(compression_ratio=self.target_ratio)
            metadata = self.compressor.compress(weights, method='adaptive')
            
        elif self.selected_method == 'statistical':
            # Choose between GMM and Simple based on non-zero count
            non_zero_count = np.count_nonzero(weights)
            
            if non_zero_count > 100:
                # Enough data for GMM
                self.compressor = GaussianStatisticalCompressor(
                    n_components=3,
                    outlier_threshold=3.0
                )
            else:
                # Too sparse, use simple
                self.compressor = SimpleGaussianCompressor(outlier_threshold=2.0)
            
            metadata = self.compressor.compress(weights)
            
        else:  # hybrid
            # Adaptive threshold based on sparsity
            threshold = 0.01 if analysis['near_zero_sparsity'] > 0.7 else 0.05
            
            self.compressor = HybridCompressor(
                sparsity_threshold=threshold,
                fourier_ratio=50
            )
            metadata = self.compressor.compress(weights)
        
        # Add analysis to metadata
        metadata['selected_method'] = self.selected_method
        metadata['selection_reason'] = analysis['reason']
        metadata['sparsity'] = analysis['sparsity']
        metadata['low_freq_power'] = analysis['low_freq_power']
        
        return metadata
    
    def decompress(self) -> np.ndarray:
        """Reconstruct weights using selected method."""
        if self.compressor is None:
            raise ValueError("No compressed data available")
        
        if self.selected_method == 'statistical':
            # Check if GMM or Simple
            if hasattr(self.compressor, 'gmm'):
                return self.compressor.decompress(deterministic=True)
            else:
                return self.compressor.decompress()
        else:
            return self.compressor.decompress()
    
    def save(self, filepath: str):
        """Save compressed representation."""
        data = {
            'selected_method': self.selected_method,
            'compressor': self.compressor,
            'original_shape': self.original_shape,
            'target_ratio': self.target_ratio
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load compressed representation."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.selected_method = data['selected_method']
        self.compressor = data['compressor']
        self.original_shape = data['original_shape']
        self.target_ratio = data['target_ratio']


def benchmark_adaptive_compression(test_cases: list = None) -> Dict:
    """
    Benchmark adaptive compressor on different weight types.
    
    Args:
        test_cases: List of (name, weights) tuples
        
    Returns:
        Benchmark results
    """
    if test_cases is None:
        # Create diverse test cases
        test_cases = []
        
        # 1. Dense structured (CNN-like)
        size = 100
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        X, Y = np.meshgrid(x, y)
        dense_structured = 0.5 * np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(size, size)
        test_cases.append(('Dense Structured (CNN)', dense_structured))
        
        # 2. Sparse random (FC with dropout)
        sparse_random = np.zeros((size, size))
        n_connections = int(size * size * 0.05)
        indices = np.random.choice(size*size, n_connections, replace=False)
        i_idx, j_idx = np.unravel_index(indices, (size, size))
        sparse_random[i_idx, j_idx] = np.random.randn(n_connections) * 0.3
        test_cases.append(('Sparse Random (FC)', sparse_random))
        
        # 3. Mixed (some structure + sparsity)
        mixed = np.zeros((size, size))
        mixed[:50, :50] = 0.3 * np.sin(X[:50, :50]) * np.cos(Y[:50, :50])
        n_sparse = int(size * size * 0.1)
        indices = np.random.choice(size*size, n_sparse, replace=False)
        i_idx, j_idx = np.unravel_index(indices, (size, size))
        mixed[i_idx, j_idx] += np.random.randn(n_sparse) * 0.2
        test_cases.append(('Mixed (Hybrid)', mixed))
    
    results = {}
    
    for name, weights in test_cases:
        # Adaptive compression
        adaptive = AdaptiveCompressor(target_ratio=100)
        metadata = adaptive.compress(weights)
        reconstructed = adaptive.decompress()
        
        error = np.mean(np.abs(weights - reconstructed)) / (np.abs(weights).mean() + 1e-10)
        
        results[name] = {
            'selected_method': metadata['selected_method'],
            'reason': metadata['selection_reason'],
            'compression_ratio': metadata['compression_ratio'],
            'error': error,
            'sparsity': metadata['sparsity'],
            'low_freq_power': metadata['low_freq_power']
        }
    
    return results


def compare_adaptive_vs_fixed(weights: np.ndarray) -> Dict:
    """
    Compare adaptive selection vs fixed methods.
    
    Args:
        weights: Weight matrix
        
    Returns:
        Comparison results
    """
    results = {}
    
    # Adaptive (auto-select)
    adaptive = AdaptiveCompressor(target_ratio=100)
    adaptive_meta = adaptive.compress(weights)
    adaptive_recon = adaptive.decompress()
    
    results['adaptive'] = {
        'method': adaptive_meta['selected_method'],
        'reason': adaptive_meta['selection_reason'],
        'compression_ratio': adaptive_meta['compression_ratio'],
        'error': np.mean(np.abs(weights - adaptive_recon)) / (np.abs(weights).mean() + 1e-10)
    }
    
    # Force Fourier
    fourier_adaptive = AdaptiveCompressor(target_ratio=100)
    fourier_meta = fourier_adaptive.compress(weights, force_method='fourier')
    fourier_recon = fourier_adaptive.decompress()
    
    results['fourier'] = {
        'method': 'fourier',
        'compression_ratio': fourier_meta['compression_ratio'],
        'error': np.mean(np.abs(weights - fourier_recon)) / (np.abs(weights).mean() + 1e-10)
    }
    
    # Force Statistical
    stat_adaptive = AdaptiveCompressor(target_ratio=100)
    stat_meta = stat_adaptive.compress(weights, force_method='statistical')
    stat_recon = stat_adaptive.decompress()
    
    results['statistical'] = {
        'method': 'statistical',
        'compression_ratio': stat_meta['compression_ratio'],
        'error': np.mean(np.abs(weights - stat_recon)) / (np.abs(weights).mean() + 1e-10)
    }
    
    # Force Hybrid
    hybrid_adaptive = AdaptiveCompressor(target_ratio=100)
    hybrid_meta = hybrid_adaptive.compress(weights, force_method='hybrid')
    hybrid_recon = hybrid_adaptive.decompress()
    
    results['hybrid'] = {
        'method': 'hybrid',
        'compression_ratio': hybrid_meta['compression_ratio'],
        'error': np.mean(np.abs(weights - hybrid_recon)) / (np.abs(weights).mean() + 1e-10)
    }
    
    return results
