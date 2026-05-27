"""
Statistical compression using Gaussian mixture models.

Key insight: Neural networks have predictable weight distributions.
Instead of storing individual weights, store the distribution parameters.

For sparse networks (95%+ zeros), this is perfect:
- Model 1: Near-zero weights (mean=0, small std)
- Model 2: Strong positive weights (mean>0, larger std)
- Model 3: Strong negative weights (mean<0, larger std)
- Store only outliers explicitly

Expected: 10,000x compression with <10% error
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy import stats
from sklearn.mixture import GaussianMixture
import pickle


class GaussianStatisticalCompressor:
    """
    Compress weights using Gaussian mixture models.
    
    Strategy:
    1. Fit Gaussian mixture to weight distribution
    2. Store only: mixture parameters + outliers
    3. Reconstruct by sampling from mixture + restoring outliers
    
    Perfect for sparse networks!
    """
    
    def __init__(self, 
                 n_components: int = 3,
                 outlier_threshold: float = 3.0):
        """
        Initialize statistical compressor.
        
        Args:
            n_components: Number of Gaussian components (typically 2-5)
            outlier_threshold: Standard deviations for outlier detection
        """
        self.n_components = n_components
        self.outlier_threshold = outlier_threshold
        
        # Fitted model
        self.gmm = None
        self.outlier_indices = None
        self.outlier_values = None
        self.original_shape = None
        self.zero_mask = None  # Track exact zeros separately
        
    def compress(self, weights: np.ndarray) -> Dict:
        """
        Compress weights using Gaussian mixture model.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Compression metadata
        """
        self.original_shape = weights.shape
        
        # Separate exact zeros (common in sparse networks)
        self.zero_mask = (weights == 0)
        non_zero_weights = weights[~self.zero_mask]
        
        if len(non_zero_weights) == 0:
            # All zeros - trivial compression
            self.gmm = None
            self.outlier_indices = np.array([])
            self.outlier_values = np.array([])
            
            return {
                'original_size': weights.size,
                'compressed_size': 0,
                'compression_ratio': float('inf'),
                'n_outliers': 0,
                'n_zeros': weights.size
            }
        
        # Fit Gaussian mixture to non-zero weights
        self.gmm = GaussianMixture(
            n_components=min(self.n_components, len(non_zero_weights)),
            covariance_type='full',
            random_state=42
        )
        
        # Reshape for sklearn (needs 2D)
        non_zero_reshaped = non_zero_weights.reshape(-1, 1)
        self.gmm.fit(non_zero_reshaped)
        
        # Identify outliers (weights far from any Gaussian)
        log_prob = self.gmm.score_samples(non_zero_reshaped)
        threshold = np.mean(log_prob) - self.outlier_threshold * np.std(log_prob)
        
        outlier_mask_nonzero = log_prob < threshold
        
        # Store outliers explicitly
        outlier_weights = non_zero_weights[outlier_mask_nonzero]
        
        # Get original indices of outliers
        non_zero_indices = np.where(~self.zero_mask)
        outlier_positions = np.where(outlier_mask_nonzero)[0]
        
        self.outlier_indices = tuple(
            non_zero_indices[i][outlier_positions] for i in range(len(non_zero_indices))
        )
        self.outlier_values = outlier_weights
        
        # Calculate compression
        # Storage: GMM params + outlier indices + outlier values + zero mask
        gmm_params_size = (
            self.gmm.n_components * 2 +  # means + variances
            self.gmm.n_components  # weights
        )
        outlier_size = len(self.outlier_values) * 2  # index + value
        zero_mask_size = np.sum(self.zero_mask) / 8  # bits -> bytes
        
        compressed_size = gmm_params_size + outlier_size + zero_mask_size
        compression_ratio = weights.size / max(1, compressed_size)
        
        return {
            'original_size': weights.size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'n_components': self.gmm.n_components,
            'n_outliers': len(self.outlier_values),
            'n_zeros': np.sum(self.zero_mask),
            'sparsity': np.sum(self.zero_mask) / weights.size
        }
    
    def decompress(self, deterministic: bool = True) -> np.ndarray:
        """
        Reconstruct weights from Gaussian mixture.
        
        Args:
            deterministic: If True, use means; if False, sample from distribution
            
        Returns:
            Reconstructed weights
        """
        if self.original_shape is None:
            raise ValueError("No compressed data available")
        
        # Start with zeros
        reconstructed = np.zeros(self.original_shape)
        
        if self.gmm is None:
            # All zeros case
            return reconstructed
        
        # Reconstruct non-zero weights
        non_zero_count = np.sum(~self.zero_mask)
        
        if deterministic:
            # Use component means weighted by probabilities
            # For each non-zero position, assign to most likely component
            non_zero_indices = np.where(~self.zero_mask)
            
            # Predict component for each position (deterministic assignment)
            # Use component means as values
            component_means = self.gmm.means_.flatten()
            component_weights = self.gmm.weights_
            
            # Assign each non-zero weight to weighted mean of components
            weighted_mean = np.sum(component_means * component_weights)
            reconstructed[non_zero_indices] = weighted_mean
            
        else:
            # Sample from mixture
            samples, _ = self.gmm.sample(non_zero_count)
            non_zero_indices = np.where(~self.zero_mask)
            reconstructed[non_zero_indices] = samples.flatten()
        
        # Restore outliers (exact values)
        if len(self.outlier_values) > 0:
            reconstructed[self.outlier_indices] = self.outlier_values
        
        return reconstructed
    
    def save(self, filepath: str):
        """Save compressed representation."""
        data = {
            'gmm': self.gmm,
            'outlier_indices': self.outlier_indices,
            'outlier_values': self.outlier_values,
            'original_shape': self.original_shape,
            'zero_mask': self.zero_mask,
            'n_components': self.n_components,
            'outlier_threshold': self.outlier_threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load compressed representation."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.gmm = data['gmm']
        self.outlier_indices = data['outlier_indices']
        self.outlier_values = data['outlier_values']
        self.original_shape = data['original_shape']
        self.zero_mask = data['zero_mask']
        self.n_components = data['n_components']
        self.outlier_threshold = data['outlier_threshold']


class SimpleGaussianCompressor:
    """
    Simplified version: single Gaussian + outliers.
    
    Faster and often sufficient for sparse networks.
    """
    
    def __init__(self, outlier_threshold: float = 2.0):
        self.outlier_threshold = outlier_threshold
        self.mean = None
        self.std = None
        self.outlier_indices = None
        self.outlier_values = None
        self.original_shape = None
        
    def compress(self, weights: np.ndarray) -> Dict:
        """Compress with single Gaussian."""
        self.original_shape = weights.shape
        
        # Calculate statistics
        self.mean = np.mean(weights)
        self.std = np.std(weights)
        
        # Identify outliers
        z_scores = np.abs((weights - self.mean) / (self.std + 1e-10))
        outlier_mask = z_scores > self.outlier_threshold
        
        self.outlier_indices = np.where(outlier_mask)
        self.outlier_values = weights[outlier_mask]
        
        # Compression: 2 params (mean, std) + outliers
        compressed_size = 2 + len(self.outlier_values) * 2
        
        return {
            'original_size': weights.size,
            'compressed_size': compressed_size,
            'compression_ratio': weights.size / compressed_size,
            'n_outliers': len(self.outlier_values)
        }
    
    def decompress(self) -> np.ndarray:
        """Reconstruct weights."""
        # Fill with mean
        reconstructed = np.full(self.original_shape, self.mean)
        
        # Restore outliers
        reconstructed[self.outlier_indices] = self.outlier_values
        
        return reconstructed


def benchmark_statistical_compression(weights: np.ndarray,
                                      n_components_list: List[int] = None,
                                      outlier_thresholds: List[float] = None) -> Dict:
    """
    Benchmark different statistical compression configurations.
    
    Args:
        weights: Weight matrix
        n_components_list: List of component counts to test
        outlier_thresholds: List of outlier thresholds to test
        
    Returns:
        Benchmark results
    """
    if n_components_list is None:
        n_components_list = [1, 2, 3, 5]
    
    if outlier_thresholds is None:
        outlier_thresholds = [2.0, 3.0, 4.0]
    
    results = {}
    
    for n_comp in n_components_list:
        for threshold in outlier_thresholds:
            compressor = GaussianStatisticalCompressor(
                n_components=n_comp,
                outlier_threshold=threshold
            )
            
            metadata = compressor.compress(weights)
            reconstructed = compressor.decompress(deterministic=True)
            
            # Calculate errors
            mse = np.mean((weights - reconstructed) ** 2)
            mae = np.mean(np.abs(weights - reconstructed))
            rel_error = mae / (np.abs(weights).mean() + 1e-10)
            
            config_name = f"n={n_comp}_thresh={threshold}"
            results[config_name] = {
                'n_components': n_comp,
                'outlier_threshold': threshold,
                'compression_ratio': metadata['compression_ratio'],
                'n_outliers': metadata['n_outliers'],
                'sparsity': metadata.get('sparsity', 0),
                'mse': mse,
                'mae': mae,
                'relative_error': rel_error
            }
    
    return results


def compare_statistical_vs_fourier(weights: np.ndarray) -> Dict:
    """
    Compare statistical compression vs Fourier methods.
    
    Args:
        weights: Weight matrix
        
    Returns:
        Comparison results
    """
    from .fourier_compression import FourierCompressor
    from .hybrid_compression import HybridCompressor
    
    results = {}
    
    # Fourier
    fourier = FourierCompressor(compression_ratio=100)
    fourier.compress(weights)
    fourier_recon = fourier.decompress()
    
    results['fourier'] = {
        'compression_ratio': 100,
        'error': np.mean(np.abs(weights - fourier_recon)) / (np.abs(weights).mean() + 1e-10)
    }
    
    # Hybrid
    hybrid = HybridCompressor(sparsity_threshold=0.01, fourier_ratio=50)
    hybrid_meta = hybrid.compress(weights)
    hybrid_recon = hybrid.decompress()
    
    results['hybrid'] = {
        'compression_ratio': hybrid_meta['compression_ratio'],
        'error': np.mean(np.abs(weights - hybrid_recon)) / (np.abs(weights).mean() + 1e-10)
    }
    
    # Statistical (Gaussian mixture)
    statistical = GaussianStatisticalCompressor(n_components=3, outlier_threshold=3.0)
    stat_meta = statistical.compress(weights)
    stat_recon = statistical.decompress(deterministic=True)
    
    results['statistical'] = {
        'compression_ratio': stat_meta['compression_ratio'],
        'error': np.mean(np.abs(weights - stat_recon)) / (np.abs(weights).mean() + 1e-10),
        'n_outliers': stat_meta['n_outliers']
    }
    
    # Simple Gaussian
    simple = SimpleGaussianCompressor(outlier_threshold=2.0)
    simple_meta = simple.compress(weights)
    simple_recon = simple.decompress()
    
    results['simple_gaussian'] = {
        'compression_ratio': simple_meta['compression_ratio'],
        'error': np.mean(np.abs(weights - simple_recon)) / (np.abs(weights).mean() + 1e-10),
        'n_outliers': simple_meta['n_outliers']
    }
    
    return results
