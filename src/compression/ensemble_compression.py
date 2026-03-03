"""
Dynamic ensemble compression with learned blending weights.

Key idea: Instead of choosing ONE method, use ALL methods and blend them
with learned weights that minimize reconstruction error.

This is like mixture-of-experts for compression - each method contributes
based on how well it handles the specific weight patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from scipy.optimize import minimize

from .fourier_compression import FourierCompressor
from .statistical_compression import GaussianStatisticalCompressor, SimpleGaussianCompressor
from .hybrid_compression import HybridCompressor


class DynamicEnsembleCompressor:
    """
    Ensemble compression with learned blending weights.
    
    Strategy:
    1. Compress with multiple methods (Fourier, Statistical, Hybrid)
    2. Learn optimal blending weights to minimize reconstruction error
    3. Reconstruct as weighted average of all methods
    
    Expected: Better than any single method!
    """
    
    def __init__(self, 
                 target_ratio: int = 100,
                 methods: List[str] = None):
        """
        Initialize ensemble compressor.
        
        Args:
            target_ratio: Target compression ratio
            methods: List of methods to ensemble ('fourier', 'statistical', 'hybrid')
        """
        self.target_ratio = target_ratio
        
        if methods is None:
            methods = ['fourier', 'statistical', 'hybrid']
        self.methods = methods
        
        # Compressors for each method
        self.compressors = {}
        
        # Learned blending weights (one per method)
        self.blend_weights = None
        
        # Original shape
        self.original_shape = None
        
    def compress(self, weights: np.ndarray, learn_weights: bool = True) -> Dict:
        """
        Compress with ensemble of methods.
        
        Args:
            weights: Weight matrix
            learn_weights: If True, learn optimal blending weights
            
        Returns:
            Compression metadata
        """
        self.original_shape = weights.shape
        
        # Compress with each method
        reconstructions = {}
        metadata_all = {}
        
        for method in self.methods:
            if method == 'fourier':
                compressor = FourierCompressor(compression_ratio=self.target_ratio)
                meta = compressor.compress(weights, method='adaptive')
                recon = compressor.decompress()
                
            elif method == 'statistical':
                # Choose GMM or Simple based on sparsity
                sparsity = np.sum(weights == 0) / weights.size
                
                if sparsity > 0.90 and np.count_nonzero(weights) > 50:
                    compressor = GaussianStatisticalCompressor(
                        n_components=3,
                        outlier_threshold=3.0
                    )
                    meta = compressor.compress(weights)
                    recon = compressor.decompress(deterministic=True)
                else:
                    compressor = SimpleGaussianCompressor(outlier_threshold=2.0)
                    meta = compressor.compress(weights)
                    recon = compressor.decompress()
                    
            elif method == 'hybrid':
                # Adaptive threshold
                near_zero = np.sum(np.abs(weights) < 0.01) / weights.size
                threshold = 0.01 if near_zero > 0.7 else 0.05
                
                compressor = HybridCompressor(
                    sparsity_threshold=threshold,
                    fourier_ratio=50
                )
                meta = compressor.compress(weights)
                recon = compressor.decompress()
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            self.compressors[method] = compressor
            reconstructions[method] = recon
            metadata_all[method] = meta
        
        # Learn optimal blending weights
        if learn_weights:
            self.blend_weights = self._learn_blend_weights(weights, reconstructions)
        else:
            # Uniform weights
            self.blend_weights = np.ones(len(self.methods)) / len(self.methods)
        
        # Calculate ensemble error
        ensemble_recon = self._blend_reconstructions(reconstructions, self.blend_weights)
        ensemble_error = np.mean(np.abs(weights - ensemble_recon)) / (np.abs(weights).mean() + 1e-10)
        
        # Calculate total compressed size (sum of all methods)
        total_compressed = sum(
            meta.get('compressed_size', meta.get('total_compressed', 0))
            for meta in metadata_all.values()
        )
        
        # Effective compression ratio (accounting for ensemble overhead)
        effective_ratio = weights.size / max(1, total_compressed)
        
        return {
            'original_size': weights.size,
            'total_compressed': total_compressed,
            'effective_ratio': effective_ratio,
            'ensemble_error': ensemble_error,
            'blend_weights': dict(zip(self.methods, self.blend_weights)),
            'individual_errors': {
                method: np.mean(np.abs(weights - recon)) / (np.abs(weights).mean() + 1e-10)
                for method, recon in reconstructions.items()
            },
            'methods_metadata': metadata_all
        }
    
    def _learn_blend_weights(self, 
                            original: np.ndarray,
                            reconstructions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Learn optimal blending weights to minimize reconstruction error.
        
        Uses gradient-free optimization (Nelder-Mead).
        """
        n_methods = len(self.methods)
        
        # Objective: minimize reconstruction error
        def objective(weights):
            # Ensure weights sum to 1 and are non-negative
            weights = np.abs(weights)
            weights = weights / (weights.sum() + 1e-10)
            
            # Blend reconstructions
            blended = self._blend_reconstructions(reconstructions, weights)
            
            # MSE
            error = np.mean((original - blended) ** 2)
            
            return error
        
        # Initial guess: uniform weights
        x0 = np.ones(n_methods) / n_methods
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-4}
        )
        
        # Normalize final weights
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / (optimal_weights.sum() + 1e-10)
        
        return optimal_weights
    
    def _blend_reconstructions(self,
                               reconstructions: Dict[str, np.ndarray],
                               weights: np.ndarray) -> np.ndarray:
        """Blend multiple reconstructions with given weights."""
        blended = np.zeros(self.original_shape)
        
        for i, method in enumerate(self.methods):
            blended += weights[i] * reconstructions[method]
        
        return blended
    
    def decompress(self) -> np.ndarray:
        """Reconstruct weights using ensemble."""
        if not self.compressors:
            raise ValueError("No compressed data available")
        
        # Decompress each method
        reconstructions = {}
        
        for method, compressor in self.compressors.items():
            if method == 'statistical':
                # Check if GMM or Simple
                if hasattr(compressor, 'gmm'):
                    recon = compressor.decompress(deterministic=True)
                else:
                    recon = compressor.decompress()
            else:
                recon = compressor.decompress()
            
            reconstructions[method] = recon
        
        # Blend with learned weights
        return self._blend_reconstructions(reconstructions, self.blend_weights)
    
    def save(self, filepath: str):
        """Save ensemble compressor."""
        data = {
            'compressors': self.compressors,
            'blend_weights': self.blend_weights,
            'methods': self.methods,
            'original_shape': self.original_shape,
            'target_ratio': self.target_ratio
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load ensemble compressor."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.compressors = data['compressors']
        self.blend_weights = data['blend_weights']
        self.methods = data['methods']
        self.original_shape = data['original_shape']
        self.target_ratio = data['target_ratio']


class RegionAdaptiveCompressor:
    """
    Apply different compression methods to different regions.
    
    Divides weight matrix into regions and compresses each optimally.
    Perfect for networks with mixed characteristics!
    """
    
    def __init__(self, 
                 region_size: int = 50,
                 target_ratio: int = 100):
        """
        Initialize region-adaptive compressor.
        
        Args:
            region_size: Size of each region (region_size x region_size)
            target_ratio: Target compression ratio
        """
        self.region_size = region_size
        self.target_ratio = target_ratio
        
        # Store compressor for each region
        self.region_compressors = {}
        self.region_methods = {}
        self.original_shape = None
        
    def compress(self, weights: np.ndarray) -> Dict:
        """Compress with region-adaptive methods."""
        self.original_shape = weights.shape
        h, w = weights.shape
        
        # Divide into regions
        n_regions_h = (h + self.region_size - 1) // self.region_size
        n_regions_w = (w + self.region_size - 1) // self.region_size
        
        total_compressed = 0
        region_stats = []
        
        for i in range(n_regions_h):
            for j in range(n_regions_w):
                # Extract region
                i_start = i * self.region_size
                i_end = min((i + 1) * self.region_size, h)
                j_start = j * self.region_size
                j_end = min((j + 1) * self.region_size, w)
                
                region = weights[i_start:i_end, j_start:j_end]
                
                # Analyze region and choose method
                sparsity = np.sum(region == 0) / region.size
                
                if sparsity > 0.90:
                    method = 'statistical'
                    compressor = SimpleGaussianCompressor(outlier_threshold=2.0)
                    meta = compressor.compress(region)
                else:
                    method = 'fourier'
                    compressor = FourierCompressor(compression_ratio=self.target_ratio)
                    meta = compressor.compress(region, method='adaptive')
                
                # Store
                region_key = (i, j)
                self.region_compressors[region_key] = compressor
                self.region_methods[region_key] = method
                
                total_compressed += meta.get('compressed_size', meta.get('total_compressed', 0))
                
                region_stats.append({
                    'region': region_key,
                    'method': method,
                    'sparsity': sparsity,
                    'size': region.size
                })
        
        effective_ratio = weights.size / max(1, total_compressed)
        
        return {
            'original_size': weights.size,
            'total_compressed': total_compressed,
            'effective_ratio': effective_ratio,
            'n_regions': len(self.region_compressors),
            'region_stats': region_stats
        }
    
    def decompress(self) -> np.ndarray:
        """Reconstruct from regions."""
        if not self.region_compressors:
            raise ValueError("No compressed data available")
        
        # Initialize output
        reconstructed = np.zeros(self.original_shape)
        
        # Decompress each region
        for (i, j), compressor in self.region_compressors.items():
            # Calculate region bounds
            i_start = i * self.region_size
            i_end = min((i + 1) * self.region_size, self.original_shape[0])
            j_start = j * self.region_size
            j_end = min((j + 1) * self.region_size, self.original_shape[1])
            
            # Decompress
            region_recon = compressor.decompress()
            
            # Place in output
            reconstructed[i_start:i_end, j_start:j_end] = region_recon
        
        return reconstructed


def benchmark_ensemble_compression(weights: np.ndarray) -> Dict:
    """
    Compare ensemble vs individual methods.
    
    Args:
        weights: Weight matrix
        
    Returns:
        Comparison results
    """
    results = {}
    
    # Individual methods
    from .adaptive_compression import AdaptiveCompressor
    
    adaptive = AdaptiveCompressor(target_ratio=100)
    adaptive_meta = adaptive.compress(weights)
    adaptive_recon = adaptive.decompress()
    
    results['adaptive'] = {
        'method': adaptive_meta['selected_method'],
        'error': np.mean(np.abs(weights - adaptive_recon)) / (np.abs(weights).mean() + 1e-10),
        'ratio': adaptive_meta['compression_ratio']
    }
    
    # Ensemble
    ensemble = DynamicEnsembleCompressor(target_ratio=100)
    ensemble_meta = ensemble.compress(weights, learn_weights=True)
    ensemble_recon = ensemble.decompress()
    
    results['ensemble'] = {
        'method': 'ensemble',
        'error': ensemble_meta['ensemble_error'],
        'ratio': ensemble_meta['effective_ratio'],
        'blend_weights': ensemble_meta['blend_weights'],
        'individual_errors': ensemble_meta['individual_errors']
    }
    
    # Region-adaptive
    region = RegionAdaptiveCompressor(region_size=50, target_ratio=100)
    region_meta = region.compress(weights)
    region_recon = region.decompress()
    
    results['region_adaptive'] = {
        'method': 'region_adaptive',
        'error': np.mean(np.abs(weights - region_recon)) / (np.abs(weights).mean() + 1e-10),
        'ratio': region_meta['effective_ratio'],
        'n_regions': region_meta['n_regions']
    }
    
    return results
