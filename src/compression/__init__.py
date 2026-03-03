"""
Compression module for Fourier-based neural network compression.
"""

from .fourier_compression import (
    FourierCompressor,
    compress_weights,
    decompress_weights,
    adaptive_frequency_selection
)
from .hybrid_compression import (
    HybridCompressor,
    AdaptiveHybridCompressor,
    benchmark_hybrid_compression,
    compare_compression_methods
)
from .perceptual_compression import (
    PerceptualCompressor,
    WeightedPerceptualCompressor,
    benchmark_perceptual_compression,
    compare_all_methods
)
from .statistical_compression import (
    GaussianStatisticalCompressor,
    SimpleGaussianCompressor,
    benchmark_statistical_compression,
    compare_statistical_vs_fourier
)

__all__ = [
    'FourierCompressor',
    'compress_weights',
    'decompress_weights',
    'adaptive_frequency_selection',
    'HybridCompressor',
    'AdaptiveHybridCompressor',
    'benchmark_hybrid_compression',
    'compare_compression_methods',
    'PerceptualCompressor',
    'WeightedPerceptualCompressor',
    'benchmark_perceptual_compression',
    'compare_all_methods',
    'GaussianStatisticalCompressor',
    'SimpleGaussianCompressor',
    'benchmark_statistical_compression',
    'compare_statistical_vs_fourier'
]
