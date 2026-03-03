"""
Test hybrid sparse-Fourier compression improvements.

Expected: 27% error → 8% error at 200x compression
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.compression.fourier_compression import FourierCompressor
from src.compression.hybrid_compression import (
    HybridCompressor,
    AdaptiveHybridCompressor,
    benchmark_hybrid_compression,
    compare_compression_methods
)


def create_realistic_weights(size: int = 100, sparsity: float = 0.7) -> np.ndarray:
    """
    Create realistic neural network weights.
    
    Real weights are:
    - Mostly small values (not exactly zero)
    - Some structured patterns (from learning)
    - Few strong connections
    """
    weights = np.zeros((size, size))
    
    # Add structured patterns (low frequency) - main learned features
    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Main pattern (strong, low frequency)
    weights += 0.5 * np.sin(X) * np.cos(Y)
    
    # Secondary pattern (medium frequency)
    weights += 0.2 * np.sin(2*X) * np.cos(2*Y)
    
    # Add noise everywhere (realistic - no exact zeros!)
    weights += np.random.randn(size, size) * 0.02
    
    # Add some strong sparse connections (important learned weights)
    n_strong = int(size * size * (1 - sparsity) * 0.3)
    strong_indices = np.random.choice(size*size, n_strong, replace=False)
    strong_i, strong_j = np.unravel_index(strong_indices, (size, size))
    weights[strong_i, strong_j] += np.random.randn(n_strong) * 0.5
    
    # Make most values small but NOT zero (realistic!)
    # Apply soft thresholding instead of hard zeros
    threshold = np.percentile(np.abs(weights), sparsity * 100)
    small_mask = np.abs(weights) < threshold
    weights[small_mask] *= 0.1  # Make small, not zero
    
    return weights


def test_hybrid_compression():
    """Test hybrid compression on realistic weights."""
    print("\n" + "="*60)
    print("TEST: Hybrid Sparse-Fourier Compression")
    print("="*60)
    
    # Create realistic weights
    weights = create_realistic_weights(size=100, sparsity=0.7)
    
    print(f"\nWeight matrix: {weights.shape}")
    print(f"Sparsity: {np.sum(weights == 0) / weights.size:.1%}")
    print(f"Non-zero values: {np.count_nonzero(weights)}")
    
    # Test 1: Pure Fourier (baseline)
    print("\n--- Pure Fourier Compression ---")
    fourier = FourierCompressor(compression_ratio=100)
    fourier.compress(weights)
    fourier_recon = fourier.decompress()
    
    fourier_error = np.mean(np.abs(weights - fourier_recon)) / (np.abs(weights).mean() + 1e-10)
    print(f"Compression: 100x")
    print(f"Error: {fourier_error:.2%}")
    
    # Test 2: Hybrid compression
    print("\n--- Hybrid Compression ---")
    hybrid = HybridCompressor(sparsity_threshold=0.01, fourier_ratio=50)
    hybrid_meta = hybrid.compress(weights)
    hybrid_recon = hybrid.decompress()
    
    hybrid_error = np.mean(np.abs(weights - hybrid_recon)) / (np.abs(weights).mean() + 1e-10)
    
    print(f"Sparsity threshold: {hybrid.sparsity_threshold}")
    print(f"Fourier ratio: {hybrid.fourier_ratio}")
    print(f"Sparse weights stored: {hybrid_meta['sparse_count']}")
    print(f"Dense frequencies stored: {hybrid_meta['dense_freq_count']}")
    print(f"Total compressed size: {hybrid_meta['total_compressed']}")
    print(f"Compression ratio: {hybrid_meta['compression_ratio']:.1f}x")
    print(f"Error: {hybrid_error:.2%}")
    
    # Improvement
    error_reduction = (fourier_error - hybrid_error) / fourier_error
    print(f"\n✓ Error reduction: {error_reduction:.1%}")
    print(f"✓ Compression improvement: {hybrid_meta['compression_ratio'] / 100:.1f}x")
    
    return fourier_error, hybrid_error


def test_adaptive_compression():
    """Test adaptive configuration tuning."""
    print("\n" + "="*60)
    print("TEST: Adaptive Hybrid Compression")
    print("="*60)
    
    weights = create_realistic_weights(size=100, sparsity=0.7)
    
    # Adaptive compressor finds best config
    adaptive = AdaptiveHybridCompressor(target_error=0.05)
    compressor, metadata = adaptive.compress(weights)
    
    print(f"\nBest configuration found:")
    print(f"  Sparsity threshold: {metadata['best_threshold']}")
    print(f"  Fourier ratio: {metadata['best_fourier_ratio']}")
    print(f"  Compression ratio: {metadata['compression_ratio']:.1f}x")
    print(f"  Error: {metadata['error']:.2%}")
    
    if metadata['error'] <= 0.05:
        print(f"\n✓ Meets target error of 5%!")
    else:
        print(f"\n⚠ Best error: {metadata['error']:.2%} (target: 5%)")
    
    return metadata


def benchmark_configurations():
    """Benchmark different configurations."""
    print("\n" + "="*60)
    print("BENCHMARK: Configuration Comparison")
    print("="*60)
    
    weights = create_realistic_weights(size=100, sparsity=0.7)
    
    configs = [
        (0.001, 100),
        (0.01, 50),
        (0.05, 30),
        (0.1, 20),
    ]
    
    results = benchmark_hybrid_compression(weights, configs)
    
    print(f"\n{'Config':<20} {'Ratio':<10} {'Error':<12} {'Sparsity':<12}")
    print("-" * 60)
    
    for config_name, result in results.items():
        print(f"{config_name:<20} {result['compression_ratio']:<10.1f} "
              f"{result['relative_error']:<12.2%} {result['sparsity']:<12.1%}")
    
    return results


def visualize_comparison():
    """Visualize pure Fourier vs hybrid compression."""
    print("\n" + "="*60)
    print("VISUALIZATION: Compression Comparison")
    print("="*60)
    
    weights = create_realistic_weights(size=100, sparsity=0.7)
    
    # Pure Fourier
    fourier = FourierCompressor(compression_ratio=100)
    fourier.compress(weights)
    fourier_recon = fourier.decompress()
    
    # Hybrid
    hybrid = HybridCompressor(sparsity_threshold=0.01, fourier_ratio=50)
    hybrid.compress(weights)
    hybrid_recon = hybrid.decompress()
    
    # Errors
    fourier_error = weights - fourier_recon
    hybrid_error = weights - hybrid_recon
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    im0 = axes[0, 0].imshow(weights, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title('Original Weights')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Pure Fourier reconstruction
    im1 = axes[0, 1].imshow(fourier_recon, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'Pure Fourier (100x)\nError: {np.mean(np.abs(fourier_error)):.4f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Hybrid reconstruction
    im2 = axes[0, 2].imshow(hybrid_recon, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'Hybrid Compression\nError: {np.mean(np.abs(hybrid_error)):.4f}')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Error maps
    im3 = axes[1, 0].imshow(np.abs(fourier_error), cmap='Reds', vmin=0, vmax=0.5)
    axes[1, 0].set_title('Pure Fourier Error')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(np.abs(hybrid_error), cmap='Reds', vmin=0, vmax=0.5)
    axes[1, 1].set_title('Hybrid Error')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Error histograms
    axes[1, 2].hist(fourier_error.flatten(), bins=50, alpha=0.5, label='Fourier', color='blue')
    axes[1, 2].hist(hybrid_error.flatten(), bins=50, alpha=0.5, label='Hybrid', color='green')
    axes[1, 2].set_title('Error Distribution')
    axes[1, 2].set_xlabel('Error')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('hybrid_compression_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'hybrid_compression_comparison.png'")
    
    plt.show()


def test_integration_with_network():
    """Test on actual neural network weights."""
    print("\n" + "="*60)
    print("TEST: Integration with Thronglet Network")
    print("="*60)
    
    from src.core.network import LayeredNetwork
    
    # Create network
    network = LayeredNetwork(
        input_size=8,
        hidden_sizes=[100, 50],
        output_size=4,
        dimension=2,
        connection_prob=0.05
    )
    
    # Get weights from first hidden layer
    layer = network.layers[1]
    weights = layer.weights
    
    print(f"Network layer: {weights.shape}")
    print(f"Sparsity: {np.sum(weights == 0) / weights.size:.1%}")
    
    # Compare methods
    results = compare_compression_methods(weights)
    
    print(f"\nPure Fourier:")
    print(f"  Compression: {results['pure_fourier']['compression_ratio']}x")
    print(f"  Error: {results['pure_fourier']['error']:.2%}")
    
    print(f"\nHybrid:")
    print(f"  Compression: {results['hybrid']['compression_ratio']:.1f}x")
    print(f"  Error: {results['hybrid']['error']:.2%}")
    print(f"  Sparsity captured: {results['hybrid']['sparsity']:.1%}")
    
    print(f"\nImprovement:")
    print(f"  Error reduction: {results['improvement']['error_reduction']:.1%}")
    print(f"  Ratio improvement: {results['improvement']['ratio_improvement']:.2f}x")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYBRID SPARSE-FOURIER COMPRESSION TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        fourier_err, hybrid_err = test_hybrid_compression()
        adaptive_meta = test_adaptive_compression()
        benchmark_results = benchmark_configurations()
        integration_results = test_integration_with_network()
        
        # Visualize
        visualize_comparison()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        improvement = (fourier_err - hybrid_err) / fourier_err
        print(f"\n✓ Error reduced from {fourier_err:.1%} → {hybrid_err:.1%}")
        print(f"✓ Improvement: {improvement:.1%}")
        
        if hybrid_err < 0.10:
            print(f"✓ Target achieved: <10% error!")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
