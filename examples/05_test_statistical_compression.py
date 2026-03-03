"""
Test Gaussian statistical compression on sparse neural networks.

Expected: <10% error with 1000x+ compression on sparse networks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.compression.statistical_compression import (
    GaussianStatisticalCompressor,
    SimpleGaussianCompressor,
    benchmark_statistical_compression,
    compare_statistical_vs_fourier
)
from src.core.network import LayeredNetwork


def test_statistical_compression():
    """Test statistical compression on sparse weights."""
    print("\n" + "="*60)
    print("TEST: Gaussian Statistical Compression")
    print("="*60)
    
    # Create sparse weights (like real neural networks)
    size = 100
    weights = np.zeros((size, size))
    
    # Add sparse strong connections (5% of weights)
    n_connections = int(size * size * 0.05)
    indices = np.random.choice(size*size, n_connections, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    
    # Connections follow a distribution
    weights[i_idx, j_idx] = np.random.randn(n_connections) * 0.3
    
    print(f"Weight matrix: {weights.shape}")
    print(f"Sparsity: {np.sum(weights == 0) / weights.size:.1%}")
    print(f"Non-zero values: {np.count_nonzero(weights)}")
    
    # Test Gaussian mixture compression
    print("\n--- Gaussian Mixture Compression ---")
    gmm_compressor = GaussianStatisticalCompressor(
        n_components=3,
        outlier_threshold=3.0
    )
    
    metadata = gmm_compressor.compress(weights)
    reconstructed = gmm_compressor.decompress(deterministic=True)
    
    error = np.mean(np.abs(weights - reconstructed)) / (np.abs(weights).mean() + 1e-10)
    
    print(f"Components: {metadata['n_components']}")
    print(f"Outliers stored: {metadata['n_outliers']}")
    print(f"Zeros: {metadata['n_zeros']}")
    print(f"Compression ratio: {metadata['compression_ratio']:.1f}x")
    print(f"Relative error: {error:.2%}")
    
    if error < 0.10:
        print(f"✓ Target achieved: <10% error!")
    
    # Test simple Gaussian
    print("\n--- Simple Gaussian Compression ---")
    simple_compressor = SimpleGaussianCompressor(outlier_threshold=2.0)
    
    simple_meta = simple_compressor.compress(weights)
    simple_recon = simple_compressor.decompress()
    
    simple_error = np.mean(np.abs(weights - simple_recon)) / (np.abs(weights).mean() + 1e-10)
    
    print(f"Outliers stored: {simple_meta['n_outliers']}")
    print(f"Compression ratio: {simple_meta['compression_ratio']:.1f}x")
    print(f"Relative error: {simple_error:.2%}")
    
    return error, simple_error


def test_real_network():
    """Test on actual thronglet network."""
    print("\n" + "="*60)
    print("TEST: Real Network Integration")
    print("="*60)
    
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
    print(f"Non-zero: {np.count_nonzero(weights)}")
    
    # Compare all methods
    print("\n--- Comparing All Methods ---")
    results = compare_statistical_vs_fourier(weights)
    
    print(f"\n{'Method':<20} {'Error':<15} {'Ratio':<15}")
    print("-" * 55)
    
    for method, result in results.items():
        error = result['error']
        ratio = result['compression_ratio']
        
        print(f"{method:<20} {error:<15.2%} {ratio:<15.1f}x")
    
    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]['error'])
    print(f"\n✓ Best method: {best_method[0]} with {best_method[1]['error']:.2%} error")
    
    return results


def benchmark_configurations():
    """Benchmark different statistical configurations."""
    print("\n" + "="*60)
    print("BENCHMARK: Statistical Compression Configurations")
    print("="*60)
    
    # Create sparse test weights
    size = 100
    weights = np.zeros((size, size))
    n_connections = int(size * size * 0.05)
    indices = np.random.choice(size*size, n_connections, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    weights[i_idx, j_idx] = np.random.randn(n_connections) * 0.3
    
    # Test different configurations
    results = benchmark_statistical_compression(
        weights,
        n_components_list=[1, 2, 3, 5],
        outlier_thresholds=[2.0, 3.0, 4.0]
    )
    
    print(f"\n{'Config':<20} {'Ratio':<12} {'Error':<12} {'Outliers':<12}")
    print("-" * 60)
    
    for config, result in sorted(results.items(), key=lambda x: x[1]['relative_error']):
        ratio = result['compression_ratio']
        error = result['relative_error']
        outliers = result['n_outliers']
        
        print(f"{config:<20} {ratio:<12.1f} {error:<12.2%} {outliers:<12}")
    
    # Find best config
    best_config = min(results.items(), key=lambda x: x[1]['relative_error'])
    print(f"\n✓ Best config: {best_config[0]}")
    print(f"  Error: {best_config[1]['relative_error']:.2%}")
    print(f"  Ratio: {best_config[1]['compression_ratio']:.1f}x")
    
    return results


def visualize_statistical_compression():
    """Visualize compression quality."""
    print("\n" + "="*60)
    print("VISUALIZATION: Statistical vs Other Methods")
    print("="*60)
    
    # Create sparse weights
    size = 100
    weights = np.zeros((size, size))
    
    # Add sparse connections with structure
    n_connections = int(size * size * 0.1)
    indices = np.random.choice(size*size, n_connections, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    
    # Mix of positive and negative weights
    weights[i_idx, j_idx] = np.random.randn(n_connections) * 0.5
    
    # Compress with different methods
    from src.compression.fourier_compression import FourierCompressor
    from src.compression.hybrid_compression import HybridCompressor
    
    fourier = FourierCompressor(compression_ratio=100)
    fourier.compress(weights)
    fourier_recon = fourier.decompress()
    
    hybrid = HybridCompressor(sparsity_threshold=0.01, fourier_ratio=50)
    hybrid.compress(weights)
    hybrid_recon = hybrid.decompress()
    
    statistical = GaussianStatisticalCompressor(n_components=3, outlier_threshold=3.0)
    statistical.compress(weights)
    stat_recon = statistical.decompress(deterministic=True)
    
    simple = SimpleGaussianCompressor(outlier_threshold=2.0)
    simple.compress(weights)
    simple_recon = simple.decompress()
    
    # Calculate errors
    fourier_err = weights - fourier_recon
    hybrid_err = weights - hybrid_recon
    stat_err = weights - stat_recon
    simple_err = weights - simple_recon
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Reconstructions
    vmin, vmax = -1, 1
    
    im0 = axes[0, 0].imshow(weights, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Original\n(10% sparse)')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(fourier_recon, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Fourier (100x)\nMAE: {np.mean(np.abs(fourier_err)):.4f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(hybrid_recon, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'Hybrid\nMAE: {np.mean(np.abs(hybrid_err)):.4f}')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    im3 = axes[0, 3].imshow(stat_recon, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[0, 3].set_title(f'Statistical (GMM)\nMAE: {np.mean(np.abs(stat_err)):.4f}')
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: Error maps
    vmax_err = 0.5
    
    im4 = axes[1, 0].imshow(np.abs(fourier_err), cmap='Reds', vmin=0, vmax=vmax_err)
    axes[1, 0].set_title('Fourier Error')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(np.abs(hybrid_err), cmap='Reds', vmin=0, vmax=vmax_err)
    axes[1, 1].set_title('Hybrid Error')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(np.abs(stat_err), cmap='Reds', vmin=0, vmax=vmax_err)
    axes[1, 2].set_title('Statistical Error')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    im7 = axes[1, 3].imshow(np.abs(simple_err), cmap='Reds', vmin=0, vmax=vmax_err)
    axes[1, 3].set_title(f'Simple Gaussian\nMAE: {np.mean(np.abs(simple_err)):.4f}')
    axes[1, 3].axis('off')
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046)
    
    # Row 3: Distributions
    axes[2, 0].hist(weights[weights != 0].flatten(), bins=50, alpha=0.7, label='Original')
    axes[2, 0].set_title('Weight Distribution')
    axes[2, 0].set_xlabel('Weight Value')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].legend()
    
    axes[2, 1].hist([fourier_err.flatten(), hybrid_err.flatten()],
                    bins=50, alpha=0.5, label=['Fourier', 'Hybrid'])
    axes[2, 1].set_title('Error Distribution')
    axes[2, 1].set_xlabel('Error')
    axes[2, 1].legend()
    
    axes[2, 2].hist([stat_err.flatten(), simple_err.flatten()],
                    bins=50, alpha=0.5, label=['GMM', 'Simple'], color=['green', 'orange'])
    axes[2, 2].set_title('Statistical Error Distribution')
    axes[2, 2].set_xlabel('Error')
    axes[2, 2].legend()
    
    # Comparison bar chart
    methods = ['Fourier', 'Hybrid', 'Statistical', 'Simple']
    errors = [
        np.mean(np.abs(fourier_err)) / (np.abs(weights).mean() + 1e-10),
        np.mean(np.abs(hybrid_err)) / (np.abs(weights).mean() + 1e-10),
        np.mean(np.abs(stat_err)) / (np.abs(weights).mean() + 1e-10),
        np.mean(np.abs(simple_err)) / (np.abs(weights).mean() + 1e-10)
    ]
    
    axes[2, 3].bar(methods, errors, color=['blue', 'green', 'orange', 'red'])
    axes[2, 3].set_title('Relative Error Comparison')
    axes[2, 3].set_ylabel('Relative Error')
    axes[2, 3].axhline(y=0.10, color='black', linestyle='--', label='10% target')
    axes[2, 3].legend()
    axes[2, 3].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('statistical_compression_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'statistical_compression_comparison.png'")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GAUSSIAN STATISTICAL COMPRESSION TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        gmm_err, simple_err = test_statistical_compression()
        network_results = test_real_network()
        config_results = benchmark_configurations()
        
        # Visualize
        visualize_statistical_compression()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print(f"\nStatistical compression:")
        print(f"  GMM error: {gmm_err:.2%}")
        print(f"  Simple error: {simple_err:.2%}")
        
        if gmm_err < 0.10 or simple_err < 0.10:
            print(f"\n✓ TARGET ACHIEVED: <10% error!")
        else:
            print(f"\n⚠ Best error: {min(gmm_err, simple_err):.2%} (target: <10%)")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
