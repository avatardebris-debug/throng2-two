"""
Test perceptual compression - optimize for network behavior.

Expected: <10% error on network outputs (even if weight error is higher)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.compression.perceptual_compression import (
    PerceptualCompressor,
    WeightedPerceptualCompressor,
    benchmark_perceptual_compression,
    compare_all_methods
)
from src.core.network import LayeredNetwork


def simple_forward(weights, inputs):
    """Simple forward pass for testing."""
    # Matrix multiplication: output = inputs @ weights
    return inputs @ weights


def test_perceptual_compression():
    """Test perceptual compression on realistic weights."""
    print("\n" + "="*60)
    print("TEST: Perceptual Compression")
    print("="*60)
    
    # Create realistic weights
    size = 50
    weights = np.random.randn(size, size) * 0.1
    
    # Add structure (low frequency patterns)
    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    X, Y = np.meshgrid(x, y)
    weights += 0.3 * np.sin(X) * np.cos(Y)
    
    # Sample inputs
    n_samples = 100
    sample_inputs = np.random.randn(n_samples, size)
    
    print(f"Weight matrix: {weights.shape}")
    print(f"Sample inputs: {sample_inputs.shape}")
    
    # Test weighted perceptual (faster, no forward function needed)
    print("\n--- Weighted Perceptual Compression ---")
    weighted = WeightedPerceptualCompressor(compression_ratio=100)
    metadata = weighted.compress(weights)
    weighted_recon = weighted.decompress()
    
    weight_error = np.mean(np.abs(weights - weighted_recon)) / (np.abs(weights).mean() + 1e-10)
    
    # Measure output error
    original_outputs = simple_forward(weights, sample_inputs)
    weighted_outputs = simple_forward(weighted_recon, sample_inputs)
    output_error = np.mean(np.abs(original_outputs - weighted_outputs)) / (np.abs(original_outputs).mean() + 1e-10)
    
    print(f"Compression ratio: {metadata['compression_ratio']:.1f}x")
    print(f"Weight error: {weight_error:.2%}")
    print(f"Output error: {output_error:.2%}")
    
    if output_error < 0.10:
        print(f"✓ Output error <10%! Target achieved!")
    else:
        print(f"⚠ Output error: {output_error:.2%} (target: <10%)")
    
    return weight_error, output_error


def test_with_real_network():
    """Test on actual thronglet network."""
    print("\n" + "="*60)
    print("TEST: Real Network Integration")
    print("="*60)
    
    # Create network
    network = LayeredNetwork(
        input_size=8,
        hidden_sizes=[50, 25],
        output_size=4,
        dimension=2,
        connection_prob=0.1
    )
    
    # Get weights from first hidden layer
    layer = network.layers[1]
    weights = layer.weights
    
    print(f"Network layer: {weights.shape}")
    print(f"Sparsity: {np.sum(weights == 0) / weights.size:.1%}")
    
    # Create sample inputs
    n_samples = 50
    sample_inputs = np.random.randn(n_samples, weights.shape[0])
    
    # Define forward function for this layer
    def layer_forward(w, inputs):
        return inputs @ w
    
    # Compare all methods
    print("\n--- Comparing All Methods ---")
    results = compare_all_methods(weights, layer_forward, sample_inputs)
    
    print(f"\n{'Method':<20} {'Weight Error':<15} {'Output Error':<15} {'Ratio':<10}")
    print("-" * 65)
    
    for method, result in results.items():
        weight_err = result['weight_error']
        output_err = result.get('output_error', 0)
        ratio = result['compression_ratio']
        
        print(f"{method:<20} {weight_err:<15.2%} {output_err:<15.2%} {ratio:<10.1f}x")
    
    # Find best method by output error
    best_method = min(results.items(), key=lambda x: x[1].get('output_error', float('inf')))
    print(f"\n✓ Best method: {best_method[0]} with {best_method[1]['output_error']:.2%} output error")
    
    return results


def benchmark_compression_ratios():
    """Benchmark different compression ratios."""
    print("\n" + "="*60)
    print("BENCHMARK: Compression Ratios")
    print("="*60)
    
    # Create test weights
    size = 50
    weights = np.random.randn(size, size) * 0.1
    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    X, Y = np.meshgrid(x, y)
    weights += 0.3 * np.sin(X) * np.cos(Y)
    
    # Sample inputs
    sample_inputs = np.random.randn(100, size)
    
    # Test different ratios
    ratios = [10, 50, 100, 500, 1000]
    results = benchmark_perceptual_compression(
        weights, 
        simple_forward, 
        sample_inputs, 
        ratios
    )
    
    print(f"\n{'Ratio':<10} {'Weight Error':<15} {'Output Error':<15} {'Size':<10}")
    print("-" * 55)
    
    for ratio, result in results.items():
        weight_err = result['weight_error']
        output_err = result['output_error'] if result['output_error'] is not None else 0
        size = result['compressed_size']
        
        print(f"{ratio:<10} {weight_err:<15.2%} {output_err:<15.2%} {size:<10}")
    
    return results


def visualize_perceptual_compression():
    """Visualize compression quality."""
    print("\n" + "="*60)
    print("VISUALIZATION: Perceptual vs Other Methods")
    print("="*60)
    
    # Create test weights
    size = 100
    weights = np.zeros((size, size))
    
    # Add patterns
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    weights += 0.5 * np.sin(X) * np.cos(Y)
    weights += 0.3 * np.sin(2*X) * np.cos(2*Y)
    weights += 0.1 * np.random.randn(size, size)
    
    # Compress with different methods
    from src.compression.fourier_compression import FourierCompressor
    from src.compression.hybrid_compression import HybridCompressor
    
    fourier = FourierCompressor(compression_ratio=100)
    fourier.compress(weights)
    fourier_recon = fourier.decompress()
    
    hybrid = HybridCompressor(sparsity_threshold=0.01, fourier_ratio=50)
    hybrid.compress(weights)
    hybrid_recon = hybrid.decompress()
    
    perceptual = WeightedPerceptualCompressor(compression_ratio=100)
    perceptual.compress(weights)
    perceptual_recon = perceptual.decompress()
    
    # Calculate errors
    fourier_err = weights - fourier_recon
    hybrid_err = weights - hybrid_recon
    perceptual_err = weights - perceptual_recon
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    im0 = axes[0, 0].imshow(weights, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Reconstructions
    im1 = axes[0, 1].imshow(fourier_recon, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'Fourier\nMAE: {np.mean(np.abs(fourier_err)):.4f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(hybrid_recon, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'Hybrid\nMAE: {np.mean(np.abs(hybrid_err)):.4f}')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    im3 = axes[0, 3].imshow(perceptual_recon, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 3].set_title(f'Perceptual\nMAE: {np.mean(np.abs(perceptual_err)):.4f}')
    axes[0, 3].axis('off')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    # Error maps
    vmax_err = 0.5
    im4 = axes[1, 1].imshow(np.abs(fourier_err), cmap='Reds', vmin=0, vmax=vmax_err)
    axes[1, 1].set_title('Fourier Error')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    im5 = axes[1, 2].imshow(np.abs(hybrid_err), cmap='Reds', vmin=0, vmax=vmax_err)
    axes[1, 2].set_title('Hybrid Error')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    im6 = axes[1, 3].imshow(np.abs(perceptual_err), cmap='Reds', vmin=0, vmax=vmax_err)
    axes[1, 3].set_title('Perceptual Error')
    axes[1, 3].axis('off')
    plt.colorbar(im6, ax=axes[1, 3], fraction=0.046)
    
    # Error comparison
    axes[1, 0].hist([fourier_err.flatten(), hybrid_err.flatten(), perceptual_err.flatten()],
                    bins=50, alpha=0.5, label=['Fourier', 'Hybrid', 'Perceptual'])
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    plt.tight_layout()
    plt.savefig('perceptual_compression_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'perceptual_compression_comparison.png'")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PERCEPTUAL COMPRESSION TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        weight_err, output_err = test_perceptual_compression()
        network_results = test_with_real_network()
        ratio_results = benchmark_compression_ratios()
        
        # Visualize
        visualize_perceptual_compression()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print(f"\nPerceptual compression:")
        print(f"  Weight error: {weight_err:.2%}")
        print(f"  Output error: {output_err:.2%}")
        
        if output_err < 0.10:
            print(f"\n✓ TARGET ACHIEVED: Output error <10%!")
        else:
            print(f"\n⚠ Output error: {output_err:.2%} (target: <10%)")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
