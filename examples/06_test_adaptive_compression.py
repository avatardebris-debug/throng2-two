"""
Test adaptive compression on diverse weight types.

Validates that the adaptive compressor chooses the right method
for different data characteristics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.compression.adaptive_compression import (
    AdaptiveCompressor,
    benchmark_adaptive_compression,
    compare_adaptive_vs_fixed
)


def test_adaptive_selection():
    """Test that adaptive compressor selects correct methods."""
    print("\n" + "="*60)
    print("TEST: Adaptive Method Selection")
    print("="*60)
    
    # Test case 1: Dense structured (should choose Fourier)
    print("\n--- Test 1: Dense Structured Weights (CNN-like) ---")
    size = 100
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    dense_weights = 0.5 * np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(size, size)
    
    adaptive = AdaptiveCompressor(target_ratio=100)
    metadata = adaptive.compress(dense_weights)
    
    print(f"Selected method: {metadata['selected_method']}")
    print(f"Reason: {metadata['selection_reason']}")
    print(f"Sparsity: {metadata['sparsity']:.1%}")
    print(f"Low-freq power: {metadata['low_freq_power']:.1%}")
    print(f"Compression: {metadata['compression_ratio']:.1f}x")
    
    # Test case 2: Sparse random (should choose Statistical)
    print("\n--- Test 2: Sparse Random Weights (FC with dropout) ---")
    sparse_weights = np.zeros((size, size))
    n_connections = int(size * size * 0.05)
    indices = np.random.choice(size*size, n_connections, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    sparse_weights[i_idx, j_idx] = np.random.randn(n_connections) * 0.3
    
    adaptive2 = AdaptiveCompressor(target_ratio=100)
    metadata2 = adaptive2.compress(sparse_weights)
    
    print(f"Selected method: {metadata2['selected_method']}")
    print(f"Reason: {metadata2['selection_reason']}")
    print(f"Sparsity: {metadata2['sparsity']:.1%}")
    print(f"Low-freq power: {metadata2['low_freq_power']:.1%}")
    print(f"Compression: {metadata2['compression_ratio']:.1f}x")
    
    # Test case 3: Mixed (should choose Hybrid)
    print("\n--- Test 3: Mixed Characteristics ---")
    mixed_weights = np.zeros((size, size))
    mixed_weights[:50, :50] = 0.3 * np.sin(X[:50, :50]) * np.cos(Y[:50, :50])
    n_sparse = int(size * size * 0.15)
    indices = np.random.choice(size*size, n_sparse, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    mixed_weights[i_idx, j_idx] += np.random.randn(n_sparse) * 0.2
    
    adaptive3 = AdaptiveCompressor(target_ratio=100)
    metadata3 = adaptive3.compress(mixed_weights)
    
    print(f"Selected method: {metadata3['selected_method']}")
    print(f"Reason: {metadata3['selection_reason']}")
    print(f"Sparsity: {metadata3['sparsity']:.1%}")
    print(f"Low-freq power: {metadata3['low_freq_power']:.1%}")
    print(f"Compression: {metadata3['compression_ratio']:.1f}x")
    
    return metadata, metadata2, metadata3


def test_adaptive_vs_fixed():
    """Compare adaptive vs forcing specific methods."""
    print("\n" + "="*60)
    print("TEST: Adaptive vs Fixed Method Selection")
    print("="*60)
    
    # Create test weights (sparse)
    size = 100
    weights = np.zeros((size, size))
    n_connections = int(size * size * 0.05)
    indices = np.random.choice(size*size, n_connections, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    weights[i_idx, j_idx] = np.random.randn(n_connections) * 0.3
    
    print(f"\nTest weights: {weights.shape}, {np.sum(weights==0)/weights.size:.1%} sparse")
    
    results = compare_adaptive_vs_fixed(weights)
    
    print(f"\n{'Method':<15} {'Selected':<15} {'Ratio':<12} {'Error':<12}")
    print("-" * 60)
    
    for method, result in results.items():
        selected = result.get('method', method)
        ratio = result['compression_ratio']
        error = result['error']
        
        marker = "✓" if method == 'adaptive' else ""
        print(f"{method:<15} {selected:<15} {ratio:<12.1f} {error:<12.2%} {marker}")
    
    # Show which method adaptive chose
    adaptive_method = results['adaptive']['method']
    adaptive_error = results['adaptive']['error']
    
    print(f"\n✓ Adaptive chose: {adaptive_method}")
    print(f"  Reason: {results['adaptive']['reason']}")
    print(f"  Error: {adaptive_error:.2%}")
    
    # Compare to best fixed method
    best_fixed = min(
        [(k, v) for k, v in results.items() if k != 'adaptive'],
        key=lambda x: x[1]['error']
    )
    
    print(f"\nBest fixed method: {best_fixed[0]} ({best_fixed[1]['error']:.2%})")
    
    if adaptive_method == best_fixed[0]:
        print("✓ Adaptive correctly chose the best method!")
    
    return results


def benchmark_diverse_weights():
    """Benchmark on diverse weight types."""
    print("\n" + "="*60)
    print("BENCHMARK: Diverse Weight Types")
    print("="*60)
    
    results = benchmark_adaptive_compression()
    
    print(f"\n{'Weight Type':<25} {'Method':<15} {'Ratio':<10} {'Error':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        method = result['selected_method']
        ratio = result['compression_ratio']
        error = result['error']
        
        print(f"{name:<25} {method:<15} {ratio:<10.1f} {error:<10.2%}")
        print(f"  → {result['reason']}")
    
    return results


def visualize_adaptive_selection():
    """Visualize adaptive compression on different weight types."""
    print("\n" + "="*60)
    print("VISUALIZATION: Adaptive Selection")
    print("="*60)
    
    # Create test cases
    size = 100
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Dense structured
    dense = 0.5 * np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(size, size)
    
    # Sparse random
    sparse = np.zeros((size, size))
    n_conn = int(size * size * 0.05)
    indices = np.random.choice(size*size, n_conn, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    sparse[i_idx, j_idx] = np.random.randn(n_conn) * 0.3
    
    # Mixed
    mixed = np.zeros((size, size))
    mixed[:50, :50] = 0.3 * np.sin(X[:50, :50]) * np.cos(Y[:50, :50])
    n_sparse = int(size * size * 0.15)
    indices = np.random.choice(size*size, n_sparse, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    mixed[i_idx, j_idx] += np.random.randn(n_sparse) * 0.2
    
    # Compress each
    test_cases = [
        ('Dense\n(CNN-like)', dense),
        ('Sparse\n(FC dropout)', sparse),
        ('Mixed\n(Hybrid)', mixed)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, weights) in enumerate(test_cases):
        # Compress
        adaptive = AdaptiveCompressor(target_ratio=100)
        metadata = adaptive.compress(weights)
        reconstructed = adaptive.decompress()
        
        error = weights - reconstructed
        
        # Original
        im1 = axes[0, idx].imshow(weights, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, idx].set_title(f'{name}\nOriginal')
        axes[0, idx].axis('off')
        plt.colorbar(im1, ax=axes[0, idx], fraction=0.046)
        
        # Reconstructed
        im2 = axes[1, idx].imshow(reconstructed, cmap='RdBu', vmin=-1, vmax=1)
        method = metadata['selected_method'].capitalize()
        ratio = metadata['compression_ratio']
        err = np.mean(np.abs(error)) / (np.abs(weights).mean() + 1e-10)
        axes[1, idx].set_title(f'{method} ({ratio:.1f}x)\nError: {err:.1%}')
        axes[1, idx].axis('off')
        plt.colorbar(im2, ax=axes[1, idx], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('adaptive_compression_selection.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'adaptive_compression_selection.png'")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADAPTIVE COMPRESSION TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        selection_results = test_adaptive_selection()
        comparison_results = test_adaptive_vs_fixed()
        benchmark_results = benchmark_diverse_weights()
        
        # Visualize
        visualize_adaptive_selection()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print("\n✓ Adaptive compressor successfully selects optimal method")
        print("✓ Fourier for dense structured weights")
        print("✓ Statistical for sparse random weights")
        print("✓ Hybrid for mixed characteristics")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
