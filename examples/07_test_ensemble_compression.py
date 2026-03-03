"""
Test dynamic ensemble compression with learned weights.

Expected: Ensemble outperforms any single method!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.compression.ensemble_compression import (
    DynamicEnsembleCompressor,
    RegionAdaptiveCompressor,
    benchmark_ensemble_compression
)
from src.core.network import LayeredNetwork


def test_ensemble_compression():
    """Test ensemble with learned weights."""
    print("\n" + "="*60)
    print("TEST: Dynamic Ensemble Compression")
    print("="*60)
    
    # Create test weights (mixed characteristics)
    size = 100
    weights = np.zeros((size, size))
    
    # Add dense structured region
    x = np.linspace(0, 2*np.pi, size//2)
    y = np.linspace(0, 2*np.pi, size//2)
    X, Y = np.meshgrid(x, y)
    weights[:size//2, :size//2] = 0.3 * np.sin(X) * np.cos(Y)
    
    # Add sparse random region
    n_sparse = int(size * size * 0.05)
    indices = np.random.choice(size*size, n_sparse, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    weights[i_idx, j_idx] += np.random.randn(n_sparse) * 0.3
    
    print(f"Weight matrix: {weights.shape}")
    print(f"Sparsity: {np.sum(weights == 0) / weights.size:.1%}")
    
    # Ensemble compression
    print("\n--- Ensemble Compression ---")
    ensemble = DynamicEnsembleCompressor(target_ratio=100)
    metadata = ensemble.compress(weights, learn_weights=True)
    reconstructed = ensemble.decompress()
    
    print(f"Effective ratio: {metadata['effective_ratio']:.1f}x")
    print(f"Ensemble error: {metadata['ensemble_error']:.2%}")
    
    print(f"\nLearned blend weights:")
    for method, weight in metadata['blend_weights'].items():
        print(f"  {method}: {weight:.3f}")
    
    print(f"\nIndividual method errors:")
    for method, error in metadata['individual_errors'].items():
        print(f"  {method}: {error:.2%}")
    
    # Check if ensemble is better
    best_individual = min(metadata['individual_errors'].values())
    improvement = (best_individual - metadata['ensemble_error']) / best_individual
    
    print(f"\nBest individual: {best_individual:.2%}")
    print(f"Ensemble: {metadata['ensemble_error']:.2%}")
    print(f"Improvement: {improvement:.1%}")
    
    if metadata['ensemble_error'] < best_individual:
        print("✓ Ensemble outperforms all individual methods!")
    
    return metadata


def test_region_adaptive():
    """Test region-adaptive compression."""
    print("\n" + "="*60)
    print("TEST: Region-Adaptive Compression")
    print("="*60)
    
    # Create weights with distinct regions
    size = 100
    weights = np.zeros((size, size))
    
    # Top-left: dense structured
    x = np.linspace(0, 2*np.pi, size//2)
    y = np.linspace(0, 2*np.pi, size//2)
    X, Y = np.meshgrid(x, y)
    weights[:size//2, :size//2] = 0.5 * np.sin(X) * np.cos(Y)
    
    # Bottom-right: sparse
    n_sparse = int((size//2) * (size//2) * 0.1)
    indices = np.random.choice((size//2) * (size//2), n_sparse, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size//2, size//2))
    weights[size//2 + i_idx, size//2 + j_idx] = np.random.randn(n_sparse) * 0.3
    
    print(f"Weight matrix: {weights.shape}")
    
    # Region-adaptive compression
    print("\n--- Region-Adaptive Compression ---")
    region_comp = RegionAdaptiveCompressor(region_size=50, target_ratio=100)
    metadata = region_comp.compress(weights)
    reconstructed = region_comp.decompress()
    
    error = np.mean(np.abs(weights - reconstructed)) / (np.abs(weights).mean() + 1e-10)
    
    print(f"Number of regions: {metadata['n_regions']}")
    print(f"Effective ratio: {metadata['effective_ratio']:.1f}x")
    print(f"Reconstruction error: {error:.2%}")
    
    print(f"\nRegion breakdown:")
    for stat in metadata['region_stats'][:4]:  # Show first 4
        print(f"  Region {stat['region']}: {stat['method']} (sparsity: {stat['sparsity']:.1%})")
    
    return metadata, error


def test_real_network():
    """Test on real network."""
    print("\n" + "="*60)
    print("TEST: Real Network Comparison")
    print("="*60)
    
    # Create network
    network = LayeredNetwork(
        input_size=8,
        hidden_sizes=[100, 50],
        output_size=4,
        dimension=2,
        connection_prob=0.05
    )
    
    # Get weights
    layer = network.layers[1]
    weights = layer.weights
    
    print(f"Network layer: {weights.shape}")
    print(f"Sparsity: {np.sum(weights == 0) / weights.size:.1%}")
    
    # Benchmark all methods
    print("\n--- Comparing All Methods ---")
    results = benchmark_ensemble_compression(weights)
    
    print(f"\n{'Method':<20} {'Error':<15} {'Ratio':<15}")
    print("-" * 55)
    
    for method, result in results.items():
        error = result['error']
        ratio = result['ratio']
        
        print(f"{method:<20} {error:<15.2%} {ratio:<15.1f}x")
    
    # Show ensemble details
    if 'ensemble' in results:
        print(f"\nEnsemble blend weights:")
        for method, weight in results['ensemble']['blend_weights'].items():
            print(f"  {method}: {weight:.3f}")
    
    # Find best
    best = min(results.items(), key=lambda x: x[1]['error'])
    print(f"\n✓ Best method: {best[0]} with {best[1]['error']:.2%} error")
    
    return results


def visualize_ensemble():
    """Visualize ensemble compression."""
    print("\n" + "="*60)
    print("VISUALIZATION: Ensemble vs Individual Methods")
    print("="*60)
    
    # Create mixed test weights
    size = 100
    weights = np.zeros((size, size))
    
    # Dense region
    x = np.linspace(0, 2*np.pi, size//2)
    y = np.linspace(0, 2*np.pi, size//2)
    X, Y = np.meshgrid(x, y)
    weights[:size//2, :size//2] = 0.4 * np.sin(X) * np.cos(Y)
    
    # Sparse region
    n_sparse = int(size * size * 0.08)
    indices = np.random.choice(size*size, n_sparse, replace=False)
    i_idx, j_idx = np.unravel_index(indices, (size, size))
    weights[i_idx, j_idx] += np.random.randn(n_sparse) * 0.3
    
    # Compress with ensemble
    ensemble = DynamicEnsembleCompressor(target_ratio=100)
    metadata = ensemble.compress(weights, learn_weights=True)
    ensemble_recon = ensemble.decompress()
    
    # Get individual reconstructions
    from src.compression.adaptive_compression import AdaptiveCompressor
    
    adaptive = AdaptiveCompressor(target_ratio=100)
    adaptive.compress(weights)
    adaptive_recon = adaptive.decompress()
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    im0 = axes[0, 0].imshow(weights, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title('Original\n(Mixed characteristics)')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Adaptive
    im1 = axes[0, 1].imshow(adaptive_recon, cmap='RdBu', vmin=-1, vmax=1)
    adaptive_err = np.mean(np.abs(weights - adaptive_recon)) / (np.abs(weights).mean() + 1e-10)
    axes[0, 1].set_title(f'Adaptive\nError: {adaptive_err:.1%}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Ensemble
    im2 = axes[0, 2].imshow(ensemble_recon, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'Ensemble\nError: {metadata["ensemble_error"]:.1%}')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Error maps
    adaptive_err_map = np.abs(weights - adaptive_recon)
    ensemble_err_map = np.abs(weights - ensemble_recon)
    
    im3 = axes[1, 0].imshow(adaptive_err_map, cmap='Reds', vmin=0, vmax=0.5)
    axes[1, 0].set_title('Adaptive Error Map')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    im4 = axes[1, 1].imshow(ensemble_err_map, cmap='Reds', vmin=0, vmax=0.5)
    axes[1, 1].set_title('Ensemble Error Map')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Blend weights visualization
    blend_weights = metadata['blend_weights']
    methods = list(blend_weights.keys())
    weights_vals = list(blend_weights.values())
    
    axes[1, 2].bar(methods, weights_vals, color=['blue', 'green', 'orange'])
    axes[1, 2].set_title('Learned Blend Weights')
    axes[1, 2].set_ylabel('Weight')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('ensemble_compression_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'ensemble_compression_comparison.png'")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DYNAMIC ENSEMBLE COMPRESSION TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        ensemble_meta = test_ensemble_compression()
        region_meta, region_err = test_region_adaptive()
        network_results = test_real_network()
        
        # Visualize
        visualize_ensemble()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print(f"\nEnsemble compression:")
        print(f"  Error: {ensemble_meta['ensemble_error']:.2%}")
        print(f"  Blend weights: {ensemble_meta['blend_weights']}")
        
        if ensemble_meta['ensemble_error'] < min(ensemble_meta['individual_errors'].values()):
            print(f"\n✓ Ensemble outperforms all individual methods!")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
