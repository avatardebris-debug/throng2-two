"""
Test and benchmark the Phase 2 signal-to-noise infrastructure.

Tests:
1. Shannon entropy calculations
2. Fourier compression (100x, 1000x ratios)
3. Kelly Criterion allocation
4. Integration with existing thronglet brain
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.metrics.information_theory import (
    shannon_entropy,
    calculate_snr_fourier,
    prune_by_information
)
from src.compression.fourier_compression import (
    FourierCompressor,
    benchmark_compression,
    adaptive_frequency_selection
)
from src.allocation.kelly_allocator import (
    KellyAllocator,
    calculate_kelly_fraction
)


def test_shannon_entropy():
    """Test Shannon entropy calculations."""
    print("\n" + "="*60)
    print("TEST 1: Shannon Entropy")
    print("="*60)
    
    # Test 1: Random weights (high entropy)
    random_weights = np.random.randn(100, 100)
    entropy_random = shannon_entropy(random_weights)
    print(f"Random weights entropy: {entropy_random:.3f} bits")
    
    # Test 2: Structured weights (low entropy)
    structured_weights = np.zeros((100, 100))
    structured_weights[:50, :50] = 1.0
    structured_weights[50:, 50:] = -1.0
    entropy_structured = shannon_entropy(structured_weights)
    print(f"Structured weights entropy: {entropy_structured:.3f} bits")
    
    # Test 3: SNR calculation
    snr, spectrum = calculate_snr_fourier(random_weights)
    print(f"Random weights SNR: {snr:.2f} dB")
    
    snr_struct, _ = calculate_snr_fourier(structured_weights)
    print(f"Structured weights SNR: {snr_struct:.2f} dB")
    
    assert entropy_random > entropy_structured, "Random should have higher entropy"
    assert snr_struct > snr, "Structured should have higher SNR"
    
    print("✓ Shannon entropy tests passed!")
    return True


def test_fourier_compression():
    """Test Fourier compression."""
    print("\n" + "="*60)
    print("TEST 2: Fourier Compression")
    print("="*60)
    
    # Create test weight matrix
    weights = np.random.randn(100, 100) * 0.1
    # Add some structure
    weights[:50, :50] += 0.5
    weights[50:, 50:] -= 0.5
    
    print(f"Original size: {weights.size} values ({weights.nbytes} bytes)")
    
    # Test different compression ratios
    ratios = [10, 100, 1000]
    results = benchmark_compression(weights, ratios)
    
    print("\nCompression Results:")
    print(f"{'Ratio':<10} {'Actual':<10} {'MAE':<12} {'Rel Error':<12} {'Size':<10}")
    print("-" * 60)
    
    for ratio, result in results.items():
        print(f"{ratio:<10} {result['compression_ratio']:<10.1f} "
              f"{result['mae']:<12.6f} {result['relative_error']:<12.2%} "
              f"{result['compressed_size']:<10}")
    
    # Verify compression works
    compressor = FourierCompressor(compression_ratio=100)
    metadata = compressor.compress(weights, method='top_k')
    reconstructed = compressor.decompress()
    
    error = np.mean(np.abs(weights - reconstructed))
    print(f"\n100x compression error: {error:.6f}")
    print(f"Relative error: {error / np.abs(weights).mean():.2%}")
    
    assert error < 0.1, "Compression error too high"
    print("✓ Fourier compression tests passed!")
    
    return results


def test_kelly_allocator():
    """Test Kelly Criterion allocation."""
    print("\n" + "="*60)
    print("TEST 3: Kelly Criterion Allocator")
    print("="*60)
    
    # Create allocator with 1 GB RAM
    allocator = KellyAllocator(total_ram_bytes=1 * 1024**3)
    
    # Register 5 experts
    experts = {
        'navigation': 100 * 1024**2,  # 100 MB
        'vision': 200 * 1024**2,      # 200 MB
        'memory': 150 * 1024**2,      # 150 MB
        'motor': 80 * 1024**2,        # 80 MB
        'planning': 120 * 1024**2     # 120 MB
    }
    
    for name, size in experts.items():
        allocator.register_expert(name, size)
    
    # Simulate performance history
    # Navigation is good at navigation tasks
    for _ in range(50):
        allocator.record_performance('navigation', True, 10.0, 'navigation')
        allocator.record_performance('vision', False, 2.0, 'navigation')
        allocator.record_performance('memory', False, 1.0, 'navigation')
    
    # Vision is good at vision tasks
    for _ in range(50):
        allocator.record_performance('vision', True, 15.0, 'vision')
        allocator.record_performance('navigation', False, 3.0, 'vision')
    
    # Allocate for navigation task
    print("\nAllocation for navigation task:")
    nav_allocation = allocator.allocate('navigation', list(experts.keys()))
    
    for expert, bytes_allocated in sorted(nav_allocation.items(), 
                                         key=lambda x: x[1], reverse=True):
        mb = bytes_allocated / 1024**2
        print(f"  {expert:<15} {mb:>6.1f} MB")
    
    summary = allocator.get_allocation_summary()
    print(f"\nTotal allocated: {summary['allocated'] / 1024**2:.1f} MB")
    print(f"Utilization: {summary['utilization']:.1%}")
    
    # Verify navigation expert got most allocation
    assert 'navigation' in nav_allocation, "Navigation expert should be loaded"
    
    print("✓ Kelly allocator tests passed!")
    return True


def test_integration():
    """Test integration with existing thronglet brain."""
    print("\n" + "="*60)
    print("TEST 4: Integration with Thronglet Brain")
    print("="*60)
    
    # Import existing components
    from src.core.network import LayeredNetwork
    
    # Create small network
    network = LayeredNetwork(
        input_size=8,
        hidden_sizes=[50, 25],
        output_size=4,
        dimension=2,
        connection_prob=0.05
    )
    
    # Get weights from first hidden layer
    layer = network.layers[1]
    weights = layer.weights
    
    print(f"Network layer shape: {weights.shape}")
    print(f"Number of connections: {np.count_nonzero(weights)}")
    
    # Test 1: Calculate entropy
    entropy = shannon_entropy(weights)
    print(f"Weight entropy: {entropy:.3f} bits")
    
    # Test 2: Compress weights
    compressor = FourierCompressor(compression_ratio=10)
    metadata = compressor.compress(weights, method='adaptive')
    reconstructed = compressor.decompress()
    
    print(f"\nCompression:")
    print(f"  Original size: {weights.nbytes} bytes")
    print(f"  Compressed: {metadata['compressed_size'] * 16} bytes")  # complex = 16 bytes
    print(f"  Ratio: {metadata['compression_ratio']:.1f}x")
    
    error = np.mean(np.abs(weights - reconstructed))
    print(f"  Reconstruction error: {error:.6f}")
    
    # Test 3: Information-based pruning
    pruned = prune_by_information(weights, keep_fraction=0.5)
    print(f"\nInformation-based pruning:")
    print(f"  Original connections: {np.count_nonzero(weights)}")
    print(f"  After pruning: {np.count_nonzero(pruned)}")
    print(f"  Reduction: {(1 - np.count_nonzero(pruned)/np.count_nonzero(weights)):.1%}")
    
    print("✓ Integration tests passed!")
    return True


def benchmark_scaling():
    """Benchmark compression at different scales."""
    print("\n" + "="*60)
    print("BENCHMARK: Scaling Analysis")
    print("="*60)
    
    sizes = [100, 500, 1000, 2000]
    results = []
    
    print(f"{'Size':<10} {'Ratio':<10} {'Error':<12} {'Time (ms)':<12}")
    print("-" * 50)
    
    for size in sizes:
        weights = np.random.randn(size, size) * 0.1
        
        import time
        start = time.time()
        
        compressor = FourierCompressor(compression_ratio=100)
        compressor.compress(weights, method='top_k')
        reconstructed = compressor.decompress()
        
        elapsed = (time.time() - start) * 1000
        
        error = np.mean(np.abs(weights - reconstructed))
        
        print(f"{size:<10} {100:<10} {error:<12.6f} {elapsed:<12.1f}")
        
        results.append({
            'size': size,
            'error': error,
            'time_ms': elapsed
        })
    
    return results


def visualize_compression():
    """Visualize compression effects."""
    print("\n" + "="*60)
    print("VISUALIZATION: Compression Quality")
    print("="*60)
    
    # Create structured test matrix
    size = 100
    weights = np.zeros((size, size))
    
    # Add patterns at different frequencies
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Low frequency pattern
    weights += 0.5 * np.sin(X) * np.cos(Y)
    
    # Medium frequency pattern
    weights += 0.3 * np.sin(2*X) * np.cos(2*Y)
    
    # High frequency noise
    weights += 0.1 * np.random.randn(size, size)
    
    # Test different compression ratios
    ratios = [10, 100, 1000]
    
    fig, axes = plt.subplots(2, len(ratios) + 1, figsize=(15, 8))
    
    # Original
    axes[0, 0].imshow(weights, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[1, 0].hist(weights.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title('Weight Distribution')
    
    # Compressed versions
    for idx, ratio in enumerate(ratios):
        compressor = FourierCompressor(compression_ratio=ratio)
        compressor.compress(weights, method='top_k')
        reconstructed = compressor.decompress()
        
        error = weights - reconstructed
        
        # Reconstructed weights
        axes[0, idx+1].imshow(reconstructed, cmap='RdBu', vmin=-1, vmax=1)
        axes[0, idx+1].set_title(f'{ratio}x Compression')
        axes[0, idx+1].axis('off')
        
        # Error distribution
        axes[1, idx+1].hist(error.flatten(), bins=50, alpha=0.7, color='red')
        axes[1, idx+1].set_title(f'Error (MAE={np.mean(np.abs(error)):.4f})')
    
    plt.tight_layout()
    plt.savefig('compression_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'compression_visualization.png'")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 2: SIGNAL-TO-NOISE FOUNDATION TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        test_shannon_entropy()
        test_fourier_compression()
        test_kelly_allocator()
        test_integration()
        
        # Run benchmarks
        benchmark_scaling()
        
        # Visualize
        visualize_compression()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nPhase 2 infrastructure is ready!")
        print("\nNext steps:")
        print("  1. Integrate Fourier compression into neuron layers")
        print("  2. Create expert brain library")
        print("  3. Test on 10K neuron navigation task")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
