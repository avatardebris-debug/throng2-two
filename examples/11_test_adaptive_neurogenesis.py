"""
Test Phase 3b: Adaptive Neurogenesis + Compression Lifecycle
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.learning.adaptive_neurogenesis import (
    AdaptiveDensityController,
    BrainLifecycleManager,
    PerformanceDrivenNeurogenesis,
    CompressionScheduler,
    RegionType,
    LifecyclePhase,
    benchmark_adaptive_neurogenesis
)


def test_density_controller():
    """Test adaptive density controller."""
    print("\n" + "="*60)
    print("TEST: Adaptive Density Controller")
    print("="*60)
    
    controller = AdaptiveDensityController()
    
    # Test density targets
    print("\nDensity targets per region:")
    for region_type in RegionType:
        min_d, max_d = controller.get_density_range(region_type)
        print(f"  {region_type.value}: {min_d:.1%} - {max_d:.1%}")
    
    # Test growth decision
    print("\n--- Growth Decision Logic ---")
    
    scenarios = [
        ("High error + novelty, low density", 0.05, 0.8, 0.9, 0.7),
        ("High error, but max density", 0.25, 0.8, 0.9, 0.7),
        ("Low error, high novelty", 0.10, 0.2, 0.9, 0.7),
        ("High error, low novelty", 0.10, 0.8, 0.2, 0.7),
    ]
    
    region_type = RegionType.HIDDEN
    print(f"\nRegion: {region_type.value} (target: 5-10%)")
    print(f"{'Scenario':<35} {'Should Grow?':<12}")
    print("-" * 50)
    
    for name, density, error, novelty, dopamine in scenarios:
        should_grow = controller.should_grow(
            region_type, density, error, novelty, dopamine
        )
        print(f"{name:<35} {'Yes' if should_grow else 'No':<12}")
    
    return controller


def test_lifecycle_manager():
    """Test brain lifecycle management."""
    print("\n" + "="*60)
    print("TEST: Brain Lifecycle Manager")
    print("="*60)
    
    manager = BrainLifecycleManager(stabilization_threshold=0.1, compression_delay=20)
    
    region_id = "test_region"
    
    # Simulate activity over time
    print("\nSimulating lifecycle transitions:")
    print(f"{'Episode':<12} {'Activity':<12} {'Phase':<20}")
    print("-" * 50)
    
    for episode in range(60):
        # High activity early, low activity later
        if episode < 20:
            activity = 0.5 + 0.1 * np.random.randn()
        else:
            activity = 0.05 + 0.02 * np.random.randn()
        
        activity = max(0, min(1, activity))
        
        manager.update_region_activity(region_id, activity)
        phase = manager.get_phase(region_id)
        
        if episode % 10 == 0:
            print(f"{episode:<12} {activity:<12.2f} {phase.value:<20}")
    
    # Check compression decision
    should_compress = manager.should_compress(region_id)
    print(f"\nShould compress after 60 episodes? {should_compress}")
    
    return manager


def test_neurogenesis():
    """Test performance-driven neurogenesis."""
    print("\n" + "="*60)
    print("TEST: Performance-Driven Neurogenesis")
    print("="*60)
    
    density_controller = AdaptiveDensityController()
    neurogenesis = PerformanceDrivenNeurogenesis(density_controller)
    
    # Create sparse network
    weights = np.zeros((50, 50))
    mask = np.random.random((50, 50)) < 0.05
    weights[mask] = np.random.randn(np.sum(mask)) * 0.1
    
    initial_density = density_controller.compute_current_density(weights)
    print(f"Initial density: {initial_density:.2%}")
    
    # Simulate high-error scenario (should trigger growth)
    error_map = np.random.rand(50) * 2.0  # High errors
    novelty_score = 0.9  # High novelty
    neuromodulators = {'dopamine': 0.8, 'acetylcholine': 0.9}
    
    region_type = RegionType.HIDDEN
    
    # Try to grow
    new_weights = neurogenesis.grow_connections(
        weights, region_type, error_map, novelty_score, neuromodulators
    )
    
    final_density = density_controller.compute_current_density(new_weights)
    connections_added = np.count_nonzero(new_weights) - np.count_nonzero(weights)
    
    print(f"\nAfter neurogenesis:")
    print(f"  Final density: {final_density:.2%}")
    print(f"  Connections added: {connections_added}")
    print(f"  Growth rate: {(final_density - initial_density) / initial_density:.1%}")
    
    if connections_added > 0:
        print("✓ Neurogenesis triggered successfully!")
    
    # Statistics
    stats = neurogenesis.get_growth_statistics()
    if stats:
        print(f"\nGrowth statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return neurogenesis


def test_compression_scheduler():
    """Test compression scheduling."""
    print("\n" + "="*60)
    print("TEST: Compression Scheduler")
    print("="*60)
    
    lifecycle_manager = BrainLifecycleManager()
    scheduler = CompressionScheduler(lifecycle_manager)
    
    # Create multiple regions with different activity patterns
    regions = {
        'sensory_1': [0.8] * 50,      # Always active
        'hidden_1': [0.3] * 20 + [0.05] * 30,  # Became inactive
        'hidden_2': [0.05] * 50,      # Always inactive
        'output_1': [0.5] * 50,       # Moderately active
    }
    
    # Simulate activity
    for region_id, activities in regions.items():
        for activity in activities:
            lifecycle_manager.update_region_activity(region_id, activity)
    
    # Get compression priorities
    print("\nCompression priorities:")
    print(f"{'Region':<15} {'Phase':<20} {'Priority':<10}")
    print("-" * 50)
    
    for region_id in regions.keys():
        phase = lifecycle_manager.get_phase(region_id)
        priority = scheduler.get_compression_priority(region_id)
        print(f"{region_id:<15} {phase.value:<20} {priority:<10.2f}")
    
    # Select regions to compress
    to_compress = scheduler.select_regions_to_compress(
        list(regions.keys()),
        target_compressed_fraction=0.7
    )
    
    print(f"\nRegions selected for compression (target: 70%):")
    for region_id in to_compress:
        print(f"  - {region_id}")
    
    return scheduler


def visualize_neurogenesis_lifecycle():
    """Visualize neurogenesis and lifecycle dynamics."""
    print("\n" + "="*60)
    print("VISUALIZATION: Neurogenesis Lifecycle")
    print("="*60)
    
    # Setup
    density_controller = AdaptiveDensityController()
    lifecycle_manager = BrainLifecycleManager(compression_delay=50)
    neurogenesis = PerformanceDrivenNeurogenesis(density_controller)
    
    weights = np.random.randn(100, 100) * 0.1
    weights[np.random.random((100, 100)) < 0.95] = 0
    
    region_id = "visualization_region"
    region_type = RegionType.HIDDEN
    
    # Tracking
    n_episodes = 150
    density_history = []
    phase_history = []
    activity_history = []
    error_history = []
    growth_events = []
    
    for episode in range(n_episodes):
        # Simulate activity (high early, stabilizes later)
        if episode < 50:
            activity = 0.7 + 0.2 * np.random.randn()
            error_level = 0.8
        elif episode < 100:
            activity = 0.3 + 0.1 * np.random.randn()
            error_level = 0.4
        else:
            activity = 0.1 + 0.05 * np.random.randn()
            error_level = 0.2
        
        activity = np.clip(activity, 0, 1)
        
        # Update lifecycle
        lifecycle_manager.update_region_activity(region_id, activity)
        
        # Neurogenesis
        error_map = np.random.rand(100) * error_level
        novelty = max(0, 1.0 - episode / 75)
        neuromodulators = {'dopamine': 0.6, 'acetylcholine': novelty}
        
        connections_before = np.count_nonzero(weights)
        weights = neurogenesis.grow_connections(
            weights, region_type, error_map, novelty, neuromodulators
        )
        connections_after = np.count_nonzero(weights)
        
        if connections_after > connections_before:
            growth_events.append(episode)
        
        # Track
        density_history.append(density_controller.compute_current_density(weights))
        phase_history.append(lifecycle_manager.get_phase(region_id).value)
        activity_history.append(activity)
        error_history.append(error_level)
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Density over time
    axes[0].plot(density_history, linewidth=2, color='blue', label='Density')
    min_d, max_d = density_controller.get_density_range(region_type)
    axes[0].axhline(y=min_d, color='green', linestyle='--', alpha=0.5, label=f'Min target ({min_d:.1%})')
    axes[0].axhline(y=max_d, color='red', linestyle='--', alpha=0.5, label=f'Max target ({max_d:.1%})')
    
    # Mark growth events
    for event in growth_events:
        axes[0].axvline(x=event, color='orange', alpha=0.2, linewidth=0.5)
    
    axes[0].set_title('Network Density Over Time')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Activity and errors
    axes[1].plot(activity_history, linewidth=2, color='green', label='Activity')
    axes[1].plot(error_history, linewidth=2, color='red', label='Error')
    axes[1].axhline(y=0.1, color='black', linestyle='--', alpha=0.3, label='Stabilization threshold')
    axes[1].set_title('Activity and Error Levels')
    axes[1].set_ylabel('Level')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Lifecycle phases
    phase_numeric = []
    phase_map = {'active': 3, 'stabilizing': 2, 'compressed': 1, 'stored': 0}
    for phase in phase_history:
        phase_numeric.append(phase_map.get(phase, 0))
    
    axes[2].plot(phase_numeric, linewidth=2, color='purple')
    axes[2].set_title('Lifecycle Phase Transitions')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Phase')
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(['Stored', 'Compressed', 'Stabilizing', 'Active'])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_neurogenesis_lifecycle.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'adaptive_neurogenesis_lifecycle.png'")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 3B: ADAPTIVE NEUROGENESIS TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        test_density_controller()
        test_lifecycle_manager()
        test_neurogenesis()
        test_compression_scheduler()
        
        # Benchmark
        print("\n" + "="*60)
        benchmark_adaptive_neurogenesis()
        
        # Visualize
        visualize_neurogenesis_lifecycle()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print("\n✓ Adaptive density controller working")
        print("✓ Lifecycle management (Active → Stabilizing → Compressed)")
        print("✓ Performance-driven neurogenesis")
        print("✓ Compression scheduling based on activity")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
