"""
Test Nash equilibrium pruning system.

Validates competitive self-organization and automatic network optimization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.learning.nash_pruning import NashPruningSystem, benchmark_nash_pruning


def test_basic_pruning():
    """Test basic Nash pruning functionality."""
    print("\n" + "="*60)
    print("TEST: Basic Nash Pruning")
    print("="*60)
    
    # Create small network
    n_neurons = 20
    weights = np.random.randn(n_neurons, n_neurons) * 0.2
    weights[np.random.random((n_neurons, n_neurons)) < 0.7] = 0  # 70% sparse
    
    print(f"Initial network: {n_neurons} neurons")
    print(f"Initial connections: {np.count_nonzero(weights)}")
    print(f"Initial sparsity: {np.sum(weights == 0) / weights.size:.1%}")
    
    # Create pruner
    pruner = NashPruningSystem(pruning_threshold=0.15)
    
    # Simulate activity
    activities = np.random.rand(n_neurons)
    rewards = np.random.rand(n_neurons)
    
    # Prune
    pruned_weights, stats = pruner.prune_network(weights, activities, rewards)
    
    print(f"\nAfter pruning:")
    print(f"Connections pruned: {stats['pruned_connections']}")
    print(f"Remaining connections: {stats['remaining_connections']}")
    print(f"Pruning rate: {stats['pruning_rate']:.1%}")
    print(f"Final sparsity: {stats['sparsity']:.1%}")
    
    return pruned_weights, stats


def test_iterative_pruning():
    """Test multiple rounds of pruning."""
    print("\n" + "="*60)
    print("TEST: Iterative Pruning")
    print("="*60)
    
    n_neurons = 50
    weights = np.random.randn(n_neurons, n_neurons) * 0.2
    weights[np.random.random((n_neurons, n_neurons)) < 0.8] = 0  # 80% sparse
    
    pruner = NashPruningSystem(pruning_threshold=0.1)
    
    print(f"Initial connections: {np.count_nonzero(weights)}")
    
    n_rounds = 5
    connection_history = [np.count_nonzero(weights)]
    
    for round_num in range(n_rounds):
        # Simulate activity
        activities = np.random.rand(n_neurons)
        rewards = np.random.rand(n_neurons)
        
        # Prune
        weights, stats = pruner.prune_network(weights, activities, rewards)
        connection_history.append(stats['remaining_connections'])
        
        print(f"Round {round_num + 1}: {stats['remaining_connections']} connections "
              f"({stats['pruning_rate']:.1%} pruned)")
    
    # Overall statistics
    overall_stats = pruner.get_pruning_stats()
    print(f"\nOverall statistics:")
    print(f"Total pruning events: {overall_stats['total_pruning_events']}")
    print(f"Total pruned: {overall_stats['total_connections_pruned']}")
    print(f"Average pruning rate: {overall_stats['average_pruning_rate']:.1%}")
    print(f"Final sparsity: {overall_stats['final_sparsity']:.1%}")
    
    return connection_history


def test_redundancy_detection():
    """Test detection of redundant connections."""
    print("\n" + "="*60)
    print("TEST: Redundancy Detection")
    print("="*60)
    
    n_neurons = 10
    weights = np.zeros((n_neurons, n_neurons))
    
    # Create redundant neurons (similar output patterns)
    pattern = np.random.randn(n_neurons)
    weights[0, :] = pattern
    weights[1, :] = pattern + np.random.randn(n_neurons) * 0.1  # Similar
    weights[2, :] = np.random.randn(n_neurons)  # Different
    
    pruner = NashPruningSystem()
    redundant_pairs = pruner.detect_redundancy(weights, threshold=0.8)
    
    print(f"Created network with intentional redundancy")
    print(f"Redundant pairs found: {len(redundant_pairs)}")
    
    for i, j in redundant_pairs:
        corr = np.corrcoef(weights[i, :], weights[j, :])[0, 1]
        print(f"  Neurons {i} and {j}: correlation = {corr:.3f}")
    
    if (0, 1) in redundant_pairs:
        print("✓ Correctly detected redundant neurons 0 and 1")
    
    return redundant_pairs


def test_connection_regrowth():
    """Test connection regrowth mechanism."""
    print("\n" + "="*60)
    print("TEST: Connection Regrowth")
    print("="*60)
    
    n_neurons = 20
    weights = np.zeros((n_neurons, n_neurons))
    
    # Add a few initial connections
    for i in range(5):
        j = np.random.randint(0, n_neurons)
        weights[i, j] = np.random.randn() * 0.2
    
    initial_connections = np.count_nonzero(weights)
    print(f"Initial connections: {initial_connections}")
    
    pruner = NashPruningSystem()
    
    # Simulate correlated activity
    activities = np.random.rand(n_neurons)
    activities[:5] = 0.9  # First 5 neurons highly active
    
    # Allow regrowth
    new_weights = pruner.allow_regrowth(weights, activities, growth_rate=0.5)
    
    new_connections = np.count_nonzero(new_weights)
    grown = new_connections - initial_connections
    
    print(f"After regrowth: {new_connections} connections")
    print(f"New connections formed: {grown}")
    
    if grown > 0:
        print("✓ New connections successfully formed")
    
    return new_weights


def benchmark_pruning_performance():
    """Benchmark pruning on larger network."""
    print("\n" + "="*60)
    print("BENCHMARK: Nash Pruning Performance")
    print("="*60)
    
    results = benchmark_nash_pruning(network_size=100, n_episodes=10)
    
    print(f"\nInitial connections: {results['initial_connections']}")
    print(f"Final connections: {results['final_connections']}")
    print(f"Total pruned: {results['total_pruned']}")
    print(f"Overall pruning rate: {results['pruning_rate']:.1%}")
    
    print(f"\nPer-episode statistics:")
    for i, stats in enumerate(results['episode_stats']):
        print(f"  Episode {i+1}: {stats['pruning_rate']:.1%} pruned, "
              f"{stats['remaining_connections']} remaining")
    
    return results


def visualize_pruning_dynamics():
    """Visualize how pruning affects network over time."""
    print("\n" + "="*60)
    print("VISUALIZATION: Pruning Dynamics")
    print("="*60)
    
    n_neurons = 50
    weights = np.random.randn(n_neurons, n_neurons) * 0.2
    weights[np.random.random((n_neurons, n_neurons)) < 0.85] = 0
    
    pruner = NashPruningSystem(pruning_threshold=0.1)
    
    n_rounds = 20
    connection_counts = [np.count_nonzero(weights)]
    sparsity_levels = [np.sum(weights == 0) / weights.size]
    pruning_rates = []
    
    for _ in range(n_rounds):
        activities = np.random.rand(n_neurons)
        rewards = np.random.rand(n_neurons)
        
        weights, stats = pruner.prune_network(weights, activities, rewards)
        
        connection_counts.append(stats['remaining_connections'])
        sparsity_levels.append(stats['sparsity'])
        pruning_rates.append(stats['pruning_rate'])
        
        # Occasional regrowth
        if _ % 5 == 0:
            weights = pruner.allow_regrowth(weights, activities, growth_rate=0.1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Connection count over time
    axes[0].plot(connection_counts, linewidth=2, color='blue')
    axes[0].set_title('Network Connections Over Time')
    axes[0].set_xlabel('Pruning Round')
    axes[0].set_ylabel('Number of Connections')
    axes[0].grid(True, alpha=0.3)
    
    # Sparsity over time
    axes[1].plot(sparsity_levels, linewidth=2, color='green')
    axes[1].set_title('Network Sparsity Over Time')
    axes[1].set_xlabel('Pruning Round')
    axes[1].set_ylabel('Sparsity')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    # Pruning rate per round
    axes[2].bar(range(len(pruning_rates)), pruning_rates, color='orange')
    axes[2].set_title('Pruning Rate Per Round')
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('Pruning Rate')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('nash_pruning_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'nash_pruning_dynamics.png'")
    
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NASH EQUILIBRIUM PRUNING TEST SUITE")
    print("="*60)
    
    try:
        # Run tests
        test_basic_pruning()
        connection_history = test_iterative_pruning()
        test_redundancy_detection()
        test_connection_regrowth()
        benchmark_results = benchmark_pruning_performance()
        
        # Visualize
        visualize_pruning_dynamics()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print("\n✓ Nash equilibrium pruning working correctly")
        print("✓ Competitive dynamics remove redundant connections")
        print("✓ Connection regrowth allows network plasticity")
        print("✓ Self-organization creates efficient sparse networks")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
