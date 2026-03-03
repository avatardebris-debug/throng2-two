"""
Nash Equilibrium Pruning - Competitive self-organization.

Key idea: Model the network as a game where neurons compete for resources.
Connections that don't contribute to Nash equilibrium are pruned.

This creates self-organizing, efficient networks that automatically
remove redundant connections while maintaining performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx


class NashPruningSystem:
    """
    Competitive pruning based on game-theoretic principles.
    
    Each neuron is a player, connections are strategies.
    Prune connections that don't contribute to Nash equilibrium.
    """
    
    def __init__(self,
                 pruning_threshold: float = 0.1,
                 resource_budget: float = 1.0,
                 competition_strength: float = 0.5):
        """
        Initialize Nash pruning system.
        
        Args:
            pruning_threshold: Minimum payoff to keep connection
            resource_budget: Total resources available per neuron
            competition_strength: How strongly neurons compete (0-1)
        """
        self.pruning_threshold = pruning_threshold
        self.resource_budget = resource_budget
        self.competition_strength = competition_strength
        
        # Track pruning statistics
        self.pruning_history = []
        self.total_pruned = 0
        
    def compute_neuron_payoffs(self,
                               weights: np.ndarray,
                               activities: np.ndarray,
                               rewards: np.ndarray) -> np.ndarray:
        """
        Compute payoff for each neuron based on activity and cost.
        
        Payoff = Benefit (activity × reward) - Cost (connections)
        
        Args:
            weights: Connection weights (n_neurons × n_neurons)
            activities: Neuron activities (n_neurons,)
            rewards: Reward signal for each neuron (n_neurons,)
            
        Returns:
            Payoffs for each neuron
        """
        n_neurons = len(activities)
        
        # Benefit: how much this neuron contributes to reward
        # Weighted by activity and downstream reward
        benefits = activities * rewards
        
        # Cost: maintaining connections uses resources
        # More connections = higher cost
        connection_counts = np.sum(np.abs(weights) > 0, axis=1)
        costs = connection_counts / n_neurons * self.resource_budget
        
        # Competition: neurons compete for limited resources
        # If many neurons are active, each gets less reward
        total_activity = np.sum(activities) + 1e-10
        competition_penalty = self.competition_strength * (activities / total_activity)
        
        # Final payoff
        payoffs = benefits - costs - competition_penalty
        
        return payoffs
    
    def find_nash_equilibrium(self,
                             weights: np.ndarray,
                             activities: np.ndarray,
                             rewards: np.ndarray) -> np.ndarray:
        """
        Find Nash equilibrium connection strengths.
        
        At equilibrium, no neuron can improve by changing its connections.
        
        Args:
            weights: Current weights
            activities: Neuron activities
            rewards: Reward signals
            
        Returns:
            Equilibrium connection strengths
        """
        n_neurons = weights.shape[0]
        
        # Compute current payoffs
        payoffs = self.compute_neuron_payoffs(weights, activities, rewards)
        
        # For each neuron, find optimal connection strength
        equilibrium_weights = np.zeros_like(weights)
        
        for i in range(n_neurons):
            # This neuron's connections
            connections = weights[i, :]
            
            # Optimal strategy: keep connections that increase payoff
            # Connection value = activity × reward × weight
            connection_values = activities * rewards * np.abs(connections)
            
            # Normalize by total activity
            total_value = np.sum(connection_values) + 1e-10
            normalized_values = connection_values / total_value
            
            # Keep connections above threshold
            keep_mask = normalized_values > self.pruning_threshold
            equilibrium_weights[i, keep_mask] = connections[keep_mask]
        
        return equilibrium_weights
    
    def prune_network(self,
                     weights: np.ndarray,
                     activities: np.ndarray,
                     rewards: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Prune network based on Nash equilibrium.
        
        Args:
            weights: Connection weights
            activities: Recent neuron activities
            rewards: Recent rewards
            
        Returns:
            Pruned weights and statistics
        """
        # Find equilibrium
        equilibrium = self.find_nash_equilibrium(weights, activities, rewards)
        
        # Prune connections not in equilibrium
        pruned_weights = weights.copy()
        prune_mask = (equilibrium == 0) & (weights != 0)
        
        n_pruned = np.sum(prune_mask)
        self.total_pruned += n_pruned
        
        pruned_weights[prune_mask] = 0
        
        # Statistics
        original_connections = np.count_nonzero(weights)
        remaining_connections = np.count_nonzero(pruned_weights)
        pruning_rate = n_pruned / max(1, original_connections)
        
        stats = {
            'original_connections': original_connections,
            'pruned_connections': n_pruned,
            'remaining_connections': remaining_connections,
            'pruning_rate': pruning_rate,
            'sparsity': 1.0 - remaining_connections / weights.size
        }
        
        self.pruning_history.append(stats)
        
        return pruned_weights, stats
    
    def detect_redundancy(self,
                         weights: np.ndarray,
                         threshold: float = 0.9) -> List[Tuple[int, int]]:
        """
        Detect redundant connections (multiple paths to same output).
        
        Args:
            weights: Connection weights
            threshold: Correlation threshold for redundancy
            
        Returns:
            List of redundant connection pairs
        """
        n_neurons = weights.shape[0]
        redundant_pairs = []
        
        # For each pair of neurons, check if they have similar output patterns
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                # Get output patterns
                pattern_i = weights[i, :]
                pattern_j = weights[j, :]
                
                # Skip if either is all zeros
                if np.all(pattern_i == 0) or np.all(pattern_j == 0):
                    continue
                
                # Compute correlation
                corr = np.corrcoef(pattern_i, pattern_j)[0, 1]
                
                if not np.isnan(corr) and abs(corr) > threshold:
                    redundant_pairs.append((i, j))
        
        return redundant_pairs
    
    def allow_regrowth(self,
                      weights: np.ndarray,
                      activities: np.ndarray,
                      growth_rate: float = 0.01) -> np.ndarray:
        """
        Allow new connections to form based on correlated activity.
        
        Hebbian principle: neurons that fire together, wire together.
        
        Args:
            weights: Current weights
            activities: Recent activities
            growth_rate: Rate of new connection formation
            
        Returns:
            Weights with potential new connections
        """
        n_neurons = len(activities)
        new_weights = weights.copy()
        
        # For each pair of neurons
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i == j:
                    continue
                
                # If no connection exists
                if weights[i, j] == 0:
                    # Check if activities are correlated
                    correlation = activities[i] * activities[j]
                    
                    # Probabilistic growth based on correlation
                    if np.random.random() < correlation * growth_rate:
                        # Create small initial connection
                        new_weights[i, j] = np.random.randn() * 0.01
        
        return new_weights
    
    def get_pruning_stats(self) -> Dict:
        """Get overall pruning statistics."""
        if not self.pruning_history:
            return {}
        
        total_original = sum(h['original_connections'] for h in self.pruning_history)
        total_pruned = sum(h['pruned_connections'] for h in self.pruning_history)
        
        return {
            'total_pruning_events': len(self.pruning_history),
            'total_connections_pruned': self.total_pruned,
            'average_pruning_rate': total_pruned / max(1, total_original),
            'final_sparsity': self.pruning_history[-1]['sparsity'] if self.pruning_history else 0
        }


def benchmark_nash_pruning(network_size: int = 100,
                          n_episodes: int = 10) -> Dict:
    """
    Benchmark Nash pruning on a simple network.
    
    Args:
        network_size: Number of neurons
        n_episodes: Number of pruning episodes
        
    Returns:
        Benchmark results
    """
    # Create random network
    weights = np.random.randn(network_size, network_size) * 0.1
    weights[np.random.random((network_size, network_size)) < 0.9] = 0  # 90% sparse
    
    pruner = NashPruningSystem(pruning_threshold=0.1)
    
    results = {
        'initial_connections': np.count_nonzero(weights),
        'episode_stats': []
    }
    
    for episode in range(n_episodes):
        # Simulate activity and rewards
        activities = np.random.rand(network_size)
        rewards = np.random.rand(network_size)
        
        # Prune
        weights, stats = pruner.prune_network(weights, activities, rewards)
        results['episode_stats'].append(stats)
        
        # Allow regrowth occasionally
        if episode % 3 == 0:
            weights = pruner.allow_regrowth(weights, activities)
    
    results['final_connections'] = np.count_nonzero(weights)
    results['total_pruned'] = results['initial_connections'] - results['final_connections']
    results['pruning_rate'] = results['total_pruned'] / results['initial_connections']
    
    return results
