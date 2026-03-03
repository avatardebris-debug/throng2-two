"""
Phase 3d Part 2: Preventive Redundancy Manager

Add backup connections before errors occur.
Complement Nash pruning with strategic redundancy.
"""

import numpy as np
from scipy.sparse import lil_matrix
from typing import Dict, List, Tuple, Optional
import sys
import os

# Import from Phase 3.5
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.learning.scaling_infrastructure import SparseWeightMatrix
except:
    SparseWeightMatrix = None  # Fallback for testing


class PreventiveRedundancyManager:
    """
    Add redundant connections based on risk predictions.
    
    Strategies:
    1. Parallel pathways - Create alternative routes
    2. Density boost - Strengthen local connectivity
    3. Cross-connections - Link isolated regions
    """
    
    def __init__(self, 
                 high_risk_threshold: float = 0.7,
                 medium_risk_threshold: float = 0.5):
        """
        Initialize redundancy manager.
        
        Args:
            high_risk_threshold: Risk above this → multiple paths
            medium_risk_threshold: Risk above this → one backup
        """
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        
        # Track redundancy additions
        self.redundancy_added = []
        
    def add_redundancy(self,
                      connection_id: Tuple[int, int],
                      risk_score: float,
                      weights: np.ndarray) -> np.ndarray:
        """
        Add redundancy for a single connection.
        
        Args:
            connection_id: (source, target)
            risk_score: Predicted risk (0-1)
            weights: Weight matrix
            
        Returns:
            Modified weights with redundancy
        """
        if risk_score < self.medium_risk_threshold:
            return weights  # No action needed
        
        source, target = connection_id
        
        if risk_score > self.high_risk_threshold:
            # High risk → multiple parallel paths
            weights = self._add_parallel_pathways(
                source, target, weights, n_paths=2
            )
        else:
            # Medium risk → one backup path
            weights = self._add_parallel_pathways(
                source, target, weights, n_paths=1
            )
        
        return weights
    
    def _add_parallel_pathways(self,
                               source: int,
                               target: int,
                               weights: np.ndarray,
                               n_paths: int = 1) -> np.ndarray:
        """
        Create parallel pathways from source to target.
        
        Strategy: source → intermediate → target
        
        Args:
            source: Source neuron
            target: Target neuron
            weights: Weight matrix
            n_paths: Number of parallel paths to add
            
        Returns:
            Modified weights
        """
        n_neurons = weights.shape[0]
        
        for _ in range(n_paths):
            # Find intermediate neuron (random for now)
            # Future: could use topology, activity patterns
            intermediate = np.random.choice(
                [i for i in range(n_neurons) if i != source and i != target]
            )
            
            # Add connections: source → intermediate → target
            # Weights proportional to original connection
            original_weight = weights[source, target]
            
            # Split weight through intermediate
            w1 = original_weight * 0.7  # source → intermediate
            w2 = original_weight * 0.7  # intermediate → target
            
            weights[source, intermediate] = w1
            weights[intermediate, target] = w2
            
            # Track
            self.redundancy_added.append({
                'original': (source, target),
                'intermediate': intermediate,
                'weights': (w1, w2)
            })
        
        return weights
    
    def add_redundancy_batch(self,
                            risks: Dict[Tuple[int, int], float],
                            weights: np.ndarray) -> np.ndarray:
        """
        Add redundancy for all high-risk connections.
        
        Args:
            risks: {connection_id: risk_score}
            weights: Weight matrix
            
        Returns:
            Modified weights with redundancy added
        """
        # Sort by risk (highest first)
        sorted_risks = sorted(risks.items(), key=lambda x: x[1], reverse=True)
        
        redundancy_count = 0
        
        for connection_id, risk_score in sorted_risks:
            if risk_score >= self.medium_risk_threshold:
                weights = self.add_redundancy(connection_id, risk_score, weights)
                redundancy_count += 1
        
        return weights
    
    def boost_local_density(self,
                           neuron_id: int,
                           weights: np.ndarray,
                           boost_factor: float = 1.5) -> np.ndarray:
        """
        Increase connection density around a neuron.
        
        Useful for neurons in error-prone regions.
        
        Args:
            neuron_id: Neuron to boost around
            weights: Weight matrix
            boost_factor: Multiply connections by this
            
        Returns:
            Modified weights
        """
        # Boost outgoing connections
        weights[neuron_id, :] *= boost_factor
        
        # Boost incoming connections
        weights[:, neuron_id] *= boost_factor
        
        return weights
    
    def create_cross_connections(self,
                                region_a: List[int],
                                region_b: List[int],
                                weights: np.ndarray,
                                connection_density: float = 0.1) -> np.ndarray:
        """
        Create connections between two regions.
        
        Useful for reducing isolation.
        
        Args:
            region_a: List of neuron indices in region A
            region_b: List of neuron indices in region B  
            weights: Weight matrix
            connection_density: Fraction of possible connections to add
            
        Returns:
            Modified weights
        """
        n_possible = len(region_a) * len(region_b)
        n_to_add = int(n_possible * connection_density)
        
        for _ in range(n_to_add):
            # Random connection
            source = np.random.choice(region_a)
            target = np.random.choice(region_b)
            
            # Add small connection if doesn't exist
            if weights[source, target] == 0:
                weights[source, target] = np.random.randn() * 0.01
        
        return weights
    
    def get_redundancy_statistics(self) -> Dict:
        """Get statistics on redundancy additions."""
        return {
            'total_redundancies_added': len(self.redundancy_added),
            'unique_connections_protected': len(set(
                r['original'] for r in self.redundancy_added
            ))
        }


def integrate_with_nash_pruning(weights: np.ndarray,
                                error_learner,
                                redundancy_manager,
                                nash_pruner=None) -> Tuple[np.ndarray, Dict]:
    """
    Unified optimization: Redundancy + Pruning
    
    This is the key integration function.
    
    Args:
        weights: Weight matrix
        error_learner: ConnectionErrorLearner
        redundancy_manager: PreventiveRedundancyManager
        nash_pruner: Optional Nash pruning system
        
    Returns:
        Optimized weights and statistics
    """
    stats = {
        'initial_connections': np.count_nonzero(weights),
        'redundancy_added': 0,
        'connections_pruned': 0,
        'final_connections': 0
    }
    
    # Step 1: Predict risks
    risks = error_learner.predict_all_risks(weights)
    
    # Step 2: Add preventive redundancy
    weights = redundancy_manager.add_redundancy_batch(risks, weights)
    stats['redundancy_added'] = len(redundancy_manager.redundancy_added)
    
    # Step 3: Prune (if nash_pruner available)
    if nash_pruner:
        # Simplified pruning: remove very weak connections
        prune_threshold = 0.01
        prune_mask = np.abs(weights) < prune_threshold
        stats['connections_pruned'] = np.sum(prune_mask)
        weights[prune_mask] = 0
    
    stats['final_connections'] = np.count_nonzero(weights)
    
    return weights, stats


def benchmark_preventive_redundancy():
    """Benchmark preventive redundancy."""
    print("\nBenchmarking Preventive Redundancy...")
    
    # Import error learner
    from src.learning.error_patterns import ConnectionErrorLearner
    
    # Create test network
    n = 100
    weights = np.random.randn(n, n) * 0.1
    weights[np.random.random((n, n)) < 0.9] = 0  # 90% sparse
    
    print(f"\nInitial network:")
    print(f"  Neurons: {n}")
    print(f"  Connections: {np.count_nonzero(weights)}")
    
    # Create managers
    error_learner = ConnectionErrorLearner()
    redundancy_manager = PreventiveRedundancyManager()
    
    # Simulate some errors to build risk model
    for i in range(n):
        for j in range(n):
            if weights[i, j] != 0:
                # Weak connections more likely to error
                if abs(weights[i, j]) < 0.05:
                    error_learner.record_error((i, j), 0.8)
    
    # Predict risks
    risks = error_learner.predict_all_risks(weights)
    high_risk = [r for r in risks.values() if r > 0.7]
    
    print(f"\nRisk analysis:")
    print(f"  High risk connections: {len(high_risk)}")
    
    # Add redundancy
    optimized_weights, stats = integrate_with_nash_pruning(
        weights, error_learner, redundancy_manager
    )
    
    print(f"\nOptimization results:")
    print(f"  Initial connections: {stats['initial_connections']}")
    print(f"  Redundancy added: {stats['redundancy_added']}")
    print(f"  Connections pruned: {stats['connections_pruned']}")
    print(f"  Final connections: {stats['final_connections']}")
    
    return optimized_weights, stats
