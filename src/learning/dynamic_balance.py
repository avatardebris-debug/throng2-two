"""
Phase 3d Part 3: Dynamic Balance Controller

Meta-cognitive controller that learns optimal balance
between pruning (efficiency) and redundancy (robustness).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class DynamicBalanceController:
    """
    Brain learns to balance efficiency vs robustness.
    
    Monitors:
    - Error rate (too high → add redundancy, prune less)
    - Network size (too large → prune more, add less redundancy)
    - Performance trend (improving → continue, declining → adjust)
    
    Self-tunes:
    - Pruning threshold (higher = more aggressive pruning)
    - Redundancy threshold (lower = more redundancy added)
    """
    
    def __init__(self,
                 initial_pruning_threshold: float = 0.05,
                 initial_redundancy_threshold: float = 0.7,
                 error_window: int = 50):
        """
        Initialize balance controller.
        
        Args:
            initial_pruning_threshold: Starting pruning threshold
            initial_redundancy_threshold: Starting redundancy threshold
            error_window: Window for computing error trends
        """
        # Tunable parameters        
        self.pruning_threshold = initial_pruning_threshold
        self.redundancy_threshold = initial_redundancy_threshold
        
        # History tracking
        self.error_history = deque(maxlen=error_window)
        self.size_history = deque(maxlen=error_window)
        self.pruning_history = deque(maxlen=error_window)
        self.redundancy_history = deque(maxlen=error_window)
        
        # Learning parameters
        self.adjustment_rate = 0.05  # How fast to adjust
        self.error_target = 0.1  # Target error rate
        
        # Statistics
        self.total_adjustments = 0
        self.improvement_count = 0
        
    def record_episode(self,
                      error_rate: float,
                      network_size: int,
                      connections_pruned: int = 0,
                      redundancies_added: int = 0):
        """Record performance for this episode."""
        self.error_history.append(error_rate)
        self.size_history.append(network_size)
        self.pruning_history.append(connections_pruned)
        self.redundancy_history.append(redundancies_added)
    
    def compute_error_trend(self, window: int = 10) -> float:
        """
        Compute error rate trend.
        
        Returns:
            Positive = errors increasing, Negative = errors decreasing
        """
        if len(self.error_history) < window * 2:
            return 0.0
        
        recent = list(self.error_history)[-window:]
        older = list(self.error_history)[-window*2:-window]
        
        return np.mean(recent) - np.mean(older)
    
    def compute_size_trend(self, window: int = 10) -> float:
        """
        Compute network size trend.
        
        Returns:
            Positive = growing, Negative = shrinking
        """
        if len(self.size_history) < window * 2:
            return 0.0
        
        recent = list(self.size_history)[-window:]
        older = list(self.size_history)[-window*2:-window]
        
        return np.mean(recent) - np.mean(older)
    
    def adjust_balance(self) -> Dict[str, float]:
        """
        Self-tune based on recent performance.
        
        Rules:
        1. If errors increasing → prune less, add more redundancy
        2. If errors decreasing → prune more, add less redundancy
        3. If network growing too much → increase pruning
        4. If network shrinking too much → decrease pruning
        
        Returns:
            Adjustment amounts
        """
        if len(self.error_history) < 20:
            return {'pruning': 0.0, 'redundancy': 0.0}  # Need more data
        
        error_trend = self.compute_error_trend()
        size_trend = self.compute_size_trend()
        current_error = np.mean(list(self.error_history)[-10:])
        
        adjustments = {'pruning': 0.0, 'redundancy': 0.0}
        
        # Rule 1: Error-based adjustment
        if current_error > self.error_target:
            # Errors too high
            if error_trend > 0:
                # And getting worse
                # → Prune LESS, add MORE redundancy
                self.pruning_threshold *= (1 - self.adjustment_rate)
                self.redundancy_threshold *= (1 - self.adjustment_rate)
                adjustments['pruning'] = -self.adjustment_rate
                adjustments['redundancy'] = -self.adjustment_rate
            else:
                # But improving
                # → Keep current strategy
                pass
        else:
            # Errors acceptable
            if error_trend < 0:
                # And still improving
                # → Can afford more efficiency: prune MORE, add LESS redundancy
                self.pruning_threshold *= (1 + self.adjustment_rate)
                self.redundancy_threshold *= (1 + self.adjustment_rate)
                adjustments['pruning'] = self.adjustment_rate
                adjustments['redundancy'] = self.adjustment_rate
        
        # Rule 2: Size-based adjustment (secondary)
        if size_trend > 100:  # Growing too fast
            # Increase pruning
            self.pruning_threshold *= 1.1
            adjustments['pruning'] += 0.1
        elif size_trend < -100:  # Shrinking too fast
            # Decrease pruning
            self.pruning_threshold *= 0.9
            adjustments['pruning'] -= 0.1
        
        # Clamp thresholds to reasonable ranges
        self.pruning_threshold = np.clip(self.pruning_threshold, 0.001, 0.5)
        self.redundancy_threshold = np.clip(self.redundancy_threshold, 0.3, 0.9)
        
        self.total_adjustments += 1
        
        if error_trend < 0:
            self.improvement_count += 1
        
        return adjustments
    
    def get_current_strategy(self) -> Dict:
        """Get current balance strategy."""
        return {
            'pruning_threshold': self.pruning_threshold,
            'redundancy_threshold': self.redundancy_threshold,
            'error_target': self.error_target
        }
    
    def get_statistics(self) -> Dict:
        """Get controller statistics."""
        if len(self.error_history) == 0:
            return {}
        
        return {
            'episodes_tracked': len(self.error_history),
            'total_adjustments': self.total_adjustments,
            'improvement_rate': self.improvement_count / max(1, self.total_adjustments),
            'current_error_rate': np.mean(list(self.error_history)[-10:]) if len(self.error_history) >= 10 else 0,
            'error_trend': self.compute_error_trend(),
            'size_trend': self.compute_size_trend(),
            'current_pruning_threshold': self.pruning_threshold,
            'current_redundancy_threshold': self.redundancy_threshold
        }


class PredictiveLearningSystem:
    """
    Complete Phase 3d system integrating all components.
    
    Combines:
    - Error pattern learning (Part 1)
    - Preventive redundancy (Part 2)  
    - Dynamic balance (Part 3)
    - Nash pruning (Phase 3)
    """
    
    def __init__(self,
                 error_learner,
                 redundancy_manager,
                 balance_controller,
                 nash_pruner=None):
        """
        Initialize complete system.
        
        Args:
            error_learner: ConnectionErrorLearner
            redundancy_manager: PreventiveRedundancyManager
            balance_controller: DynamicBalanceController
            nash_pruner: Optional Nash pruning system
        """
        self.error_learner = error_learner
        self.redundancy_manager = redundancy_manager
        self.balance_controller = balance_controller
        self.nash_pruner = nash_pruner
        
        # Performance tracking
        self.episode_count = 0
        self.total_errors_prevented = 0
    
    def optimize_network(self, weights: np.ndarray, current_error: float) -> Tuple[np.ndarray, Dict]:
        """
        Full optimization cycle.
        
        Steps:
        1. Predict error risks
        2. Add preventive redundancy
        3. Prune weak connections
        4. Adjust balance based on results
        
        Args:
            weights: Current weight matrix
            current_error: Current error rate
            
        Returns:
            Optimized weights and statistics
        """
        initial_size = np.count_nonzero(weights)
        
        # Get current strategy
        strategy = self.balance_controller.get_current_strategy()
        
        # Step 1: Predict risks
        risks = self.error_learner.predict_all_risks(weights)
        
        # Step 2: Add redundancy (using current threshold)
        self.redundancy_manager.redundancy_threshold = strategy['redundancy_threshold']
        weights = self.redundancy_manager.add_redundancy_batch(risks, weights)
        redundancies_added = len(self.redundancy_manager.redundancy_added) - self.redundancy_manager.get_redundancy_statistics()['total_redundancies_added']
        
        # Step 3: Prune (using current threshold)
        connections_pruned = 0
        if self.nash_pruner:
            # Use nash pruner with current threshold
            pass  # Integrated later
        else:
            # Simple pruning
            prune_mask = np.abs(weights) < strategy['pruning_threshold']
            connections_pruned = np.sum(prune_mask)
            weights[prune_mask] = 0
        
        final_size = np.count_nonzero(weights)
        
        # Step 4: Record and adjust
        self.balance_controller.record_episode(
            current_error, final_size, connections_pruned, redundancies_added
        )
        adjustments = self.balance_controller.adjust_balance()
        
        # Statistics
        stats = {
            'initial_size': initial_size,
            'final_size': final_size,
            'redundancies_added': redundancies_added,
            'connections_pruned': connections_pruned,
            'net_change': final_size - initial_size,
            'adjustments': adjustments,
            'strategy': strategy
        }
        
        self.episode_count += 1
        
        return weights, stats
    
    def get_system_statistics(self) -> Dict:
        """Get complete system statistics."""
        return {
            'episodes': self.episode_count,
            'error_learning': self.error_learner.get_statistics(),
            'redundancy': self.redundancy_manager.get_redundancy_statistics(),
            'balance': self.balance_controller.get_statistics()
        }


def benchmark_dynamic_balance():
    """Benchmark dynamic balance controller."""
    print("\nBenchmarking Dynamic Balance Controller...")
    
    controller = DynamicBalanceController()
    
    # Simulate learning
    print("\nSimulating 100 episodes...")
    
    for episode in range(100):
        # Simulate improving performance
        base_error = 0.5 * np.exp(-episode / 50)  # Exponential decay
        noise = np.random.randn() * 0.05
        error = max(0, base_error + noise)
        
        # Simulate network size
        network_size = 1000 + episode * 2  # Slowly growing
        
        # Record
        controller.record_episode(error, network_size)
        
        # Adjust every 10 episodes
        if episode % 10 == 9:
            adjustments = controller.adjust_balance()
            stats = controller.get_statistics()
            
            print(f"\nEpisode {episode + 1}:")
            print(f"  Error: {stats['current_error_rate']:.3f}")
            print(f"  Pruning threshold: {stats['current_pruning_threshold']:.3f}")
            print(f"  Redundancy threshold: {stats['current_redundancy_threshold']:.3f}")
    
    # Final statistics
    final_stats = controller.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Improvement rate: {final_stats['improvement_rate']:.1%}")
    print(f"  Total adjustments: {final_stats['total_adjustments']}")
    
    return controller
