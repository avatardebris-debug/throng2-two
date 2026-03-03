"""
Kelly Criterion-based resource allocation for expert brains.

Optimally allocates limited RAM to experts based on their
signal-to-noise ratio and expected performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class ExpertPerformanceTracker:
    """Track expert performance history for probability estimation."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize tracker.
        
        Args:
            window_size: Number of recent episodes to track
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.task_history = {}  # task_type -> deque of results
        
    def record(self, success: bool, reward: float, task_type: str = 'default'):
        """
        Record performance on a task.
        
        Args:
            success: Whether task succeeded
            reward: Reward received
            task_type: Type of task
        """
        self.history.append({
            'success': success,
            'reward': reward,
            'task_type': task_type
        })
        
        # Track by task type
        if task_type not in self.task_history:
            self.task_history[task_type] = deque(maxlen=self.window_size)
        self.task_history[task_type].append({
            'success': success,
            'reward': reward
        })
    
    def get_win_rate(self, task_type: Optional[str] = None) -> float:
        """
        Get success rate.
        
        Args:
            task_type: Specific task type (None for overall)
            
        Returns:
            Win rate (0-1)
        """
        if task_type and task_type in self.task_history:
            history = self.task_history[task_type]
        else:
            history = self.history
        
        if len(history) == 0:
            return 0.5  # Neutral prior
        
        successes = sum(1 for h in history if h['success'])
        return successes / len(history)
    
    def get_average_reward(self, task_type: Optional[str] = None) -> float:
        """Get average reward."""
        if task_type and task_type in self.task_history:
            history = self.task_history[task_type]
        else:
            history = self.history
        
        if len(history) == 0:
            return 0.0
        
        return np.mean([h['reward'] for h in history])
    
    def get_variance(self, task_type: Optional[str] = None) -> float:
        """Get reward variance (risk measure)."""
        if task_type and task_type in self.task_history:
            history = self.task_history[task_type]
        else:
            history = self.history
        
        if len(history) == 0:
            return 1.0
        
        rewards = [h['reward'] for h in history]
        return np.var(rewards)


def calculate_kelly_fraction(win_prob: float, 
                             payoff_ratio: float,
                             safety_factor: float = 0.5) -> float:
    """
    Calculate Kelly Criterion fraction.
    
    Formula: f* = (p*b - q) / b
    where:
        p = probability of winning
        q = probability of losing (1-p)
        b = payoff ratio (gain/loss)
    
    Args:
        win_prob: Probability of success (0-1)
        payoff_ratio: Expected gain / expected loss
        safety_factor: Fraction of Kelly to use (0.25-0.5 recommended)
        
    Returns:
        Optimal allocation fraction (0-1)
    """
    p = np.clip(win_prob, 0.01, 0.99)  # Avoid extremes
    q = 1 - p
    b = max(0.01, payoff_ratio)  # Avoid division by zero
    
    # Kelly formula
    kelly = (p * b - q) / b
    
    # Apply safety factor (fractional Kelly)
    safe_kelly = kelly * safety_factor
    
    # Clip to valid range
    return np.clip(safe_kelly, 0.0, 1.0)


def estimate_win_probability(performance_tracker: ExpertPerformanceTracker,
                            task_type: str,
                            prior: float = 0.5) -> float:
    """
    Estimate probability expert will succeed on task.
    
    Uses Bayesian updating with performance history.
    
    Args:
        performance_tracker: Performance history
        task_type: Type of task
        prior: Prior probability (default 0.5)
        
    Returns:
        Estimated win probability
    """
    # Get historical win rate
    historical_rate = performance_tracker.get_win_rate(task_type)
    
    # Number of samples
    n_samples = len(performance_tracker.task_history.get(task_type, []))
    
    if n_samples == 0:
        return prior
    
    # Bayesian update: posterior = (prior * likelihood) / evidence
    # Simplified: weighted average of prior and historical rate
    weight = min(1.0, n_samples / 20)  # Full weight after 20 samples
    
    probability = (1 - weight) * prior + weight * historical_rate
    
    return probability


def estimate_expected_payoff(performance_tracker: ExpertPerformanceTracker,
                            task_type: str,
                            baseline_reward: float = 1.0) -> float:
    """
    Estimate expected payoff ratio.
    
    Args:
        performance_tracker: Performance history
        task_type: Type of task
        baseline_reward: Baseline reward for comparison
        
    Returns:
        Expected payoff ratio (gain/loss)
    """
    avg_reward = performance_tracker.get_average_reward(task_type)
    
    if avg_reward <= 0:
        return 0.1  # Small positive to avoid division issues
    
    # Payoff ratio = expected gain / expected loss
    payoff_ratio = avg_reward / baseline_reward
    
    return max(0.1, payoff_ratio)


class KellyAllocator:
    """
    Allocate RAM to expert brains using Kelly Criterion.
    
    Optimally balances expected return vs risk.
    """
    
    def __init__(self, total_ram_bytes: int = 4 * 1024**3):
        """
        Initialize allocator.
        
        Args:
            total_ram_bytes: Total RAM budget in bytes (default 4 GB)
        """
        self.total_ram = total_ram_bytes
        self.expert_trackers = {}  # expert_name -> ExpertPerformanceTracker
        self.current_allocation = {}  # expert_name -> bytes allocated
        self.expert_sizes = {}  # expert_name -> bytes required
        
    def register_expert(self, name: str, size_bytes: int):
        """
        Register an expert brain.
        
        Args:
            name: Expert identifier
            size_bytes: Memory footprint
        """
        self.expert_trackers[name] = ExpertPerformanceTracker()
        self.expert_sizes[name] = size_bytes
        
    def record_performance(self, expert_name: str, 
                          success: bool, 
                          reward: float,
                          task_type: str = 'default'):
        """Record expert performance."""
        if expert_name in self.expert_trackers:
            self.expert_trackers[expert_name].record(success, reward, task_type)
    
    def allocate(self, task_type: str,
                available_experts: List[str],
                safety_factor: float = 0.5) -> Dict[str, int]:
        """
        Allocate RAM to experts for a task.
        
        Args:
            task_type: Type of task
            available_experts: List of expert names to consider
            safety_factor: Kelly safety factor (0.25-0.5)
            
        Returns:
            Dict of expert_name -> bytes allocated
        """
        # Calculate Kelly fractions for each expert
        kelly_fractions = {}
        
        for expert_name in available_experts:
            if expert_name not in self.expert_trackers:
                continue
            
            tracker = self.expert_trackers[expert_name]
            
            # Estimate probability and payoff
            win_prob = estimate_win_probability(tracker, task_type)
            payoff_ratio = estimate_expected_payoff(tracker, task_type)
            
            # Calculate Kelly fraction
            kelly = calculate_kelly_fraction(win_prob, payoff_ratio, safety_factor)
            
            kelly_fractions[expert_name] = kelly
        
        # Normalize fractions to sum to 1
        total_kelly = sum(kelly_fractions.values())
        if total_kelly == 0:
            # Uniform allocation if no data
            total_kelly = len(kelly_fractions)
            kelly_fractions = {k: 1.0 for k in kelly_fractions}
        
        normalized_fractions = {
            k: v / total_kelly for k, v in kelly_fractions.items()
        }
        
        # Allocate RAM based on fractions
        allocation = {}
        remaining_ram = self.total_ram
        
        # Sort by fraction (highest first)
        sorted_experts = sorted(
            normalized_fractions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for expert_name, fraction in sorted_experts:
            # Calculate allocation
            target_allocation = fraction * self.total_ram
            expert_size = self.expert_sizes.get(expert_name, 0)
            
            # Can we fit this expert?
            if expert_size <= remaining_ram:
                allocation[expert_name] = expert_size
                remaining_ram -= expert_size
            else:
                # Can't fit - skip
                continue
        
        self.current_allocation = allocation
        return allocation
    
    def get_allocation_summary(self) -> Dict:
        """Get summary of current allocation."""
        total_allocated = sum(self.current_allocation.values())
        
        return {
            'total_ram': self.total_ram,
            'allocated': total_allocated,
            'free': self.total_ram - total_allocated,
            'utilization': total_allocated / self.total_ram,
            'num_experts_loaded': len(self.current_allocation)
        }
    
    def compare_strategies(self, task_type: str,
                          available_experts: List[str],
                          n_trials: int = 100) -> Dict:
        """
        Compare Kelly vs uniform vs random allocation.
        
        Args:
            task_type: Task type
            available_experts: Available experts
            n_trials: Number of trials to simulate
            
        Returns:
            Comparison results
        """
        results = {
            'kelly': [],
            'uniform': [],
            'random': []
        }
        
        for _ in range(n_trials):
            # Kelly allocation
            kelly_alloc = self.allocate(task_type, available_experts)
            kelly_score = self._evaluate_allocation(kelly_alloc, task_type)
            results['kelly'].append(kelly_score)
            
            # Uniform allocation
            uniform_alloc = self._uniform_allocation(available_experts)
            uniform_score = self._evaluate_allocation(uniform_alloc, task_type)
            results['uniform'].append(uniform_score)
            
            # Random allocation
            random_alloc = self._random_allocation(available_experts)
            random_score = self._evaluate_allocation(random_alloc, task_type)
            results['random'].append(random_score)
        
        # Summarize
        summary = {}
        for strategy, scores in results.items():
            summary[strategy] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return summary
    
    def _uniform_allocation(self, experts: List[str]) -> Dict[str, int]:
        """Allocate RAM uniformly."""
        allocation = {}
        ram_per_expert = self.total_ram // len(experts)
        
        for expert in experts:
            size = self.expert_sizes.get(expert, 0)
            if size <= ram_per_expert:
                allocation[expert] = size
        
        return allocation
    
    def _random_allocation(self, experts: List[str]) -> Dict[str, int]:
        """Allocate RAM randomly."""
        allocation = {}
        remaining = self.total_ram
        
        shuffled = np.random.permutation(experts)
        for expert in shuffled:
            size = self.expert_sizes.get(expert, 0)
            if size <= remaining:
                allocation[expert] = size
                remaining -= size
        
        return allocation
    
    def _evaluate_allocation(self, allocation: Dict[str, int],
                           task_type: str) -> float:
        """Evaluate quality of allocation."""
        score = 0.0
        
        for expert_name in allocation:
            tracker = self.expert_trackers.get(expert_name)
            if tracker:
                win_rate = tracker.get_win_rate(task_type)
                avg_reward = tracker.get_average_reward(task_type)
                score += win_rate * avg_reward
        
        return score
