"""
Fruit Fly (Drosophila) Benchmarks

Tests based on actual fruit fly capabilities:
- Odor discrimination and learning
- Visual pattern recognition
- T-maze navigation
- One-shot learning

Baseline: 100,000 neurons, ~10 trials to learn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.benchmarks.biological import (
    BiologicalBenchmark,
    BiologicalTier,
    BenchmarkResult
)


class FruitFlyBenchmark(BiologicalBenchmark):
    """Benchmarks based on Drosophila (100K neurons)."""
    
    def __init__(self):
        super().__init__(BiologicalTier.FRUIT_FLY)
    
    def test_odor_learning(self, brain) -> BenchmarkResult:
        """
        Test odor discrimination and learning.
        
        Fruit flies can learn to associate odors with rewards/punishments
        in as few as 10 trials (one-shot learning in some cases).
        """
        result = BenchmarkResult("Odor Learning", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 1000)
        result.our_trials = self._test_odor_association(brain)
        result.our_performance = self._measure_odor_performance(brain)
        result.our_energy = 1e-6
        
        result.compute_efficiency()
        return result
    
    def _test_odor_association(self, brain) -> int:
        """
        Measure trials to learn odor-reward association.
        
        Task: Learn which of 4 odors is rewarded.
        """
        target_performance = 0.8
        max_trials = 200
        
        # 4 different odors (one-hot encoded)
        odors = [
            np.array([1, 0, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 0, 1])
        ]
        
        # Odor 0 is rewarded
        rewarded_odor = 0
        
        for trial in range(max_trials):
            # Present random odor
            odor_idx = np.random.randint(0, 4)
            odor = odors[odor_idx]
            
            # Reward if correct odor
            reward = 1.0 if odor_idx == rewarded_odor else -0.5
            
            # Train
            if hasattr(brain, 'train_step'):
                brain.train_step(odor, reward)
            
            # Check performance every 10 trials
            if trial % 10 == 0 and trial > 0:
                perf = self._measure_odor_performance(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_odor_performance(self, brain) -> float:
        """Test odor discrimination performance."""
        n_tests = 20
        correct = 0
        
        odors = [
            np.array([1, 0, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 0, 1])
        ]
        rewarded_odor = 0
        
        for _ in range(n_tests):
            odor_idx = np.random.randint(0, 4)
            odor = odors[odor_idx]
            
            response = self._simple_forward(brain, odor)
            
            # Correct if highest response to rewarded odor
            if odor_idx == rewarded_odor and response[0] > 0:
                correct += 1
            elif odor_idx != rewarded_odor and response[0] <= 0:
                correct += 1
        
        return correct / n_tests
    
    def test_visual_pattern(self, brain) -> BenchmarkResult:
        """
        Test visual pattern recognition.
        
        Fruit flies can distinguish visual patterns (T-shapes, stripes, etc.)
        """
        result = BenchmarkResult("Visual Pattern Recognition", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 1000)
        result.our_trials = self._test_pattern_learning(brain)
        result.our_performance = self._measure_pattern_performance(brain)
        result.our_energy = 1e-6
        
        result.compute_efficiency()
        return result
    
    def _test_pattern_learning(self, brain) -> int:
        """Learn to distinguish between two visual patterns."""
        target_performance = 0.8
        max_trials = 200
        
        for trial in range(max_trials):
            # Two patterns: vertical (rewarded) vs horizontal (not rewarded)
            if np.random.rand() > 0.5:
                # Vertical pattern
                pattern = np.array([1, 1, 0, 0, 1, 1, 0, 0])
                reward = 1.0
            else:
                # Horizontal pattern
                pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])
                reward = -0.5
            
            # Train
            if hasattr(brain, 'train_step'):
                brain.train_step(pattern, reward)
            
            # Check performance
            if trial % 10 == 0 and trial > 0:
                perf = self._measure_pattern_performance(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_pattern_performance(self, brain) -> float:
        """Test pattern discrimination."""
        n_tests = 20
        correct = 0
        
        for _ in range(n_tests):
            if np.random.rand() > 0.5:
                pattern = np.array([1, 1, 0, 0, 1, 1, 0, 0])
                is_rewarded = True
            else:
                pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0])
                is_rewarded = False
            
            response = self._simple_forward(brain, pattern)
            
            if (response[0] > 0) == is_rewarded:
                correct += 1
        
        return correct / n_tests
    
    def test_spatial_navigation(self, brain) -> BenchmarkResult:
        """
        Test T-maze navigation.
        
        Fruit flies can learn to navigate T-mazes to find food.
        """
        result = BenchmarkResult("T-Maze Navigation", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 1000)
        result.our_trials = self._test_tmaze(brain)
        result.our_performance = self._measure_tmaze_performance(brain)
        result.our_energy = 1e-6
        
        result.compute_efficiency()
        return result
    
    def _test_tmaze(self, brain) -> int:
        """Learn T-maze (left or right turn to find reward)."""
        target_performance = 0.8
        max_trials = 200
        
        # Correct choice is always "right" (1)
        correct_choice = 1
        
        for trial in range(max_trials):
            # State: position in maze (simplified)
            state = np.array([1.0, 0.0])  # At T-junction
            
            # Brain chooses left (0) or right (1)
            response = self._simple_forward(brain, state)
            choice = 1 if response[0] > 0 else 0
            
            # Reward
            reward = 1.0 if choice == correct_choice else -0.5
            
            # Train
            if hasattr(brain, 'train_step'):
                brain.train_step(state, reward)
            
            # Check performance
            if trial % 10 == 0 and trial > 0:
                perf = self._measure_tmaze_performance(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_tmaze_performance(self, brain) -> float:
        """Test T-maze performance."""
        n_tests = 20
        correct = 0
        
        state = np.array([1.0, 0.0])
        correct_choice = 1
        
        for _ in range(n_tests):
            response = self._simple_forward(brain, state)
            choice = 1 if response[0] > 0 else 0
            
            if choice == correct_choice:
                correct += 1
        
        return correct / n_tests
    
    def _simple_forward(self, brain, inputs: np.ndarray) -> np.ndarray:
        """Simple forward pass compatible with different brain types."""
        if hasattr(brain, 'forward'):
            if hasattr(brain, 'n_neurons'):
                padded = np.zeros(brain.n_neurons)
                padded[:len(inputs)] = inputs
                return brain.forward(padded)
            return brain.forward(inputs)
        else:
            return np.random.randn(1)


def test_fruit_fly_benchmarks():
    """Test fruit fly benchmarks."""
    print("\n" + "="*60)
    print("FRUIT FLY BENCHMARKS (100K neurons baseline)")
    print("="*60)
    
    # Create test brain
    class SimpleBrain:
        def __init__(self):
            self.n_neurons = 1000
            self.weights = np.random.randn(self.n_neurons, self.n_neurons) * 0.1
        
        def forward(self, inputs):
            padded = np.zeros(self.n_neurons)
            padded[:len(inputs)] = inputs
            return np.tanh(self.weights @ padded)[:1]
        
        def train_step(self, state, reward):
            activations = self.forward(state)
            delta = 0.01 * reward * np.outer(activations, state)
            self.weights[:len(delta), :len(delta[0])] += delta
    
    benchmark = FruitFlyBenchmark()
    
    # Test 1: Odor Learning
    print("\n1. ODOR LEARNING")
    print("-" * 60)
    brain1 = SimpleBrain()
    result1 = benchmark.test_odor_learning(brain1)
    summary1 = result1.get_summary()
    
    print(f"Biological: {summary1['biological']['neurons']:,} neurons, {summary1['biological']['trials']} trials")
    print(f"Our system: {summary1['our_system']['neurons']:,} neurons, {summary1['our_system']['trials']} trials")
    print(f"Performance: {summary1['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary1['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary1['efficiency']['learning']:.2f}x")
    print(f"Status: {summary1['status']}")
    
    # Test 2: Visual Pattern
    print("\n2. VISUAL PATTERN RECOGNITION")
    print("-" * 60)
    brain2 = SimpleBrain()
    result2 = benchmark.test_visual_pattern(brain2)
    summary2 = result2.get_summary()
    
    print(f"Biological: {summary2['biological']['neurons']:,} neurons, {summary2['biological']['trials']} trials")
    print(f"Our system: {summary2['our_system']['neurons']:,} neurons, {summary2['our_system']['trials']} trials")
    print(f"Performance: {summary2['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary2['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary2['efficiency']['learning']:.2f}x")
    print(f"Status: {summary2['status']}")
    
    # Test 3: T-Maze
    print("\n3. T-MAZE NAVIGATION")
    print("-" * 60)
    brain3 = SimpleBrain()
    result3 = benchmark.test_spatial_navigation(brain3)
    summary3 = result3.get_summary()
    
    print(f"Biological: {summary3['biological']['neurons']:,} neurons, {summary3['biological']['trials']} trials")
    print(f"Our system: {summary3['our_system']['neurons']:,} neurons, {summary3['our_system']['trials']} trials")
    print(f"Performance: {summary3['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary3['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary3['efficiency']['learning']:.2f}x")
    print(f"Status: {summary3['status']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results = [result1, result2, result3]
    avg_neuron_eff = sum(r.neuron_efficiency for r in results) / len(results)
    avg_learning_eff = sum(r.learning_efficiency for r in results) / len(results)
    
    print(f"\nAverage Efficiency vs Fruit Fly:")
    print(f"  Neurons: {avg_neuron_eff:.2f}x")
    print(f"  Learning: {avg_learning_eff:.2f}x")
    
    if avg_neuron_eff < 1.0:
        print(f"\nGap: Using {1/avg_neuron_eff:.1f}x more neurons than fruit fly")
    if avg_learning_eff < 1.0:
        print(f"Gap: Learning {1/avg_learning_eff:.1f}x slower than fruit fly")
    
    print("\nFruit fly benchmarks complete!")


if __name__ == "__main__":
    test_fruit_fly_benchmarks()
