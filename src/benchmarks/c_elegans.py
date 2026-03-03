"""
C. elegans Benchmarks

Tests based on actual C. elegans capabilities:
- Chemotaxis (gradient following)
- Tap withdrawal (simple reflex)
- Associative learning (pair stimulus with outcome)

Baseline: 302 neurons, ~10 trials to learn
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


class CElegansBenchmark(BiologicalBenchmark):
    """Benchmarks based on C. elegans (302 neurons)."""
    
    def __init__(self):
        super().__init__(BiologicalTier.C_ELEGANS)
    
    def test_chemotaxis(self, brain) -> BenchmarkResult:
        """
        Test gradient following (chemotaxis).
        
        C. elegans can follow chemical gradients to food.
        This is one of its most basic behaviors.
        """
        result = BenchmarkResult("Chemotaxis", self.tier)
        
        # Measure our system
        result.our_neurons = getattr(brain, 'n_neurons', 1000)
        result.our_trials = self._test_gradient_following(brain)
        result.our_performance = self._measure_gradient_performance(brain)
        result.our_energy = 1e-6  # Estimated for now
        
        result.compute_efficiency()
        
        return result
    
    def _test_gradient_following(self, brain) -> int:
        """
        Measure trials to learn gradient following.
        
        Task: Move toward higher concentration.
        """
        target_performance = 0.8
        max_trials = 200
        
        for trial in range(max_trials):
            # Simulate gradient environment
            position = np.random.rand() * 10  # 0-10 position
            gradient = np.array([position / 10])  # Concentration increases with position
            
            # Brain decides: move left (-1) or right (+1)
            action_values = self._simple_forward(brain, gradient)
            action = 1 if action_values[0] > 0 else -1
            
            # Reward if moving toward higher concentration
            correct = (action > 0 and position < 5) or (action < 0 and position > 5)
            
            # Simple learning
            if hasattr(brain, 'train_step'):
                reward = 1.0 if correct else -1.0
                brain.train_step(gradient, reward)
            
            # Check performance every 10 trials
            if trial % 10 == 0 and trial > 0:
                perf = self._measure_gradient_performance(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_gradient_performance(self, brain) -> float:
        """Test gradient following performance."""
        n_tests = 20
        correct = 0
        
        for _ in range(n_tests):
            position = np.random.rand() * 10
            gradient = np.array([position / 10])
            
            action_values = self._simple_forward(brain, gradient)
            action = 1 if action_values[0] > 0 else -1
            
            # Correct if moving toward higher concentration
            if (action > 0 and position < 5) or (action < 0 and position > 5):
                correct += 1
        
        return correct / n_tests
    
    def _simple_forward(self, brain, inputs: np.ndarray) -> np.ndarray:
        """Simple forward pass compatible with different brain types."""
        if hasattr(brain, 'forward'):
            # Pad inputs if needed
            if hasattr(brain, 'n_neurons'):
                padded = np.zeros(brain.n_neurons)
                padded[:len(inputs)] = inputs
                return brain.forward(padded)
            return brain.forward(inputs)
        else:
            # Fallback: random
            return np.random.randn(1)
    
    def test_associative_learning(self, brain) -> BenchmarkResult:
        """
        Test associative learning.
        
        C. elegans can learn to associate stimuli (e.g., odor + food).
        """
        result = BenchmarkResult("Associative Learning", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 1000)
        result.our_trials = self._test_association(brain)
        result.our_performance = self._measure_association_performance(brain)
        result.our_energy = 1e-6
        
        result.compute_efficiency()
        
        return result
    
    def _test_association(self, brain) -> int:
        """Measure trials to learn stimulus-reward association."""
        target_performance = 0.8
        max_trials = 200
        
        # Two stimuli: A (rewarded) and B (not rewarded)
        stimulus_a = np.array([1.0, 0.0])
        stimulus_b = np.array([0.0, 1.0])
        
        for trial in range(max_trials):
            # Random stimulus
            if np.random.rand() > 0.5:
                stimulus = stimulus_a
                reward = 1.0
            else:
                stimulus = stimulus_b
                reward = -1.0
            
            # Train
            if hasattr(brain, 'train_step'):
                brain.train_step(stimulus, reward)
            
            # Check performance
            if trial % 10 == 0 and trial > 0:
                perf = self._measure_association_performance(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_association_performance(self, brain) -> float:
        """Test association performance."""
        n_tests = 20
        correct = 0
        
        stimulus_a = np.array([1.0, 0.0])
        stimulus_b = np.array([0.0, 1.0])
        
        for _ in range(n_tests):
            if np.random.rand() > 0.5:
                stimulus = stimulus_a
                expected_positive = True
            else:
                stimulus = stimulus_b
                expected_positive = False
            
            response = self._simple_forward(brain, stimulus)
            is_positive = response[0] > 0
            
            if is_positive == expected_positive:
                correct += 1
        
        return correct / n_tests


def test_c_elegans_benchmarks():
    """Test C. elegans benchmarks."""
    print("\n" + "="*60)
    print("C. ELEGANS BENCHMARKS")
    print("="*60)
    
    # Create simple test brain
    class SimpleBrain:
        def __init__(self):
            self.n_neurons = 500
            self.weights = np.random.randn(self.n_neurons, self.n_neurons) * 0.1
        
        def forward(self, inputs):
            padded = np.zeros(self.n_neurons)
            padded[:len(inputs)] = inputs
            return np.tanh(self.weights @ padded)[:1]
        
        def train_step(self, state, reward):
            # Simple Hebbian-like update
            activations = self.forward(state)
            delta = 0.01 * reward * np.outer(activations, state)
            # Update small portion of weights
            self.weights[:len(delta), :len(delta[0])] += delta
    
    brain = SimpleBrain()
    benchmark = CElegansBenchmark()
    
    # Test 1: Chemotaxis
    print("\n1. CHEMOTAXIS (Gradient Following)")
    print("-" * 60)
    result1 = benchmark.test_chemotaxis(brain)
    summary1 = result1.get_summary()
    
    print(f"Biological: {summary1['biological']['neurons']} neurons, {summary1['biological']['trials']} trials")
    print(f"Our system: {summary1['our_system']['neurons']} neurons, {summary1['our_system']['trials']} trials")
    print(f"Performance: {summary1['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary1['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary1['efficiency']['learning']:.2f}x")
    print(f"Status: {summary1['status']}")
    
    # Test 2: Associative Learning
    print("\n2. ASSOCIATIVE LEARNING")
    print("-" * 60)
    brain2 = SimpleBrain()  # Fresh brain
    result2 = benchmark.test_associative_learning(brain2)
    summary2 = result2.get_summary()
    
    print(f"Biological: {summary2['biological']['neurons']} neurons, {summary2['biological']['trials']} trials")
    print(f"Our system: {summary2['our_system']['neurons']} neurons, {summary2['our_system']['trials']} trials")
    print(f"Performance: {summary2['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary2['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary2['efficiency']['learning']:.2f}x")
    print(f"Status: {summary2['status']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_neuron_eff = (result1.neuron_efficiency + result2.neuron_efficiency) / 2
    avg_learning_eff = (result1.learning_efficiency + result2.learning_efficiency) / 2
    
    print(f"\nAverage Efficiency vs C. elegans:")
    print(f"  Neurons: {avg_neuron_eff:.2f}x")
    print(f"  Learning: {avg_learning_eff:.2f}x")
    
    if avg_neuron_eff < 1.0:
        print(f"\n⚠️  Using {1/avg_neuron_eff:.1f}x more neurons than C. elegans")
    if avg_learning_eff < 1.0:
        print(f"⚠️  Learning {1/avg_learning_eff:.1f}x slower than C. elegans")
    
    print("\n✓ C. elegans benchmarks complete!")


if __name__ == "__main__":
    test_c_elegans_benchmarks()
