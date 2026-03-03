"""
Honeybee Benchmarks

Tests based on actual honeybee capabilities:
- Delayed match-to-sample (working memory)
- Same/different concept learning (abstract reasoning)
- Spatial encoding (simplified waggle dance)

Baseline: 1,000,000 neurons, ~20 trials to learn
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


class HoneybeeBenchmark(BiologicalBenchmark):
    """Benchmarks based on Honeybee (1M neurons)."""
    
    def __init__(self):
        super().__init__(BiologicalTier.HONEYBEE)
    
    def test_delayed_match_to_sample(self, brain) -> BenchmarkResult:
        """
        Test working memory (delayed match-to-sample).
        
        Honeybees can remember a sample stimulus and match it after a delay.
        This tests working memory capacity.
        """
        result = BenchmarkResult("Delayed Match-to-Sample", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 1000)
        result.our_trials = self._test_working_memory(brain)
        result.our_performance = self._measure_working_memory(brain)
        result.our_energy = 1e-5
        
        result.compute_efficiency()
        return result
    
    def _test_working_memory(self, brain) -> int:
        """
        Measure trials to learn delayed matching.
        
        Task: See sample, delay, choose matching stimulus.
        """
        target_performance = 0.75
        max_trials = 300
        
        for trial in range(max_trials):
            # Sample stimulus (color: red=1 or blue=2)
            sample = np.random.randint(1, 3)
            sample_vec = np.array([1, 0]) if sample == 1 else np.array([0, 1])
            
            # Present sample
            if hasattr(brain, 'forward'):
                brain.forward(sample_vec)
            
            # Delay (simulated by noise)
            delay_noise = np.random.randn(2) * 0.1
            
            # Test: which matches sample?
            choice1 = np.array([1, 0])
            choice2 = np.array([0, 1])
            
            # Brain should choose matching one
            response1 = self._simple_forward(brain, choice1 + delay_noise)
            response2 = self._simple_forward(brain, choice2 + delay_noise)
            
            chosen = 1 if response1[0] > response2[0] else 2
            correct = (chosen == sample)
            
            # Train
            reward = 1.0 if correct else -0.5
            if hasattr(brain, 'train_step'):
                brain.train_step(sample_vec, reward)
            
            # Check performance
            if trial % 20 == 0 and trial > 0:
                perf = self._measure_working_memory(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_working_memory(self, brain) -> float:
        """Test working memory performance."""
        n_tests = 20
        correct = 0
        
        for _ in range(n_tests):
            sample = np.random.randint(1, 3)
            sample_vec = np.array([1, 0]) if sample == 1 else np.array([0, 1])
            
            # Delay
            delay_noise = np.random.randn(2) * 0.1
            
            # Test
            choice1 = np.array([1, 0])
            choice2 = np.array([0, 1])
            
            response1 = self._simple_forward(brain, choice1 + delay_noise)
            response2 = self._simple_forward(brain, choice2 + delay_noise)
            
            chosen = 1 if response1[0] > response2[0] else 2
            
            if chosen == sample:
                correct += 1
        
        return correct / n_tests
    
    def test_same_different_concept(self, brain) -> BenchmarkResult:
        """
        Test abstract concept learning (same vs different).
        
        Honeybees can learn the abstract concept of "same" vs "different"
        and generalize to novel stimuli. This is remarkable!
        """
        result = BenchmarkResult("Same/Different Concept", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 1000)
        result.our_trials = self._test_abstract_concept(brain)
        result.our_performance = self._measure_abstract_concept(brain)
        result.our_energy = 1e-5
        
        result.compute_efficiency()
        return result
    
    def _test_abstract_concept(self, brain) -> int:
        """
        Measure trials to learn same/different concept.
        
        Task: Learn that "same" patterns are rewarded.
        """
        target_performance = 0.75
        max_trials = 300
        
        for trial in range(max_trials):
            # Generate two patterns
            if np.random.rand() > 0.5:
                # SAME: both patterns identical
                pattern = np.random.randint(0, 2, size=4)
                pattern1 = pattern.copy()
                pattern2 = pattern.copy()
                is_same = True
            else:
                # DIFFERENT: patterns differ
                pattern1 = np.random.randint(0, 2, size=4)
                pattern2 = np.random.randint(0, 2, size=4)
                # Ensure they're actually different
                while np.array_equal(pattern1, pattern2):
                    pattern2 = np.random.randint(0, 2, size=4)
                is_same = False
            
            # Concatenate patterns
            combined = np.concatenate([pattern1, pattern2])
            
            # Train
            reward = 1.0 if is_same else -0.5
            if hasattr(brain, 'train_step'):
                brain.train_step(combined, reward)
            
            # Check performance
            if trial % 20 == 0 and trial > 0:
                perf = self._measure_abstract_concept(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_abstract_concept(self, brain) -> float:
        """Test same/different concept."""
        n_tests = 20
        correct = 0
        
        for _ in range(n_tests):
            if np.random.rand() > 0.5:
                pattern = np.random.randint(0, 2, size=4)
                combined = np.concatenate([pattern, pattern])
                is_same = True
            else:
                pattern1 = np.random.randint(0, 2, size=4)
                pattern2 = np.random.randint(0, 2, size=4)
                while np.array_equal(pattern1, pattern2):
                    pattern2 = np.random.randint(0, 2, size=4)
                combined = np.concatenate([pattern1, pattern2])
                is_same = False
            
            response = self._simple_forward(brain, combined)
            prediction_same = response[0] > 0
            
            if prediction_same == is_same:
                correct += 1
        
        return correct / n_tests
    
    def test_spatial_encoding(self, brain) -> BenchmarkResult:
        """
        Test spatial encoding (simplified waggle dance).
        
        Honeybees encode spatial information (distance, direction) in their
        waggle dance. We test simplified spatial encoding.
        """
        result = BenchmarkResult("Spatial Encoding", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 1000)
        result.our_trials = self._test_spatial_learning(brain)
        result.our_performance = self._measure_spatial_performance(brain)
        result.our_energy = 1e-5
        
        result.compute_efficiency()
        return result
    
    def _test_spatial_learning(self, brain) -> int:
        """Learn to encode spatial location."""
        target_performance = 0.75
        max_trials = 300
        
        for trial in range(max_trials):
            # Random location (x, y in [0, 1])
            location = np.random.rand(2)
            
            # Encode as direction (angle) and distance
            angle = np.arctan2(location[1], location[0])
            distance = np.linalg.norm(location)
            
            # Input: location
            # Output should encode: angle, distance
            target_output = np.array([np.cos(angle), np.sin(angle), distance])
            
            # Train
            if hasattr(brain, 'forward'):
                output = brain.forward(location)
                # Simple error-based reward
                error = np.mean((output[:3] - target_output) ** 2)
                reward = 1.0 / (1.0 + error)
                
                if hasattr(brain, 'train_step'):
                    brain.train_step(location, reward)
            
            # Check performance
            if trial % 20 == 0 and trial > 0:
                perf = self._measure_spatial_performance(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_spatial_performance(self, brain) -> float:
        """Test spatial encoding accuracy."""
        n_tests = 20
        total_error = 0
        
        for _ in range(n_tests):
            location = np.random.rand(2)
            
            angle = np.arctan2(location[1], location[0])
            distance = np.linalg.norm(location)
            target = np.array([np.cos(angle), np.sin(angle), distance])
            
            output = self._simple_forward(brain, location)
            error = np.mean((output[:3] - target) ** 2)
            total_error += error
        
        avg_error = total_error / n_tests
        # Convert error to performance (0-1)
        performance = 1.0 / (1.0 + avg_error)
        
        return performance
    
    def _simple_forward(self, brain, inputs: np.ndarray) -> np.ndarray:
        """Simple forward pass compatible with different brain types."""
        if hasattr(brain, 'forward'):
            if hasattr(brain, 'n_neurons'):
                padded = np.zeros(brain.n_neurons)
                padded[:len(inputs)] = inputs
                return brain.forward(padded)
            return brain.forward(inputs)
        else:
            return np.random.randn(3)


def test_honeybee_benchmarks():
    """Test honeybee benchmarks."""
    print("\n" + "="*60)
    print("HONEYBEE BENCHMARKS (1M neurons baseline)")
    print("="*60)
    
    # Create test brain
    class SimpleBrain:
        def __init__(self):
            self.n_neurons = 2000  # Slightly larger for abstract tasks
            self.weights = np.random.randn(self.n_neurons, self.n_neurons) * 0.1
        
        def forward(self, inputs):
            padded = np.zeros(self.n_neurons)
            padded[:len(inputs)] = inputs
            return np.tanh(self.weights @ padded)[:3]
        
        def train_step(self, state, reward):
            activations = self.forward(state)
            delta = 0.01 * reward * np.outer(activations, state)
            self.weights[:len(delta), :len(delta[0])] += delta
    
    benchmark = HoneybeeBenchmark()
    
    # Test 1: Working Memory
    print("\n1. DELAYED MATCH-TO-SAMPLE (Working Memory)")
    print("-" * 60)
    brain1 = SimpleBrain()
    result1 = benchmark.test_delayed_match_to_sample(brain1)
    summary1 = result1.get_summary()
    
    print(f"Biological: {summary1['biological']['neurons']:,} neurons, {summary1['biological']['trials']} trials")
    print(f"Our system: {summary1['our_system']['neurons']:,} neurons, {summary1['our_system']['trials']} trials")
    print(f"Performance: {summary1['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary1['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary1['efficiency']['learning']:.2f}x")
    print(f"Status: {summary1['status']}")
    
    # Test 2: Abstract Concept
    print("\n2. SAME/DIFFERENT CONCEPT (Abstract Reasoning)")
    print("-" * 60)
    brain2 = SimpleBrain()
    result2 = benchmark.test_same_different_concept(brain2)
    summary2 = result2.get_summary()
    
    print(f"Biological: {summary2['biological']['neurons']:,} neurons, {summary2['biological']['trials']} trials")
    print(f"Our system: {summary2['our_system']['neurons']:,} neurons, {summary2['our_system']['trials']} trials")
    print(f"Performance: {summary2['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary2['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary2['efficiency']['learning']:.2f}x")
    print(f"Status: {summary2['status']}")
    
    # Test 3: Spatial Encoding
    print("\n3. SPATIAL ENCODING (Waggle Dance)")
    print("-" * 60)
    brain3 = SimpleBrain()
    result3 = benchmark.test_spatial_encoding(brain3)
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
    
    print(f"\nAverage Efficiency vs Honeybee:")
    print(f"  Neurons: {avg_neuron_eff:.2f}x")
    print(f"  Learning: {avg_learning_eff:.2f}x")
    
    if avg_neuron_eff >= 1.0:
        print(f"\nExceeds: Using {avg_neuron_eff:.0f}x FEWER neurons than honeybee!")
    else:
        print(f"\nGap: Using {1/avg_neuron_eff:.1f}x more neurons than honeybee")
    
    if avg_learning_eff < 1.0:
        print(f"Gap: Learning {1/avg_learning_eff:.1f}x slower than honeybee")
    
    print("\nHoneybee benchmarks complete!")


if __name__ == "__main__":
    test_honeybee_benchmarks()
