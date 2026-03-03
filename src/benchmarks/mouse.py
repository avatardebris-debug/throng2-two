"""
Half-Mouse Benchmarks (10M neurons)

Tests based on actual mouse capabilities:
- Spatial navigation (Morris water maze)
- Fear conditioning (associative learning)
- Object recognition (visual memory)

Baseline: 10,000,000 neurons (cortex), ~50 trials to learn

Uses EFFICIENT sparse matrix initialization to avoid initialization loops.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from scipy.sparse import random as sparse_random
import time
from src.benchmarks.biological import (
    BiologicalBenchmark,
    BiologicalTier,
    BenchmarkResult
)


class MouseBenchmark(BiologicalBenchmark):
    """Benchmarks based on Half-Mouse (10M neurons)."""
    
    def __init__(self):
        super().__init__(BiologicalTier.MOUSE)
    
    def test_spatial_navigation(self, brain) -> BenchmarkResult:
        """
        Test spatial navigation (Morris water maze).
        
        Mice can learn to navigate to a hidden platform using spatial cues.
        This tests hippocampal-dependent spatial learning.
        """
        result = BenchmarkResult("Spatial Navigation", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 10000)
        result.our_trials = self._test_navigation_learning(brain)
        result.our_performance = self._measure_navigation_performance(brain)
        result.our_energy = 0.02
        
        result.compute_efficiency()
        return result
    
    def _test_navigation_learning(self, brain) -> int:
        """
        Measure trials to learn spatial navigation.
        
        Task: Learn to navigate from random start to goal location.
        """
        target_performance = 0.75
        max_trials = 500
        
        # Fixed goal location
        goal = np.array([0.8, 0.8])
        
        for trial in range(max_trials):
            # Random start position
            start = np.random.rand(2)
            
            # Spatial cues (landmarks)
            cue1 = np.array([0.2, 0.9])
            cue2 = np.array([0.9, 0.2])
            
            # Input: current position + cue positions
            state = np.concatenate([start, cue1, cue2])
            
            # Get action from brain
            action = self._simple_forward(brain, state)[:2]
            
            # Move toward action
            new_pos = start + action * 0.1
            new_pos = np.clip(new_pos, 0, 1)
            
            # Reward based on distance to goal
            distance = np.linalg.norm(new_pos - goal)
            reward = 1.0 if distance < 0.15 else -0.1 * distance
            
            # Train
            if hasattr(brain, 'train_step'):
                brain.train_step(state, reward)
            
            # Check performance
            if trial % 25 == 0 and trial > 0:
                perf = self._measure_navigation_performance(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_navigation_performance(self, brain) -> float:
        """Test navigation performance."""
        n_tests = 30
        successes = 0
        goal = np.array([0.8, 0.8])
        
        for _ in range(n_tests):
            start = np.random.rand(2)
            cue1 = np.array([0.2, 0.9])
            cue2 = np.array([0.9, 0.2])
            
            # Simulate navigation (5 steps)
            pos = start.copy()
            for _ in range(5):
                state = np.concatenate([pos, cue1, cue2])
                action = self._simple_forward(brain, state)[:2]
                pos = pos + action * 0.1
                pos = np.clip(pos, 0, 1)
            
            # Success if within threshold
            if np.linalg.norm(pos - goal) < 0.15:
                successes += 1
        
        return successes / n_tests
    
    def test_fear_conditioning(self, brain) -> BenchmarkResult:
        """
        Test fear conditioning (associative learning).
        
        Mice can learn to associate a neutral stimulus (tone) with
        an aversive stimulus (shock). This tests amygdala-dependent learning.
        """
        result = BenchmarkResult("Fear Conditioning", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 10000)
        result.our_trials = self._test_conditioning_learning(brain)
        result.our_performance = self._measure_conditioning_performance(brain)
        result.our_energy = 0.02
        
        result.compute_efficiency()
        return result
    
    def _test_conditioning_learning(self, brain) -> int:
        """
        Measure trials to learn fear conditioning.
        
        Task: Learn that tone predicts shock.
        """
        target_performance = 0.80
        max_trials = 500
        
        for trial in range(max_trials):
            # 70% of time: tone + shock (CS+)
            # 30% of time: tone alone (CS-)
            paired = np.random.rand() < 0.7
            
            # Tone stimulus
            tone = np.array([1.0, 0.0])
            
            # Shock (if paired)
            shock = 1.0 if paired else 0.0
            
            # Present tone
            response = self._simple_forward(brain, tone)
            
            # Train: learn to predict shock from tone
            reward = -shock  # Negative reward for shock
            if hasattr(brain, 'train_step'):
                brain.train_step(tone, reward)
            
            # Check performance
            if trial % 25 == 0 and trial > 0:
                perf = self._measure_conditioning_performance(brain)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_conditioning_performance(self, brain) -> float:
        """Test conditioning performance."""
        n_tests = 30
        correct = 0
        
        for _ in range(n_tests):
            tone = np.array([1.0, 0.0])
            response = self._simple_forward(brain, tone)
            
            # Should show fear response (high activation)
            if response[0] > 0.5:
                correct += 1
        
        return correct / n_tests
    
    def test_object_recognition(self, brain) -> BenchmarkResult:
        """
        Test object recognition (visual memory).
        
        Mice can recognize novel vs familiar objects.
        This tests perirhinal cortex-dependent recognition memory.
        """
        result = BenchmarkResult("Object Recognition", self.tier)
        
        result.our_neurons = getattr(brain, 'n_neurons', 10000)
        result.our_trials = self._test_recognition_learning(brain)
        result.our_performance = self._measure_recognition_performance(brain)
        result.our_energy = 0.02
        
        result.compute_efficiency()
        return result
    
    def _test_recognition_learning(self, brain) -> int:
        """
        Measure trials to learn object recognition.
        
        Task: Distinguish familiar from novel objects.
        """
        target_performance = 0.75
        max_trials = 500
        
        # Store "familiar" objects
        familiar_objects = []
        
        for trial in range(max_trials):
            if trial % 50 == 0:
                # Add new familiar object
                obj = np.random.randn(8)
                familiar_objects.append(obj)
            
            if len(familiar_objects) > 0:
                # 50% familiar, 50% novel
                is_novel = np.random.rand() < 0.5
                
                if is_novel:
                    obj = np.random.randn(8)
                    expected_response = 1.0  # Novel
                else:
                    obj = familiar_objects[np.random.randint(len(familiar_objects))]
                    expected_response = 0.0  # Familiar
                
                # Get response
                response = self._simple_forward(brain, obj)
                prediction = 1.0 if response[0] > 0.5 else 0.0
                
                # Reward
                correct = (prediction == expected_response)
                reward = 1.0 if correct else -0.5
                
                if hasattr(brain, 'train_step'):
                    brain.train_step(obj, reward)
            
            # Check performance
            if trial % 25 == 0 and trial > 0:
                perf = self._measure_recognition_performance(brain, familiar_objects)
                if perf >= target_performance:
                    return trial + 1
        
        return max_trials
    
    def _measure_recognition_performance(self, brain, familiar_objects=None) -> float:
        """Test recognition performance."""
        if familiar_objects is None or len(familiar_objects) == 0:
            return 0.5  # Random performance
        
        n_tests = 30
        correct = 0
        
        for _ in range(n_tests):
            is_novel = np.random.rand() < 0.5
            
            if is_novel:
                obj = np.random.randn(8)
                expected = 1.0
            else:
                obj = familiar_objects[np.random.randint(len(familiar_objects))]
                expected = 0.0
            
            response = self._simple_forward(brain, obj)
            prediction = 1.0 if response[0] > 0.5 else 0.0
            
            if prediction == expected:
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
            return np.random.randn(3)


def create_efficient_mouse_brain(n_neurons: int = 10_000_000, density: float = 0.01):
    """
    Create sparse mouse-scale brain efficiently using scipy.
    
    Uses sparse matrix generation to avoid initialization loops.
    Lower density (1%) for mouse scale to keep memory manageable.
    """
    print(f"\nCreating {n_neurons:,} neuron mouse brain with {density:.1%} density...")
    start = time.time()
    
    # Use scipy's efficient sparse random matrix generator
    weights = sparse_random(n_neurons, n_neurons, density=density, format='csr')
    weights.data = weights.data * 0.01  # Scale to reasonable values
    
    elapsed = time.time() - start
    
    # Stats
    memory_mb = (weights.data.nbytes + weights.indices.nbytes + 
                weights.indptr.nbytes) / 1024**2
    
    print(f"  Created in {elapsed:.2f}s")
    print(f"  Connections: {weights.nnz:,}")
    print(f"  Memory: {memory_mb:.1f} MB ({memory_mb/1024:.2f} GB)")
    
    return weights


def test_mouse_benchmarks():
    """Test mouse benchmarks with efficient initialization."""
    print("\n" + "="*60)
    print("HALF-MOUSE BENCHMARKS (10M neurons baseline)")
    print("="*60)
    
    # Create test brain (1M neurons - honeybee scale)
    class SparseBrain:
        def __init__(self, n_neurons=1_000_000):
            self.n_neurons = n_neurons
            print(f"\nInitializing sparse brain with {n_neurons:,} neurons...")
            # Use efficient sparse initialization (very low density for 1M neurons)
            self.weights = create_efficient_mouse_brain(n_neurons, density=0.001)
        
        def forward(self, inputs):
            padded = np.zeros(self.n_neurons)
            padded[:len(inputs)] = inputs
            # Sparse matrix multiplication
            output = self.weights @ padded
            return np.tanh(output)[:3]
        
        def train_step(self, state, reward):
            # Simple Hebbian-like update
            activations = self.forward(state)
            # Only update a small subset (efficient)
            n_update = min(100, self.n_neurons)
            update_idx = np.random.choice(self.n_neurons, n_update, replace=False)
            
            for idx in update_idx:
                row = self.weights.getrow(idx).toarray().flatten()
                delta = 0.001 * reward * activations[0] * state[:len(row)]
                row[:len(delta)] += delta
                self.weights[idx] = row
    
    benchmark = MouseBenchmark()
    
    # Test 1: Spatial Navigation
    print("\n1. SPATIAL NAVIGATION (Morris Water Maze)")
    print("-" * 60)
    brain1 = SparseBrain()  # Uses default 1M neurons
    result1 = benchmark.test_spatial_navigation(brain1)
    summary1 = result1.get_summary()
    
    print(f"Biological: {summary1['biological']['neurons']:,} neurons, {summary1['biological']['trials']} trials")
    print(f"Our system: {summary1['our_system']['neurons']:,} neurons, {summary1['our_system']['trials']} trials")
    print(f"Performance: {summary1['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary1['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary1['efficiency']['learning']:.2f}x")
    print(f"Status: {summary1['status']}")
    
    # Test 2: Fear Conditioning
    print("\n2. FEAR CONDITIONING (Associative Learning)")
    print("-" * 60)
    brain2 = SparseBrain()  # Uses default 1M neurons
    result2 = benchmark.test_fear_conditioning(brain2)
    summary2 = result2.get_summary()
    
    print(f"Biological: {summary2['biological']['neurons']:,} neurons, {summary2['biological']['trials']} trials")
    print(f"Our system: {summary2['our_system']['neurons']:,} neurons, {summary2['our_system']['trials']} trials")
    print(f"Performance: {summary2['our_system']['performance']:.1%}")
    print(f"\nEfficiency:")
    print(f"  Neurons: {summary2['efficiency']['neurons']:.2f}x")
    print(f"  Learning: {summary2['efficiency']['learning']:.2f}x")
    print(f"Status: {summary2['status']}")
    
    # Test 3: Object Recognition
    print("\n3. OBJECT RECOGNITION (Visual Memory)")
    print("-" * 60)
    brain3 = SparseBrain()  # Uses default 1M neurons
    result3 = benchmark.test_object_recognition(brain3)
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
    
    print(f"\nAverage Efficiency vs Half-Mouse:")
    print(f"  Neurons: {avg_neuron_eff:.2f}x")
    print(f"  Learning: {avg_learning_eff:.2f}x")
    
    if avg_neuron_eff >= 1.0:
        print(f"\n✓ Exceeds: Using {avg_neuron_eff:.0f}x FEWER neurons than half-mouse!")
    else:
        print(f"\n⚠️  Gap: Using {1/avg_neuron_eff:.1f}x more neurons than half-mouse")
    
    if avg_learning_eff < 1.0:
        print(f"⚠️  Gap: Learning {1/avg_learning_eff:.1f}x slower than half-mouse")
    
    print("\n✓ Half-mouse benchmarks complete!")
    print("\nNote: Tests use 1M neurons for realistic testing. Scale to 10M for full benchmark.")


if __name__ == "__main__":
    test_mouse_benchmarks()
