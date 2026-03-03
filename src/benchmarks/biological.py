"""
Phase 3f: Biological Benchmarking Framework

Compare Thronglet Brain against biological organisms to identify
optimization opportunities and measure progress toward biological efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class BiologicalTier(Enum):
    """Biological complexity tiers."""
    C_ELEGANS = "c_elegans"
    FRUIT_FLY = "fruit_fly"
    HONEYBEE = "honeybee"
    MOUSE = "mouse"
    HUMAN = "human"


@dataclass
class BiologicalBaseline:
    """Biological organism baseline metrics."""
    name: str
    neurons: int
    synapses: int
    learning_trials: int  # Typical trials to learn simple task
    energy_watts: float
    bits_per_synapse: float = 4.7  # Estimated
    
    def __repr__(self):
        return f"{self.name}: {self.neurons:,} neurons, {self.synapses:,} synapses"


# Biological baselines
BIOLOGICAL_BASELINES = {
    BiologicalTier.C_ELEGANS: BiologicalBaseline(
        name="C. elegans",
        neurons=302,
        synapses=7000,
        learning_trials=10,
        energy_watts=1e-9
    ),
    BiologicalTier.FRUIT_FLY: BiologicalBaseline(
        name="Fruit Fly",
        neurons=100_000,
        synapses=100_000_000,
        learning_trials=10,
        energy_watts=1e-6
    ),
    BiologicalTier.HONEYBEE: BiologicalBaseline(
        name="Honeybee",
        neurons=1_000_000,
        synapses=1_000_000_000,
        learning_trials=20,
        energy_watts=1e-5
    ),
    BiologicalTier.MOUSE: BiologicalBaseline(
        name="Half-Mouse (Cortex)",
        neurons=10_000_000,  # Reduced from 75M to 10M for practical training
        synapses=10_000_000_000,  # Proportionally reduced
        learning_trials=50,
        energy_watts=0.02  # Proportionally reduced
    ),
    BiologicalTier.HUMAN: BiologicalBaseline(
        name="Human",
        neurons=86_000_000_000,
        synapses=100_000_000_000_000,
        learning_trials=100,
        energy_watts=20.0
    )
}


class BenchmarkResult:
    """Results from a benchmark test."""
    
    def __init__(self, task_name: str, tier: BiologicalTier):
        self.task_name = task_name
        self.tier = tier
        self.biological = BIOLOGICAL_BASELINES[tier]
        
        # Metrics to measure
        self.our_neurons = 0
        self.our_trials = 0
        self.our_performance = 0.0
        self.our_energy = 0.0
        
        # Computed ratios
        self.neuron_efficiency = 0.0
        self.learning_efficiency = 0.0
        self.energy_efficiency = 0.0
        
    def compute_efficiency(self):
        """Calculate efficiency ratios (>1 = we're better, <1 = worse)."""
        self.neuron_efficiency = self.biological.neurons / max(1, self.our_neurons)
        self.learning_efficiency = self.biological.learning_trials / max(1, self.our_trials)
        self.energy_efficiency = self.biological.energy_watts / max(1e-10, self.our_energy)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            'task': self.task_name,
            'tier': self.tier.value,
            'biological': {
                'neurons': self.biological.neurons,
                'trials': self.biological.learning_trials,
                'energy': self.biological.energy_watts
            },
            'our_system': {
                'neurons': self.our_neurons,
                'trials': self.our_trials,
                'performance': self.our_performance,
                'energy': self.our_energy
            },
            'efficiency': {
                'neurons': self.neuron_efficiency,
                'learning': self.learning_efficiency,
                'energy': self.energy_efficiency
            },
            'status': self._get_status()
        }
    
    def _get_status(self) -> str:
        """Overall status vs biological."""
        avg_efficiency = (self.neuron_efficiency + self.learning_efficiency) / 2
        
        if avg_efficiency >= 1.0:
            return "EXCEEDS_BIOLOGICAL"
        elif avg_efficiency >= 0.7:
            return "NEAR_BIOLOGICAL"
        elif avg_efficiency >= 0.3:
            return "BELOW_BIOLOGICAL"
        else:
            return "FAR_BELOW_BIOLOGICAL"


class BiologicalBenchmark:
    """Base class for biological benchmarks."""
    
    def __init__(self, tier: BiologicalTier):
        self.tier = tier
        self.baseline = BIOLOGICAL_BASELINES[tier]
        
    def test(self, brain) -> BenchmarkResult:
        """Run benchmark test. Override in subclasses."""
        raise NotImplementedError
    
    def _measure_neurons_needed(self, task_fn, target_performance: float = 0.8) -> int:
        """Find minimum neurons needed for target performance."""
        neuron_counts = [100, 200, 300, 500, 1000, 2000, 5000]
        
        for n in neuron_counts:
            performance = task_fn(n_neurons=n)
            if performance >= target_performance:
                return n
        
        return neuron_counts[-1]  # Return max if never reached
    
    def _measure_learning_speed(self, brain, task_fn, target_performance: float = 0.8) -> int:
        """Measure episodes/trials to reach target performance."""
        max_episodes = 1000
        
        for episode in range(max_episodes):
            performance = task_fn(brain, episode)
            if performance >= target_performance:
                return episode + 1
        
        return max_episodes


def test_biological_framework():
    """Test the benchmark framework."""
    print("\n" + "="*60)
    print("BIOLOGICAL BENCHMARKING FRAMEWORK")
    print("="*60)
    
    # Display baselines
    print("\nBiological Baselines:")
    print("-" * 60)
    
    for tier, baseline in BIOLOGICAL_BASELINES.items():
        print(f"\n{baseline.name}:")
        print(f"  Neurons: {baseline.neurons:,}")
        print(f"  Synapses: {baseline.synapses:,}")
        print(f"  Learning trials: {baseline.learning_trials}")
        print(f"  Energy: {baseline.energy_watts:.2e} watts")
    
    # Test benchmark result
    print("\n" + "="*60)
    print("EXAMPLE BENCHMARK RESULT")
    print("="*60)
    
    result = BenchmarkResult("Simple Navigation", BiologicalTier.C_ELEGANS)
    
    # Simulate our system (worse than biological)
    result.our_neurons = 1000  # vs 302 biological
    result.our_trials = 100    # vs 10 biological
    result.our_performance = 0.85
    result.our_energy = 1e-6   # vs 1e-9 biological
    
    result.compute_efficiency()
    
    summary = result.get_summary()
    
    print(f"\nTask: {summary['task']}")
    print(f"Tier: {summary['tier']}")
    print(f"\nBiological ({result.biological.name}):")
    print(f"  Neurons: {summary['biological']['neurons']:,}")
    print(f"  Trials: {summary['biological']['trials']}")
    
    print(f"\nOur System:")
    print(f"  Neurons: {summary['our_system']['neurons']:,}")
    print(f"  Trials: {summary['our_system']['trials']}")
    print(f"  Performance: {summary['our_system']['performance']:.1%}")
    
    print(f"\nEfficiency Ratios (>1 = better, <1 = worse):")
    print(f"  Neuron efficiency: {summary['efficiency']['neurons']:.2f}x")
    print(f"  Learning efficiency: {summary['efficiency']['learning']:.2f}x")
    print(f"  Energy efficiency: {summary['efficiency']['energy']:.2f}x")
    
    print(f"\nStatus: {summary['status']}")
    
    # Identify gaps
    print("\n" + "="*60)
    print("GAPS IDENTIFIED")
    print("="*60)
    
    if result.neuron_efficiency < 1.0:
        gap = 1.0 / result.neuron_efficiency
        print(f"\n⚠️  Using {gap:.1f}x more neurons than biological")
        print("   Recommendation: Increase pruning, sparse initialization")
    
    if result.learning_efficiency < 1.0:
        gap = 1.0 / result.learning_efficiency
        print(f"\n⚠️  Learning {gap:.1f}x slower than biological")
        print("   Recommendation: Better credit assignment, meta-learning")
    
    if result.energy_efficiency < 1.0:
        gap = 1.0 / result.energy_efficiency
        print(f"\n⚠️  Using {gap:.0f}x more energy than biological")
        print("   Recommendation: Neuromorphic hardware, event-driven computation")
    
    print("\n✓ Biological benchmarking framework ready!")
    
    return result


if __name__ == "__main__":
    result = test_biological_framework()
