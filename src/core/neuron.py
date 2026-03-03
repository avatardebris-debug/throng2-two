"""
Thronglet Neuron - Individual spiking neuron with simple local rules.

Inspired by Conway's Game of Life: simple rules create emergent complexity.
"""

import numpy as np
from typing import List, Tuple


class Thronglet:
    """
    Single spiking neuron with:
    - Leaky integrate-and-fire dynamics
    - Adaptive threshold (homeostatic regulation)
    - Eligibility trace for credit assignment
    - Activity history tracking
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 leak_rate: float = 0.9,
                 homeostatic_rate: float = 0.01):
        """
        Initialize neuron.
        
        Args:
            threshold: Firing threshold (0-1)
            leak_rate: How fast activation decays (0-1, higher = slower decay)
            homeostatic_rate: How fast threshold adapts
        """
        self.threshold = threshold
        self.leak_rate = leak_rate
        self.homeostatic_rate = homeostatic_rate
        
        # State variables
        self.activation = 0.0
        self.last_spike = 0  # Binary: did it spike last step?
        self.eligibility = 0.0  # For credit assignment
        
        # History for homeostatic regulation
        self.activity_history = []
        self.max_history = 100
        
        # Target firing rate (homeostatic set point)
        self.target_activity = 0.1  # Aim for 10% activity
        
    def update(self, input_current: float) -> int:
        """
        Update neuron state for one timestep.
        
        Args:
            input_current: Sum of weighted inputs from other neurons
            
        Returns:
            1 if neuron spiked, 0 otherwise
        """
        # Accumulate input
        self.activation += input_current
        
        # Check if threshold exceeded
        if self.activation >= self.threshold:
            spike = 1
            self.activation = 0.0  # Reset after spike
        else:
            spike = 0
            # Leak (decay toward zero)
            self.activation *= self.leak_rate
            
        # Update eligibility trace (for learning)
        self.eligibility = self.eligibility * 0.9 + spike
        
        # Track activity
        self.activity_history.append(spike)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
            
        # Homeostatic regulation (adjust threshold to maintain target firing rate)
        if len(self.activity_history) >= 20:
            avg_activity = np.mean(self.activity_history[-20:])
            error = avg_activity - self.target_activity
            self.threshold += self.homeostatic_rate * error
            # Keep threshold in reasonable range
            self.threshold = np.clip(self.threshold, 0.1, 1.5)
            
        self.last_spike = spike
        return spike
    
    def get_activity_rate(self) -> float:
        """Get recent average firing rate."""
        if len(self.activity_history) == 0:
            return 0.0
        return np.mean(self.activity_history)
    
    def reset(self):
        """Reset neuron state."""
        self.activation = 0.0
        self.last_spike = 0
        self.eligibility = 0.0


class NeuronPopulation:
    """
    Collection of Thronglet neurons for batch processing.
    """
    
    def __init__(self, n_neurons: int, **neuron_params):
        """
        Create population of neurons.
        
        Args:
            n_neurons: Number of neurons
            **neuron_params: Parameters passed to each Thronglet
        """
        self.neurons = [Thronglet(**neuron_params) for _ in range(n_neurons)]
        self.n_neurons = n_neurons
        
    def update(self, inputs: np.ndarray) -> np.ndarray:
        """
        Update all neurons in parallel.
        
        Args:
            inputs: Array of input currents (one per neuron)
            
        Returns:
            Array of spike outputs (1s and 0s)
        """
        spikes = np.zeros(self.n_neurons)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.update(inputs[i])
        return spikes
    
    def get_eligibility_traces(self) -> np.ndarray:
        """Get eligibility traces from all neurons (for learning)."""
        return np.array([n.eligibility for n in self.neurons])
    
    def get_activity_rates(self) -> np.ndarray:
        """Get firing rates from all neurons."""
        return np.array([n.get_activity_rate() for n in self.neurons])
    
    def reset(self):
        """Reset all neurons."""
        for neuron in self.neurons:
            neuron.reset()
