"""
Thronglet Network - Statistical world model through geometric neural network.

Stores correlations and associations, not discrete facts.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.sparse import csr_matrix, issparse
from .neuron import NeuronPopulation
from .geometry import place_neurons, small_world_connections, small_world_connections_sparse


class ThrongletNetwork:
    """
    Network of thronglet neurons forming a statistical world model.
    
    Key features:
    - Geometric neuron placement
    - Sparse small-world connectivity
    - Association-based memory (not fact storage)
    - Continuous adaptation
    """
    
    def __init__(self,
                 n_neurons: int = 500,
                 dimension: int = 2,
                 connection_prob: float = 0.02,
                 use_fibonacci: bool = True,
                 **neuron_params):
        """
        Initialize network.
        
        Args:
            n_neurons: Number of neurons
            dimension: 2D or 3D placement
            connection_prob: Base connection probability
            use_fibonacci: Use Fibonacci placement (vs random)
            **neuron_params: Parameters for neurons
        """
        self.n_neurons = n_neurons
        self.dimension = dimension
        
        # Create neurons
        self.neurons = NeuronPopulation(n_neurons, **neuron_params)
        
        # Place neurons in space
        if use_fibonacci:
            self.positions = place_neurons(n_neurons, dimension)
        else:
            # Random placement
            if dimension == 2:
                self.positions = np.random.randn(n_neurons, 2)
            else:
                self.positions = np.random.randn(n_neurons, 3)
                # Normalize to sphere
                norms = np.linalg.norm(self.positions, axis=1, keepdims=True)
                self.positions = self.positions / norms
                
        # Create connections - use sparse for large networks
        if n_neurons > 10000:
            print(f"Using sparse matrix for {n_neurons:,} neurons...")
            self.weights = small_world_connections_sparse(
                self.positions,
                connection_prob=connection_prob
            )
            self.use_sparse = True
        else:
            self.weights = small_world_connections(
                self.positions,
                connection_prob=connection_prob
            )
            self.use_sparse = False
        
        # Track state
        self.current_spikes = np.zeros(n_neurons)
        self.step_count = 0
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Process inputs through network.
        
        Args:
            inputs: External inputs (can be shorter than n_neurons)
            
        Returns:
            Spike outputs from all neurons
        """
        # Compute input currents to each neuron
        currents = np.zeros(self.n_neurons)
        
        # Add external inputs to first neurons
        input_size = min(len(inputs), self.n_neurons)
        currents[:input_size] = inputs[:input_size]
        
        # Add recurrent inputs (from other neurons' spikes)
        if self.use_sparse:
            recurrent = self.weights @ self.current_spikes
        else:
            recurrent = self.weights @ self.current_spikes
        currents += recurrent
        
        # Update neurons
        self.current_spikes = self.neurons.update(currents)
        
        self.step_count += 1
        
        return self.current_spikes
    
    def hebbian_update(self, 
                       learning_rate: float = 0.01,
                       modulation: float = 1.0,
                       sparse_update: bool = None,
                       n_updates: int = 1000):
        """
        Hebbian learning: "Fire together, wire together"
        
        VECTORIZED version - avoids O(n²) nested loops!
        
        Modulated by dopamine-like signal (reward).
        
        Args:
            learning_rate: How fast to learn
            modulation: Neuromodulator level (0-1, typically from reward)
            sparse_update: If True, only update random subset. Auto-detected if None.
            n_updates: Number of connections to update in sparse mode
        """
        # Auto-detect sparse update mode for large networks
        if sparse_update is None:
            sparse_update = self.n_neurons > 50000
        
        # Get recent spike activity
        eligibility = self.neurons.get_eligibility_traces()
        
        # Find active neurons (eligibility > threshold)
        active_mask = eligibility > 0.1
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) < 2:
            return  # Need at least 2 active neurons
        
        if sparse_update:
            # SPARSE UPDATE MODE: Only update random subset of connections
            # This is much faster for large networks
            n_active = len(active_indices)
            n_pairs = min(n_updates, n_active * (n_active - 1) // 2)
            
            for _ in range(n_pairs):
                # Random pair of active neurons
                i, j = np.random.choice(active_indices, size=2, replace=False)
                
                # Hebbian update
                delta = (learning_rate * 
                        modulation * 
                        eligibility[i] * 
                        eligibility[j])
                
                if self.use_sparse:
                    self.weights[i, j] = self.weights[i, j] + delta
                    self.weights[j, i] = self.weights[j, i] + delta
                else:
                    self.weights[i, j] += delta
                    self.weights[j, i] += delta
        else:
            # VECTORIZED UPDATE MODE: Update all active pairs efficiently
            # Use outer product to compute all pairwise updates at once
            
            # Extract eligibility for active neurons
            active_elig = eligibility[active_indices]
            
            # Compute outer product: active_elig[:, None] @ active_elig[None, :]
            # This gives all pairwise products in one operation!
            delta_matrix = np.outer(active_elig, active_elig)
            delta_matrix *= learning_rate * modulation
            
            # Zero out diagonal (no self-connections)
            np.fill_diagonal(delta_matrix, 0)
            
            # Update weights for active neurons only
            if self.use_sparse:
                # For sparse matrices, convert to LIL for efficient updates
                if not isinstance(self.weights, csr_matrix):
                    self.weights = csr_matrix(self.weights)
                
                weights_lil = self.weights.tolil()
                for local_i, global_i in enumerate(active_indices):
                    for local_j, global_j in enumerate(active_indices):
                        if global_i != global_j:
                            weights_lil[global_i, global_j] += delta_matrix[local_i, local_j]
                self.weights = weights_lil.tocsr()
            else:
                # Dense matrix - use advanced indexing
                idx = np.ix_(active_indices, active_indices)
                self.weights[idx] += delta_matrix
        
        # Clip weights to [0, 1]
        if self.use_sparse:
            self.weights.data = np.clip(self.weights.data, 0, 1)
            # Weight decay
            self.weights.data *= 0.9995
        else:
            self.weights = np.clip(self.weights, 0, 1)
            # Weight decay (prevents runaway growth)
            self.weights *= 0.9995
        
    def prune_weak_connections(self, threshold: float = 0.05):
        """
        Remove weak connections (use it or lose it).
        
        Args:
            threshold: Connections below this are pruned
        """
        if self.use_sparse:
            self.weights.data[self.weights.data < threshold] = 0
            self.weights.eliminate_zeros()
        else:
            self.weights[self.weights < threshold] = 0
        
    def get_statistics(self) -> dict:
        """Get network statistics."""
        from .geometry import connection_statistics
        stats = connection_statistics(self.weights)
        stats['active_neurons'] = np.sum(self.neurons.get_activity_rates() > 0.05)
        return stats
    
    def reset(self):
        """Reset network state (but keep connections)."""
        self.neurons.reset()
        self.current_spikes = np.zeros(self.n_neurons)


class LayeredNetwork:
    """
    Layered version with distinct input, hidden, and output layers.
    
    Better for structured tasks (navigation, control, etc.)
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list,
                 output_size: int,
                 **network_params):
        """
        Create layered network.
        
        Args:
            input_size: Number of input neurons
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons
            **network_params: Parameters for each subnet
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Create layer networks
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = []
        
        for size in layer_sizes:
            layer = ThrongletNetwork(n_neurons=size, **network_params)
            self.layers.append(layer)
            
        # Inter-layer connections
        self.layer_weights = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.uniform(0, 0.3, (layer_sizes[i+1], layer_sizes[i]))
            self.layer_weights.append(w)
            
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through layers."""
        activation = inputs
        
        # Process through layers
        for i, layer in enumerate(self.layers[:-1]):
            # Process in current layer
            spikes = layer.forward(activation)
            
            # Project to next layer
            activation = self.layer_weights[i] @ spikes
            
        # Output layer
        output_spikes = self.layers[-1].forward(activation)
        
        return output_spikes
    
    def learn(self, learning_rate: float = 0.01, modulation: float = 1.0):
        """Apply learning to all layers."""
        for layer in self.layers:
            layer.hebbian_update(learning_rate, modulation)
            
    def reset(self):
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()
