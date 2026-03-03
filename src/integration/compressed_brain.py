"""
Phase 2 Integration Layer

Connects Phase 1 (Basic Brain) with Phase 2 (Compression) into unified system.
Enables training brains WITH compression from the start.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Optional, Dict, Any
from src.compression.fourier_compression import FourierNeuronGroup
from src.compression.statistical_compression import GaussianCompressor
from src.compression.hybrid_compression import HybridCompressor
from src.compression.statistical_sampling import StatisticalSampler


class CompressedBrain:
    """
    Unified brain interface with integrated compression.
    
    Combines Phase 1 functionality with Phase 2 compression methods.
    """
    
    def __init__(self,
                 n_neurons: int,
                 compression_method: str = 'none',
                 compression_ratio: float = 10.0,
                 **kwargs):
        """
        Initialize compressed brain.
        
        Args:
            n_neurons: Number of neurons
            compression_method: 'none', 'fourier', 'statistical', 'hybrid', 'sampling'
            compression_ratio: Target compression ratio (e.g., 10 = 10x compression)
            **kwargs: Additional compression-specific parameters
        """
        self.n_neurons = n_neurons
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        
        # Initialize weights (uncompressed initially)
        self.weights = np.random.randn(n_neurons, n_neurons) * 0.1
        
        # Initialize compression
        self.compressor = self._create_compressor(compression_method, **kwargs)
        
        # Training state
        self.episode_count = 0
        self.compression_active = False
        
    def _create_compressor(self, method: str, **kwargs):
        """Create appropriate compressor."""
        if method == 'none':
            return None
        
        elif method == 'fourier':
            return FourierNeuronGroup(
                n_neurons=self.n_neurons,
                compression_ratio=self.compression_ratio
            )
        
        elif method == 'statistical':
            return GaussianCompressor(
                target_compression=self.compression_ratio
            )
        
        elif method == 'hybrid':
            return HybridCompressor(
                n_neurons=self.n_neurons,
                target_compression=self.compression_ratio
            )
        
        elif method == 'sampling':
            sample_fraction = 1.0 / self.compression_ratio
            return StatisticalSampler(
                sample_fraction=sample_fraction,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through brain.
        
        Automatically uses compressed or uncompressed weights.
        """
        # Get current weights (compressed or uncompressed)
        W = self.get_weights()
        
        # Simple forward pass
        activations = np.tanh(W @ inputs)
        
        return activations
    
    def update_weights(self, delta_W: np.ndarray):
        """
        Update weights with learning.
        
        Args:
            delta_W: Weight update (from Hebbian, backprop, etc.)
        """
        if self.compression_active:
            # Decompress, update, recompress
            self.decompress()
            self.weights += delta_W
            self.compress()
        else:
            # Direct update
            self.weights += delta_W
    
    def compress(self):
        """Compress current weights."""
        if self.compressor is None:
            return
        
        if self.compression_method == 'fourier':
            self.compressor.compress(self.weights)
        
        elif self.compression_method in ['statistical', 'hybrid']:
            self.compressor.compress(self.weights)
        
        elif self.compression_method == 'sampling':
            # Sample weights
            importance = np.abs(self.weights)
            self.compressor.sample_weights(self.weights, importance)
        
        self.compression_active = True
    
    def decompress(self):
        """Decompress weights back to full representation."""
        if self.compressor is None or not self.compression_active:
            return
        
        if self.compression_method == 'fourier':
            self.weights = self.compressor.reconstruct()
        
        elif self.compression_method in ['statistical', 'hybrid']:
            self.weights = self.compressor.decompress()
        
        elif self.compression_method == 'sampling':
            self.weights = self.compressor.reconstruct()
        
        self.compression_active = False
    
    def get_weights(self) -> np.ndarray:
        """Get current weights (decompressing if needed)."""
        if self.compression_active:
            return self.decompress_readonly()
        return self.weights
    
    def decompress_readonly(self) -> np.ndarray:
        """Decompress without changing state (for inference)."""
        if not self.compression_active:
            return self.weights
        
        if self.compression_method == 'fourier':
            return self.compressor.reconstruct()
        elif self.compression_method in ['statistical', 'hybrid']:
            return self.compressor.decompress()
        elif self.compression_method == 'sampling':
            return self.compressor.reconstruct()
        
        return self.weights
    
    def train_episode(self, environment, learning_rate: float = 0.01):
        """
        Train for one episode.
        
        Simplified training loop demonstrating compression integration.
        """
        # Initialize
        state = environment.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Forward pass
            action_values = self.forward(state)
            action = np.argmax(action_values)
            
            # Environment step
            next_state, reward, done, _ = environment.step(action)
            total_reward += reward
            
            # Simple Hebbian update
            delta_W = learning_rate * np.outer(action_values, state)
            self.update_weights(delta_W)
            
            state = next_state
        
        self.episode_count += 1
        
        return total_reward
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        uncompressed_size = self.weights.nbytes
        
        if self.compression_active and self.compressor is not None:
            compressed_size = self.compressor.get_compressed_size()
        else:
            compressed_size = uncompressed_size
        
        return {
            'uncompressed_mb': uncompressed_size / 1024**2,
            'compressed_mb': compressed_size / 1024**2,
            'compression_ratio': uncompressed_size / compressed_size,
            'savings_percent': (1 - compressed_size / uncompressed_size) * 100
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get brain statistics."""
        mem = self.get_memory_usage()
        
        return {
            'n_neurons': self.n_neurons,
            'compression_method': self.compression_method,
            'compression_active': self.compression_active,
            'episode_count': self.episode_count,
            **mem
        }


def demonstrate_compression_integration():
    """Demonstrate integrated compression."""
    print("\n" + "="*60)
    print("PHASE 2 INTEGRATION: Compressed Brain Training")
    print("="*60)
    
    # Simple test environment
    class DummyEnvironment:
        def __init__(self):
            self.state_dim = 10
            
        def reset(self):
            return np.random.randn(self.state_dim)
        
        def step(self, action):
            next_state = np.random.randn(self.state_dim)
            reward = np.random.rand()
            done = np.random.rand() > 0.9  # 10% chance to end
            return next_state, reward, done, {}
    
    env = DummyEnvironment()
    
    # Test different compression methods
    methods = ['none', 'fourier', 'statistical', 'sampling']
    
    results = {}
    
    for method in methods:
        print(f"\n{method.upper()} Compression:")
        print("-" * 40)
        
        # Create brain
        brain = CompressedBrain(
            n_neurons=100,
            compression_method=method,
            compression_ratio=10.0
        )
        
        # Train for a few episodes
        for ep in range(5):
            reward = brain.train_episode(env)
            
            # Compress after episode 2
            if ep == 2 and method != 'none':
                brain.compress()
                print(f"  Compressed after episode {ep}")
        
        # Get stats
        stats = brain.get_stats()
        results[method] = stats
        
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Memory: {stats['uncompressed_mb']:.2f} MB → {stats['compressed_mb']:.2f} MB")
        print(f"  Compression: {stats['compression_ratio']:.1f}x ({stats['savings_percent']:.1f}% savings)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Compression Comparison")
    print("="*60)
    
    print(f"\n{'Method':<15} {'Memory (MB)':<15} {'Ratio':<10} {'Savings':<10}")
    print("-" * 60)
    
    for method, stats in results.items():
        print(f"{method:<15} {stats['compressed_mb']:<15.2f} {stats['compression_ratio']:<10.1f}x {stats['savings_percent']:<10.1f}%")
    
    print("\n✓ Phase 2 compression integrated with Phase 1 training!")
    
    return results


if __name__ == "__main__":
    results = demonstrate_compression_integration()
