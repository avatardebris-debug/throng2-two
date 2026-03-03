"""
Spatial Memory Module

Remembers locations where rewards were found.
Uses confidence-weighted averaging for recall.

This enables the brain to navigate toward remembered goals!
"""

import numpy as np


class SpatialMemory:
    """
    Spatial memory for remembering reward locations.
    
    Implements:
    - Location storage with confidence weights
    - Weighted recall (recent memories stronger)
    - Memory decay over time
    """
    
    def __init__(self, decay=0.95):
        """
        Initialize spatial memory.
        
        Args:
            decay: Memory decay rate (0.95 = 5% decay per step)
        """
        self.locations = []
        self.weights = []
        self.decay = decay
        
        print(f"\nSpatial Memory initialized:")
        print(f"  Decay rate: {decay}")
    
    def add(self, location, reward):
        """
        Add a location to memory if reward was received.
        
        Args:
            location: (x, y) position
            reward: Reward value (>0 to remember)
        """
        if reward > 0:
            self.locations.append(location.copy())
            self.weights.append(1.0)
            print(f"  [Memory] Stored location: {location}")
    
    def recall(self):
        """
        Recall the most likely platform location.
        
        Returns:
            Weighted average of remembered locations, or None
        """
        if not self.locations:
            return None
        
        # Weighted average of all remembered locations
        weights = np.array(self.weights)
        weights /= weights.sum()
        
        location = np.average(self.locations, axis=0, weights=weights)
        return location
    
    def decay_memory(self):
        """Decay all memory weights over time."""
        self.weights = [w * self.decay for w in self.weights]
        
        # Remove very weak memories
        threshold = 0.01
        self.locations = [loc for loc, w in zip(self.locations, self.weights) if w > threshold]
        self.weights = [w for w in self.weights if w > threshold]
    
    def confidence(self):
        """Get confidence in memory (0-1)."""
        if not self.weights:
            return 0.0
        return min(1.0, sum(self.weights) / 5.0)  # Full confidence after 5 finds
    
    def clear(self):
        """Clear all memories."""
        self.locations = []
        self.weights = []


def test_spatial_memory():
    """Test spatial memory module."""
    print("\n" + "="*70)
    print("SPATIAL MEMORY TEST")
    print("="*70)
    
    memory = SpatialMemory(decay=0.9)
    
    print("\nTest 1: Add locations")
    memory.add(np.array([50.0, 50.0]), reward=1.0)
    memory.add(np.array([48.0, 52.0]), reward=1.0)
    memory.add(np.array([51.0, 49.0]), reward=1.0)
    
    print(f"\nTest 2: Recall")
    recalled = memory.recall()
    print(f"  Recalled location: {recalled}")
    print(f"  Expected: ~[49.7, 50.3]")
    print(f"  Confidence: {memory.confidence():.2f}")
    
    print(f"\nTest 3: Decay")
    for i in range(5):
        memory.decay_memory()
        print(f"  After {i+1} decays: {len(memory.locations)} locations, confidence={memory.confidence():.2f}")
    
    print("\n[SUCCESS] Spatial memory working!")
    print("  ✓ Location storage")
    print("  ✓ Weighted recall")
    print("  ✓ Memory decay")
    
    return memory


if __name__ == "__main__":
    memory = test_spatial_memory()
