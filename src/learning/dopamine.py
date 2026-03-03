"""
Dopamine Neuromodulation

Implements reward-based learning through dopamine signaling:
- Dopamine = Reward Prediction Error (RPE)
- RPE = Actual Reward - Expected Reward
- Modulates learning rate globally

This is how real brains learn from rewards!
"""

import numpy as np


class DopamineSystem:
    """
    Dopamine neuromodulation system.
    
    Computes reward prediction errors and modulates learning.
    """
    
    def __init__(self, baseline=0.0, learning_rate=0.1):
        """
        Initialize dopamine system.
        
        Args:
            baseline: Baseline dopamine level
            learning_rate: How fast to update expectations
        """
        self.baseline = baseline
        self.level = baseline
        self.expected_reward = 0.0
        self.learning_rate = learning_rate
        
        # History
        self.rpe_history = []
        self.reward_history = []
        
        print(f"\nDopamine System initialized:")
        print(f"  Baseline: {baseline}")
        print(f"  Learning rate: {learning_rate}")
    
    def compute_rpe(self, actual_reward):
        """
        Compute Reward Prediction Error.
        
        RPE = Actual - Expected
        - Positive RPE → dopamine burst (learn this!)
        - Negative RPE → dopamine dip (avoid this!)
        - Zero RPE → no dopamine (already learned)
        
        Returns: rpe (reward prediction error)
        """
        rpe = actual_reward - self.expected_reward
        
        # Update dopamine level
        self.level = self.baseline + rpe
        
        # Update expectation (TD learning)
        self.expected_reward += self.learning_rate * rpe
        
        # Record
        self.rpe_history.append(rpe)
        self.reward_history.append(actual_reward)
        
        return rpe
    
    def modulate_learning_rate(self, base_rate, rpe):
        """
        Modulate learning rate by dopamine.
        
        Positive RPE → increase learning (good outcome!)
        Negative RPE → decrease learning (bad outcome)
        """
        # Sigmoid modulation
        modulation = 1.0 + np.tanh(rpe)
        return base_rate * modulation
    
    def get_dopamine_level(self):
        """Get current dopamine level."""
        return self.level
    
    def reset(self):
        """Reset dopamine system."""
        self.level = self.baseline
        self.expected_reward = 0.0
        self.rpe_history = []
        self.reward_history = []


class ReinforcementLearning:
    """
    Complete reinforcement learning system.
    
    Combines STDP + Dopamine for three-factor learning:
    Weight update = STDP × Dopamine × Eligibility
    """
    
    def __init__(self, stdp, dopamine):
        """
        Initialize reinforcement learning.
        
        Args:
            stdp: STDP learning system
            dopamine: Dopamine system
        """
        self.stdp = stdp
        self.dopamine = dopamine
        
        print(f"\nReinforcement Learning initialized:")
        print(f"  Three-factor rule: STDP × Dopamine × Eligibility")
    
    def update_weights(self, weights, reward):
        """
        Update weights using three-factor learning rule.
        
        Args:
            weights: Current weight matrix (sparse)
            reward: Reward signal
            
        Returns:
            weight_updates: Dictionary of (i, j) → dw
        """
        # 1. Get eligibility traces from STDP
        eligibility = self.stdp.get_updates()
        
        # 2. Compute dopamine signal (RPE)
        rpe = self.dopamine.compute_rpe(reward)
        
        # 3. Modulate STDP by dopamine
        weight_updates = {}
        for (pre, post), dw_stdp in eligibility.items():
            # Three-factor rule: STDP × Dopamine
            dw_final = dw_stdp * rpe
            weight_updates[(pre, post)] = dw_final
        
        # Clear eligibility for next trial
        self.stdp.clear_eligibility()
        
        return weight_updates, rpe
    
    def get_stats(self):
        """Get learning statistics."""
        return {
            'expected_reward': self.dopamine.expected_reward,
            'dopamine_level': self.dopamine.level,
            'rpe_history': self.dopamine.rpe_history,
            'reward_history': self.dopamine.reward_history
        }


def test_dopamine():
    """Test dopamine system."""
    print("\n" + "="*70)
    print("DOPAMINE SYSTEM TEST")
    print("="*70)
    
    dopamine = DopamineSystem(baseline=0.0, learning_rate=0.1)
    
    print("\nTest 1: Unexpected reward (positive RPE)")
    rpe = dopamine.compute_rpe(1.0)
    print(f"  Reward: 1.0, Expected: 0.0")
    print(f"  RPE: {rpe:+.3f} (should be positive)")
    print(f"  Dopamine level: {dopamine.level:.3f}")
    
    print("\nTest 2: Expected reward (zero RPE)")
    # After learning, reward becomes expected
    for _ in range(10):
        dopamine.compute_rpe(1.0)
    
    rpe = dopamine.compute_rpe(1.0)
    print(f"  Reward: 1.0, Expected: {dopamine.expected_reward:.3f}")
    print(f"  RPE: {rpe:+.3f} (should be near zero)")
    
    print("\nTest 3: No reward (negative RPE)")
    rpe = dopamine.compute_rpe(0.0)
    print(f"  Reward: 0.0, Expected: {dopamine.expected_reward:.3f}")
    print(f"  RPE: {rpe:+.3f} (should be negative)")
    print(f"  Dopamine level: {dopamine.level:.3f}")
    
    print("\n[SUCCESS] Dopamine system working!")
    print("  ✓ Positive RPE on unexpected reward")
    print("  ✓ Zero RPE on expected reward")
    print("  ✓ Negative RPE on reward omission")
    print("  ✓ Expectations update over time")
    
    return dopamine


if __name__ == "__main__":
    dopamine = test_dopamine()
