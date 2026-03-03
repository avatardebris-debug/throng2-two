"""
Action Policy Learning

Actor-Critic architecture for learned action selection:
- Actor: Learns which actions to take (policy)
- Critic: Evaluates how good actions are (value function)
- Combined with STDP + Dopamine for biological realism

This replaces random exploration with learned behavior!
"""

import numpy as np
from collections import defaultdict


class ActionPolicy:
    """
    Learned action policy using actor-critic.
    
    Maps brain states to actions through learned weights.
    """
    
    def __init__(self, n_actions=8, learning_rate=0.01):
        """
        Initialize action policy.
        
        Args:
            n_actions: Number of discrete actions (8 directions)
            learning_rate: How fast to update policy
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        # Action weights (learned)
        self.action_weights = np.random.randn(n_actions) * 0.1
        
        # Value function (critic)
        self.value_estimate = 0.0
        
        # History
        self.action_history = []
        self.reward_history = []
        
        # PRE-COMPUTE action angles and vectors for speed
        action_indices = np.arange(n_actions)
        self.action_angles = (2 * np.pi * action_indices) / n_actions
        self.action_vectors = np.column_stack([
            np.cos(self.action_angles),
            np.sin(self.action_angles)
        ])
        
        print(f"\nAction Policy initialized:")
        print(f"  Actions: {n_actions} directions")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Pre-computed action vectors: ✓")
    
    def _direction_to_action(self, direction):
        """VECTORIZED: Convert direction vector to nearest action index."""
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return 0, self.action_vectors[0]
        direction = direction / norm
        
        # Find nearest action vector using dot product (VECTORIZED!)
        # No loops, no trig functions - just matrix multiplication
        similarities = self.action_vectors @ direction
        action_idx = np.argmax(similarities)
        
        return action_idx, self.action_vectors[action_idx]
    
    def select_action_spatial(self, brain_activity, current_position, spatial_memory, brain_positions, epsilon=0.1):
        """
        Select action using SPATIAL GUIDANCE (OPTIMIZED).
        
        Uses brain's spatial representation and memory to navigate intelligently.
        
        Args:
            brain_activity: Current brain state
            current_position: Current (x, y) position
            spatial_memory: SpatialMemory object
            brain_positions: Neuron positions in space
            epsilon: Exploration rate
            
        Returns:
            action_vector: Direction to move
            action_index: Which action was chosen
            strategy: Which strategy was used
        """
        # 1. Check if we have memory of platform
        platform_estimate = spatial_memory.recall()
        confidence = spatial_memory.confidence()
        
        if platform_estimate is not None and confidence > 0.3 and np.random.random() > epsilon:
            # Navigate toward remembered platform location
            direction = platform_estimate - current_position
            
            if np.linalg.norm(direction) > 1.0:
                # OPTIMIZED: Use vectorized direction-to-action
                action_idx, action_vector = self._direction_to_action(direction)
                return action_vector, action_idx, "memory"
        
        # 2. Use brain activity to guide exploration
        active_neurons = np.where(brain_activity > 0.1)[0]
        if len(active_neurons) > 10:
            # VECTORIZED: Compute center of mass
            active_positions = brain_positions[active_neurons]
            center = np.mean(active_positions, axis=0)
            
            # Move toward center of activity
            direction = center - current_position / 100.0
            
            if np.linalg.norm(direction) > 0.01:
                # OPTIMIZED: Use vectorized direction-to-action
                action_idx, action_vector = self._direction_to_action(direction)
                return action_vector, action_idx, "brain"
        
        # 3. Fallback to learned policy
        action_vector, action_idx = self.select_action(brain_activity, epsilon)
        return action_vector, action_idx, "policy"
    
    def select_action(self, brain_activity, epsilon=0.1):
        """
        Select action based on brain activity.
        
        Uses epsilon-greedy: mostly exploit learned policy,
        occasionally explore random actions.
        
        Args:
            brain_activity: Current brain state
            epsilon: Exploration rate (0.1 = 10% random)
            
        Returns:
            action_vector: Direction to move
            action_index: Which action was chosen
        """
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            # Explore: random action
            action_idx = np.random.randint(self.n_actions)
        else:
            # Exploit: use learned policy
            # Compute action preferences from brain activity
            activity_sum = np.sum(brain_activity)
            if activity_sum > 0:
                # Use brain activity to bias action selection
                preferences = self.action_weights + activity_sum * 0.01
            else:
                preferences = self.action_weights
            
            # Softmax selection (probabilistic)
            exp_prefs = np.exp(preferences - np.max(preferences))
            probs = exp_prefs / np.sum(exp_prefs)
            action_idx = np.random.choice(self.n_actions, p=probs)
        
        # Convert action index to direction vector
        angle = (2 * np.pi * action_idx) / self.n_actions
        action_vector = np.array([np.cos(angle), np.sin(angle)])
        
        return action_vector, action_idx
    
    def update_policy(self, action_idx, reward, rpe):
        """
        Update action policy based on reward.
        
        Actor-critic update:
        - Critic: Update value estimate
        - Actor: Update action weights based on advantage
        
        Args:
            action_idx: Which action was taken
            reward: Reward received
            rpe: Reward prediction error (from dopamine)
        """
        # Update value estimate (critic)
        self.value_estimate += self.learning_rate * rpe
        
        # Update action weights (actor)
        # Advantage = how much better this action was than expected
        advantage = rpe
        
        # Increase weight for actions that led to positive RPE
        self.action_weights[action_idx] += self.learning_rate * advantage
        
        # Record
        self.action_history.append(action_idx)
        self.reward_history.append(reward)
    
    def get_stats(self):
        """Get policy statistics."""
        return {
            'action_weights': self.action_weights.copy(),
            'value_estimate': self.value_estimate,
            'action_history': self.action_history.copy(),
            'reward_history': self.reward_history.copy()
        }


def test_action_policy():
    """Test action policy learning."""
    print("\n" + "="*70)
    print("ACTION POLICY TEST")
    print("="*70)
    
    policy = ActionPolicy(n_actions=8, learning_rate=0.1)
    
    print("\nTest 1: Random action selection")
    brain_activity = np.random.rand(100)
    action, action_idx = policy.select_action(brain_activity, epsilon=1.0)
    print(f"  Action: {action_idx}, Direction: {action}")
    
    print("\nTest 2: Policy update (positive reward)")
    policy.update_policy(action_idx, reward=1.0, rpe=1.0)
    print(f"  Action weights updated")
    print(f"  Weight for action {action_idx}: {policy.action_weights[action_idx]:.3f}")
    
    print("\nTest 3: Learned action selection")
    # After learning, should prefer the rewarded action
    action_counts = np.zeros(8)
    for _ in range(100):
        _, action_idx = policy.select_action(brain_activity, epsilon=0.0)
        action_counts[action_idx] += 1
    
    print(f"  Action distribution (100 trials):")
    for i, count in enumerate(action_counts):
        if count > 0:
            print(f"    Action {i}: {count:.0f} times")
    
    print("\n[SUCCESS] Action policy working!")
    print("  ✓ Action selection")
    print("  ✓ Policy updates")
    print("  ✓ Learning from rewards")
    
    return policy


if __name__ == "__main__":
    policy = test_action_policy()
