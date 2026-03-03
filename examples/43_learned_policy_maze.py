"""
Morris Water Maze with Learned Action Policy

Complete biological learning system:
- STDP: Temporal credit assignment
- Dopamine: Reward signaling
- Action Policy: Learned behavior (NOT random!)

This is the full system - brain learns WHAT to do!
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from src.core.predictive_thronglet import PredictiveThrongletBrain
from src.learning.stdp import STDPLearning
from src.learning.dopamine import DopamineSystem, ReinforcementLearning
from src.learning.action_policy import ActionPolicy


class MorrisWaterMaze:
    """Morris Water Maze environment."""
    
    def __init__(self, pool_radius=100, platform_radius=10):
        self.pool_radius = pool_radius
        self.platform_radius = platform_radius
        self.platform_pos = np.array([50.0, 50.0])
        
    def reset(self):
        angle = np.random.uniform(0, 2*np.pi)
        self.position = np.array([
            self.pool_radius * np.cos(angle),
            self.pool_radius * np.sin(angle)
        ])
        self.path = [self.position.copy()]
        self.steps = 0
        return self.position
    
    def step(self, action):
        self.position += action * 5.0
        dist_from_center = np.linalg.norm(self.position)
        if dist_from_center > self.pool_radius:
            self.position = self.position / dist_from_center * self.pool_radius
        
        self.path.append(self.position.copy())
        self.steps += 1
        
        dist_to_platform = np.linalg.norm(self.position - self.platform_pos)
        
        if dist_to_platform < self.platform_radius:
            return self.position, 1.0, True
        else:
            return self.position, 0.0, False
    
    def get_path_length(self):
        if len(self.path) < 2:
            return 0
        total = 0
        for i in range(len(self.path) - 1):
            total += np.linalg.norm(self.path[i+1] - self.path[i])
        return total


def position_to_neurons(brain, position, pool_radius=100):
    """Convert position to active neurons."""
    norm_pos = position / pool_radius
    distances = np.sqrt((brain.positions[:, 0] - norm_pos[0])**2 + 
                       (brain.positions[:, 1] - norm_pos[1])**2)
    nearest = np.argsort(distances)[:1000]
    return nearest


def test_learned_policy_maze():
    """Morris Water Maze with LEARNED action policy."""
    print("\n" + "="*70)
    print("MORRIS WATER MAZE - LEARNED ACTION POLICY")
    print("="*70)
    
    print("\nComplete Biological Learning System:")
    print("  - STDP: Temporal credit assignment")
    print("  - Dopamine: Reward signaling")
    print("  - Action Policy: LEARNED behavior (not random!)")
    
    print("\nBiological Benchmark (Real Mice):")
    print("  - Trials: 50 trials")
    print("  - Success rate: 10% → 75%")
    
    # Create brain
    print("\n" + "="*70)
    print("INITIALIZATION")
    print("="*70)
    
    print("\nCreating Predictive Thronglet Brain (1M neurons)...")
    brain = PredictiveThrongletBrain(n_neurons=1_000_000, avg_connections=10, local_ratio=0.8)
    
    # Create learning systems
    print("\nAdding STDP + Dopamine...")
    stdp = STDPLearning(tau_plus=0.020, tau_minus=0.020, A_plus=0.01, A_minus=0.01)
    dopamine = DopamineSystem(baseline=0.0, learning_rate=0.1)
    rl = ReinforcementLearning(stdp, dopamine)
    
    # Create action policy
    print("\nAdding Action Policy...")
    policy = ActionPolicy(n_actions=8, learning_rate=0.1)
    
    # Create environment
    print("\nCreating Morris Water Maze...")
    maze = MorrisWaterMaze(pool_radius=100, platform_radius=10)
    print(f"  Platform at: {maze.platform_pos} (hidden)")
    print(f"  Max steps per trial: 200")
    
    # Training
    print("\n" + "="*70)
    print("TRAINING (50 trials with LEARNED policy)")
    print("="*70)
    
    n_trials = 50
    max_steps = 200
    results = []
    
    # Exploration schedule: start high, decrease over time
    epsilon_start = 0.5  # 50% exploration initially
    epsilon_end = 0.1    # 10% exploration at end
    
    for trial in range(n_trials):
        position = maze.reset()
        done = False
        total_reward = 0
        trial_start = time.time()
        
        # Decay exploration over trials
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (trial / n_trials)
        
        while not done and maze.steps < max_steps:
            # Get active neurons
            active_neurons = position_to_neurons(brain, position)
            
            # Create brain activity pattern
            brain_activity = np.zeros(len(brain.positions))
            brain_activity[active_neurons] = 1.0
            
            # Update STDP eligibility
            current_time = time.time()
            stdp.update_eligibility(active_neurons, current_time)
            
            # SELECT ACTION USING LEARNED POLICY (not random!)
            action, action_idx = policy.select_action(brain_activity, epsilon=epsilon)
            
            # Take action
            new_position, reward, done = maze.step(action)
            total_reward += reward
            
            # Learn with STDP + Dopamine
            if done and reward > 0:
                weight_updates, rpe = rl.update_weights(brain.weights, reward)
                
                # UPDATE ACTION POLICY (learn which action was good!)
                policy.update_policy(action_idx, reward, rpe)
                
                print(f"  Trial {trial+1}/50: FOUND in {maze.steps} steps! "
                      f"RPE={rpe:+.3f}, Epsilon={epsilon:.2f}")
            
            position = new_position
        
        trial_time = time.time() - trial_start
        path_length = maze.get_path_length()
        
        results.append({
            'trial': trial + 1,
            'steps': maze.steps,
            'time': trial_time,
            'path_length': path_length,
            'found': total_reward > 0,
            'epsilon': epsilon,
        })
        
        if total_reward == 0:
            if (trial + 1) % 10 == 0:
                print(f"  Trial {trial+1}/50: timeout ({maze.steps} steps, epsilon={epsilon:.2f})")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Calculate success rate over time
    window_size = 10
    success_rates = []
    for i in range(0, len(results), window_size):
        window = results[i:i+window_size]
        success_rate = sum([r['found'] for r in window]) / len(window)
        success_rates.append(success_rate)
    
    early_success = success_rates[0] if success_rates else 0
    late_success = success_rates[-1] if success_rates else 0
    
    total_successes = sum([r['found'] for r in results])
    
    print(f"\nLearning Progress:")
    print(f"  Early success (trials 1-10): {early_success:.0%}")
    print(f"  Late success (trials 41-50): {late_success:.0%}")
    print(f"  Improvement: +{(late_success - early_success)*100:.0f}%")
    print(f"  Total successes: {total_successes}/{n_trials}")
    
    print(f"\nAction Policy:")
    print(f"  Exploration: {epsilon_start:.0%} → {epsilon_end:.0%}")
    print(f"  Final action weights:")
    for i, weight in enumerate(policy.action_weights):
        if abs(weight) > 0.1:
            angle = (360 * i) / 8
            print(f"    Action {i} ({angle:.0f}°): {weight:+.3f}")
    
    print(f"\nBiological Comparison:")
    print(f"  Real mice: 10% → 75% success")
    print(f"  Our AI (random): 0% → 10% success")
    print(f"  Our AI (learned): {early_success:.0%} → {late_success:.0%}")
    
    if late_success > early_success + 0.2:  # 20% improvement
        print(f"\n[SUCCESS] Strong learning detected!")
        print("  ✓ Action policy learning working")
        print("  ✓ STDP + Dopamine + Policy = Complete system")
    elif late_success > early_success:
        print(f"\n[PROGRESS] Learning detected!")
        print("  ✓ Action policy improving")
        print("  ✓ System learning from experience")
    
    print("\nThis is biological reinforcement learning with learned actions! 🐝🧠✨")
    
    return brain, maze, results, rl, policy


if __name__ == "__main__":
    brain, maze, results, rl, policy = test_learned_policy_maze()
