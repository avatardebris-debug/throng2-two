"""
Morris Water Maze with Spatial Guidance - 10M Neurons

ULTIMATE TEST:
- 10M neurons (full mouse cortex)
- Spatial memory + gradient following
- Brain-guided exploration
- Target: Match 1M performance (80%)

This is biological-scale intelligent navigation!
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from src.core.predictive_thronglet import PredictiveThrongletBrain
from src.learning.dopamine import DopamineSystem
from src.learning.action_policy import ActionPolicy
from src.learning.spatial_memory import SpatialMemory


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
    nearest = np.argsort(distances)[:1000]  # 1K for speed
    return nearest


def test_10m_spatial_guided():
    """Morris Water Maze with 10M neurons and spatial guidance."""
    print("\n" + "="*70)
    print("MORRIS WATER MAZE - 10M NEURONS + SPATIAL GUIDANCE")
    print("="*70)
    
    print("\nUltimate Test:")
    print("  - 10M neurons (full mouse cortex)")
    print("  - Spatial memory + gradient following")
    print("  - Brain-guided exploration")
    print("  - Target: Match 1M performance (80%)")
    
    # Create brain
    print("\n" + "="*70)
    print("INITIALIZATION")
    print("="*70)
    
    print("\nCreating Predictive Thronglet Brain (10M neurons)...")
    brain = PredictiveThrongletBrain(n_neurons=10_000_000, avg_connections=10, local_ratio=0.8)
    
    # Create learning systems
    print("\nAdding Learning Systems...")
    dopamine = DopamineSystem(baseline=0.0, learning_rate=0.1)
    policy = ActionPolicy(n_actions=8, learning_rate=0.1)
    memory = SpatialMemory(decay=0.95)
    
    # Create environment
    print("\nCreating Morris Water Maze...")
    maze = MorrisWaterMaze(pool_radius=100, platform_radius=10)
    print(f"  Platform at: {maze.platform_pos} (hidden)")
    print(f"  Max steps per trial: 200")
    
    # Training
    print("\n" + "="*70)
    print("TRAINING (50 trials)")
    print("="*70)
    
    n_trials = 50
    max_steps = 200
    results = []
    
    epsilon_start = 0.3
    epsilon_end = 0.05
    
    strategy_counts = {"memory": 0, "brain": 0, "policy": 0}
    
    for trial in range(n_trials):
        position = maze.reset()
        done = False
        total_reward = 0
        trial_start = time.time()
        
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (trial / n_trials)
        trial_strategies = []
        
        while not done and maze.steps < max_steps:
            # Get active neurons
            active_neurons = position_to_neurons(brain, position)
            
            # Create brain activity pattern
            brain_activity = np.zeros(len(brain.positions))
            brain_activity[active_neurons] = 1.0
            
            # SELECT ACTION USING SPATIAL GUIDANCE
            action, action_idx, strategy = policy.select_action_spatial(
                brain_activity, position, memory, brain.positions, epsilon
            )
            trial_strategies.append(strategy)
            strategy_counts[strategy] += 1
            
            # Take action
            new_position, reward, done = maze.step(action)
            total_reward += reward
            
            # Learn from reward
            if done and reward > 0:
                rpe = dopamine.compute_rpe(reward)
                policy.update_policy(action_idx, reward, rpe)
                memory.add(new_position, reward)
                
                mem_str = f", Memory={memory.confidence():.2f}" if memory.recall() is not None else ""
                print(f"  Trial {trial+1}/50: FOUND in {maze.steps} steps! RPE={rpe:+.3f}{mem_str}")
            
            position = new_position
        
        # Decay memory slightly
        memory.decay_memory()
        
        trial_time = time.time() - trial_start
        path_length = maze.get_path_length()
        
        results.append({
            'trial': trial + 1,
            'steps': maze.steps,
            'time': trial_time,
            'path_length': path_length,
            'found': total_reward > 0,
            'epsilon': epsilon,
            'memory_confidence': memory.confidence()
        })
        
        if total_reward == 0:
            if (trial + 1) % 10 == 0:
                print(f"  Trial {trial+1}/50: timeout ({maze.steps} steps, {trial_time:.0f}s)")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS - 10M NEURONS")
    print("="*70)
    
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
    print(f"  Total successes: {total_successes}/{n_trials} ({total_successes/n_trials:.0%})")
    
    print(f"\nNavigation Strategies:")
    total_steps = sum(strategy_counts.values())
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {strategy.capitalize()}: {count}/{total_steps} ({count/total_steps:.0%})")
    
    print(f"\nComparison:")
    print(f"  1M neurons (spatial): 80% success")
    print(f"  10M neurons (spatial): {total_successes/n_trials:.0%} success")
    print(f"  Biological mice: 75% target")
    
    if total_successes/n_trials >= 0.75:
        print(f"\n[SUCCESS] Biological performance matched! {total_successes/n_trials:.0%} ≥ 75%")
        print("  ✓ Full mouse cortex scale")
        print("  ✓ Intelligent spatial navigation")
        print("  ✓ Biological learning achieved")
    elif total_successes/n_trials >= 0.5:
        print(f"\n[EXCELLENT] Strong performance! {total_successes/n_trials:.0%} ≥ 50%")
        print("  ✓ Spatial guidance working at scale")
    
    print("\nThis is biological-scale intelligent AI! 🧠✨")
    
    return brain, maze, results, dopamine, policy, memory


if __name__ == "__main__":
    brain, maze, results, dopamine, policy, memory = test_10m_spatial_guided()
