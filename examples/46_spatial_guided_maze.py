"""
Morris Water Maze with Spatial Guidance - 1M Neurons

INTELLIGENT NAVIGATION:
- Spatial memory (remember platform)
- Gradient following (move toward memory)
- Brain-guided exploration (use activity)

Target: 40-60% success rate!
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
    nearest = np.argsort(distances)[:1000]
    return nearest


def test_spatial_guided_maze():
    """Morris Water Maze with SPATIAL GUIDANCE."""
    print("\n" + "="*70)
    print("MORRIS WATER MAZE - SPATIAL GUIDANCE")
    print("="*70)
    
    print("\nIntelligent Navigation System:")
    print("  - Spatial memory (remember platform)")
    print("  - Gradient following (move toward memory)")
    print("  - Brain-guided exploration (use activity)")
    
    print("\nTarget: 40-60% success rate (vs 18-22% random)")
    
    # Create brain
    print("\n" + "="*70)
    print("INITIALIZATION")
    print("="*70)
    
    print("\nCreating Predictive Thronglet Brain (1M neurons)...")
    brain = PredictiveThrongletBrain(n_neurons=1_000_000, avg_connections=10, local_ratio=0.8)
    
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
    print("TRAINING (50 trials with SPATIAL GUIDANCE)")
    print("="*70)
    
    n_trials = 50
    max_steps = 200
    results = []
    
    epsilon_start = 0.3  # Less random exploration
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
                strat_dist = {k: trial_strategies.count(k) for k in set(trial_strategies)}
                print(f"  Trial {trial+1}/50: FOUND in {maze.steps} steps! "
                      f"RPE={rpe:+.3f}{mem_str}, Strategies={strat_dist}")
            
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
                print(f"  Trial {trial+1}/50: timeout ({maze.steps} steps)")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS - SPATIAL GUIDANCE")
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
    print(f"  Random (previous): 18-22% success")
    print(f"  Spatial guidance: {total_successes/n_trials:.0%} success")
    
    if total_successes/n_trials >= 0.4:
        print(f"\n[SUCCESS] Target achieved! {total_successes/n_trials:.0%} ≥ 40%")
        print("  ✓ Spatial memory working")
        print("  ✓ Intelligent navigation")
        print("  ✓ Ready for 10M scale test")
    elif total_successes/n_trials > 0.25:
        print(f"\n[PROGRESS] Improvement detected! {total_successes/n_trials:.0%} > 25%")
        print("  ✓ Better than random")
        print("  ✓ Spatial guidance helping")
    else:
        print(f"\n[RESULT] {total_successes/n_trials:.0%} success rate")
    
    print("\nThis is intelligent spatial navigation! 🧠🗺️")
    
    return brain, maze, results, dopamine, policy, memory


if __name__ == "__main__":
    brain, maze, results, dopamine, policy, memory = test_spatial_guided_maze()
