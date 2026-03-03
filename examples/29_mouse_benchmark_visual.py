"""
Mouse Benchmark Test with Visualization - 10M Neurons

Real-time visualization of learning progress on mouse tasks.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MouseScaleBrain:
    """10M neuron brain optimized for mouse benchmarks."""
    
    def __init__(self, n_neurons=10_000_000, n_connections=2_000_000):
        from scipy.sparse import coo_matrix
        
        self.n_neurons = n_neurons
        print(f"\nInitializing {n_neurons:,} neuron brain...")
        
        start = time.time()
        
        # Build in chunks
        chunk_size = 1_000_000
        chunks = []
        
        for i in range(0, n_connections, chunk_size):
            end = min(i + chunk_size, n_connections)
            n_conn = end - i
            
            rows = np.random.randint(0, n_neurons, n_conn, dtype=np.int32)
            cols = np.random.randint(0, n_neurons, n_conn, dtype=np.int32)
            data = np.random.uniform(0, 0.3, n_conn).astype(np.float32)
            
            chunk = coo_matrix((data, (rows, cols)), shape=(n_neurons, n_neurons), dtype=np.float32)
            chunks.append(chunk)
        
        if len(chunks) == 1:
            weights = chunks[0]
        else:
            weights = chunks[0]
            for chunk in chunks[1:]:
                weights = weights + chunk
        
        weights = (weights + weights.T) / 2
        self.weights = weights.tocsr()
        
        print(f"  Created in {time.time() - start:.2f}s")
        
        self.current_spikes = np.zeros(n_neurons, dtype=np.float32)
        self.eligibility = np.zeros(n_neurons, dtype=np.float32)
        self.performance_history = []
    
    def forward(self, inputs):
        input_current = np.zeros(self.n_neurons, dtype=np.float32)
        input_current[:len(inputs)] = inputs[:len(inputs)]
        
        recurrent = self.weights @ self.current_spikes
        total_current = input_current + recurrent
        self.current_spikes = np.tanh(total_current)
        
        self.eligibility = self.eligibility * 0.9 + np.abs(self.current_spikes) * 0.1
        
        return self.current_spikes
    
    def train_step(self, state, reward):
        outputs = self.forward(state)
        
        active = np.where(self.eligibility > 0.1)[0]
        
        if len(active) >= 2:
            n_updates = min(500, len(active) * (len(active) - 1) // 2)
            
            for _ in range(n_updates):
                i, j = np.random.choice(active, size=2, replace=False)
                delta = 0.01 * reward * self.eligibility[i] * self.eligibility[j]
                self.weights[i, j] = self.weights[i, j] + delta
                self.weights[j, i] = self.weights[j, i] + delta
            
            self.weights.data = np.clip(self.weights.data, 0, 1)
            self.weights.data *= 0.9995
        
        return outputs


def test_with_visualization(brain, task_name, test_fn, max_trials=200):
    """Run test with live visualization."""
    
    # Setup plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{task_name} - 10M Neuron Brain', fontsize=14, fontweight='bold')
    
    trials = []
    performances = []
    rewards = []
    
    # Biological baseline
    bio_baseline = 50
    
    print(f"\n{'='*70}")
    print(f"{task_name.upper()}")
    print(f"{'='*70}")
    print(f"Biological baseline: {bio_baseline} trials")
    print("Training with live visualization...\n")
    
    for trial in range(max_trials):
        # Run trial
        reward, performance = test_fn(brain, trial)
        
        trials.append(trial)
        performances.append(performance)
        rewards.append(reward)
        
        # Update plot every 5 trials
        if trial % 5 == 0:
            ax1.clear()
            ax2.clear()
            
            # Plot 1: Performance over time
            ax1.plot(trials, performances, 'b-', linewidth=2, label='Performance')
            ax1.axhline(y=0.75, color='g', linestyle='--', label='Target (75%)')
            ax1.axvline(x=bio_baseline, color='r', linestyle='--', label=f'Mouse baseline ({bio_baseline} trials)')
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Performance')
            ax1.set_title('Learning Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])
            
            # Plot 2: Reward history
            window = 20
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax2.plot(range(len(smoothed)), smoothed, 'g-', linewidth=2)
            else:
                ax2.plot(trials, rewards, 'g-', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Trial')
            ax2.set_ylabel('Reward (smoothed)')
            ax2.set_title('Reward Signal')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.01)
        
        # Print progress
        if trial % 25 == 0:
            print(f"  Trial {trial}: Performance = {performance:.1%}, Reward = {reward:.3f}")
        
        # Check if learned
        if performance >= 0.75 and trial >= 25:
            print(f"\n  [SUCCESS] Learned in {trial} trials!")
            print(f"  Biological: {bio_baseline} trials")
            print(f"  Efficiency: {bio_baseline/trial:.2f}x")
            
            plt.ioff()
            plt.savefig(f'{task_name.lower().replace(" ", "_")}_learning.png', dpi=150, bbox_inches='tight')
            print(f"  Saved plot: {task_name.lower().replace(' ', '_')}_learning.png")
            plt.show(block=False)
            
            return trial
    
    print(f"\n  [TIMEOUT] Did not reach target in {max_trials} trials")
    print(f"  Final performance: {performances[-1]:.1%}")
    
    plt.ioff()
    plt.savefig(f'{task_name.lower().replace(" ", "_")}_learning.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)
    
    return max_trials


def spatial_navigation_trial(brain, trial):
    """Single trial of spatial navigation."""
    goal = np.array([0.8, 0.8])
    start = np.random.rand(2)
    cue1 = np.array([0.2, 0.9])
    cue2 = np.array([0.9, 0.2])
    
    state = np.concatenate([start, cue1, cue2])
    outputs = brain.forward(state)
    action = outputs[:2]
    
    new_pos = start + action * 0.1
    new_pos = np.clip(new_pos, 0, 1)
    
    distance = np.linalg.norm(new_pos - goal)
    reward = 1.0 if distance < 0.15 else -0.1 * distance
    
    brain.train_step(state, reward)
    
    # Test performance
    if trial % 5 == 0:
        successes = 0
        for _ in range(10):
            test_start = np.random.rand(2)
            test_state = np.concatenate([test_start, cue1, cue2])
            test_output = brain.forward(test_state)
            test_action = test_output[:2]
            test_pos = test_start + test_action * 0.1
            test_pos = np.clip(test_pos, 0, 1)
            if np.linalg.norm(test_pos - goal) < 0.15:
                successes += 1
        performance = successes / 10
    else:
        performance = brain.performance_history[-1] if brain.performance_history else 0
    
    brain.performance_history.append(performance)
    
    return reward, performance


def main():
    """Run benchmark with visualization."""
    print("\n" + "="*70)
    print("MOUSE BENCHMARK WITH VISUALIZATION - 10M NEURONS")
    print("="*70)
    print("\nThis will show real-time learning progress!")
    print("Watch the plots update as the brain learns.\n")
    
    # Create brain
    brain = MouseScaleBrain(n_neurons=10_000_000, n_connections=2_000_000)
    
    # Run spatial navigation test with visualization
    trials = test_with_visualization(
        brain,
        "Spatial Navigation (Morris Water Maze)",
        spatial_navigation_trial,
        max_trials=200
    )
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"\nLearning completed in {trials} trials")
    print(f"Biological baseline: 50 trials")
    print(f"Efficiency: {50/trials:.2f}x" if trials < 200 else "Did not reach target")
    
    input("\nPress Enter to close plots...")


if __name__ == "__main__":
    main()
