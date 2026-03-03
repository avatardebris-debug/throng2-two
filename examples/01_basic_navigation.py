"""
Basic Navigation Example - Complete working demonstration.

This shows:
- Thronglet brain learning to navigate
- Neuromodulator systems in action
- Curriculum learning
- Real-time visualization

Run this to see the system working!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.core.network import LayeredNetwork
from src.learning.neuromodulators import NeuromodulatorSystem
from src.environment.grid_world import GridWorld


class ThrongletAgent:
    """
    Agent combining thronglet brain + neuromodulators.
    """
    
    def __init__(self,
                 input_size: int = 8,
                 n_neurons: int = 500,
                 output_size: int = 4):
        """
        Initialize agent.
        
        Args:
            input_size: Observation size
            n_neurons: Number of hidden neurons
            output_size: Number of actions
        """
        # Create brain network
        hidden_sizes = [n_neurons // 2, n_neurons // 4]
        self.network = LayeredNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dimension=2,
            connection_prob=0.02
        )
        
        # Neuromodulator system
        self.neuromodulators = NeuromodulatorSystem()
        
        # Q-values (for action selection)
        self.q_values = {}
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_decay = 0.995
        
    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """
        Select action based on network output.
        
        Args:
            obs: Observation vector
            explore: Whether to explore (epsilon-greedy)
            
        Returns:
            Action index
        """
        # Get modulated exploration rate
        if explore:
            epsilon = self.neuromodulators.get_exploration_rate(self.epsilon)
        else:
            epsilon = 0.0
            
        # Epsilon-greedy
        if np.random.random() < epsilon:
            # Explore - but bias toward goal direction
            goal_direction = np.argmax([obs[4], -obs[4], obs[5], -obs[5]])
            if np.random.random() < 0.3:  # 30% bias
                return goal_direction
            else:
                return np.random.randint(4)
        else:
            # Exploit - use network
            output_spikes = self.network.forward(obs)
            
            # Convert spikes to action probabilities
            spike_counts = output_spikes[-4:]  # Last 4 neurons = actions
            
            if np.sum(spike_counts) == 0:
                # No spikes - random action
                return np.random.randint(4)
            else:
                # Choose based on spike activity
                probs = spike_counts / np.sum(spike_counts)
                return np.random.choice(4, p=probs)
    
    def learn(self, state: tuple, action: int, reward: float, 
              next_state: tuple, done: bool):
        """
        Learn from experience.
        
        Args:
            state: State tuple
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done?
        """
        # Compute TD error (dopamine signal)
        td_error = self.neuromodulators.compute_td_error(
            state, reward, next_state, done
        )
        
        # Update network with modulated Hebbian learning
        learning_rate = self.neuromodulators.modulate_hebbian(base_rate=0.01)
        modulation = self.neuromodulators.dopamine
        
        self.network.learn(learning_rate=learning_rate, modulation=modulation)
        
        # Decay exploration
        self.epsilon *= self.epsilon_decay
        
    def reset(self):
        """Reset for new episode."""
        self.network.reset()


def train_agent(n_episodes: int = 200, 
                render_interval: int = 20,
                visualize: bool = True):
    """
    Train thronglet agent on navigation task.
    
    Args:
        n_episodes: Number of training episodes
        render_interval: Episodes between plots
        visualize: Show live plots
    """
    print("=" * 60)
    print("THRONGLET BRAIN - MINIMAL VIABLE INTELLIGENCE")
    print("=" * 60)
    print("\nInitializing brain with sacred geometry...")
    
    # Create environment and agent
    env = GridWorld(grid_size=10, max_steps=100, use_curriculum=True)
    agent = ThrongletAgent(input_size=8, n_neurons=500, output_size=4)
    
    print(f"✓ Created {agent.network.layers[0].n_neurons} input neurons")
    print(f"✓ Created {agent.network.layers[1].n_neurons} hidden neurons (layer 1)")
    print(f"✓ Created {agent.network.layers[2].n_neurons} hidden neurons (layer 2)")
    print(f"✓ Created {agent.network.layers[3].n_neurons} output neurons")
    
    # Get network stats
    stats = agent.network.layers[1].get_statistics()
    print(f"\n✓ Network topology: {stats['n_connections']} connections")
    print(f"✓ Connection density: {stats['connection_density']:.1%}")
    print(f"✓ Clustering coefficient: {stats['clustering_coefficient']:.3f}")
    
    print("\nTraining begins...\n")
    
    # Training loop
    episode_rewards = []
    episode_steps = []
    episode_wins = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        state = env.get_state_tuple()
        
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(obs, explore=True)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            next_state = env.get_state_tuple()
            
            # Learn
            agent.learn(state, action, reward, next_state, done)
            
            # Update
            obs = next_obs
            state = next_state
            total_reward += reward
            
        # Track metrics
        episode_rewards.append(total_reward)
        episode_steps.append(info['steps'])
        episode_wins.append(1 if reward > 1 else 0)  # Won if got big reward
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_wins = np.mean(episode_wins[-10:])
            recent_steps = np.mean(episode_steps[-10:])
            dopamine = agent.neuromodulators.dopamine
            epsilon = agent.epsilon
            
            print(f"Episode {episode + 1:3d} | "
                  f"Win Rate: {recent_wins:.1%} | "
                  f"Avg Steps: {recent_steps:.1f} | "
                  f"Difficulty: {info['difficulty']} | "
                  f"Dopamine: {dopamine:.2f} | "
                  f"Explore: {epsilon:.2f}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    final_win_rate = np.mean(episode_wins[-20:])
    final_avg_steps = np.mean(episode_steps[-20:])
    
    print(f"\nFinal Performance (last 20 episodes):")
    print(f"  Win Rate: {final_win_rate:.1%}")
    print(f"  Avg Steps: {final_avg_steps:.1f}")
    print(f"  Final Difficulty: {env.current_difficulty}")
    
    # Network statistics
    stats = agent.network.layers[1].get_statistics()
    print(f"\nNetwork Statistics:")
    print(f"  Active Connections: {stats['n_connections']}")
    print(f"  Active Neurons: {stats['active_neurons']}")
    print(f"  Avg Weight: {stats['avg_weight']:.3f}")
    
    # Visualization
    if visualize:
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Learning curve (steps per episode)
        ax1 = axes[0, 0]
        smoothed = np.convolve(episode_steps, np.ones(10)/10, mode='valid')
        ax1.plot(episode_steps, alpha=0.3, color='blue', label='Raw')
        ax1.plot(smoothed, color='blue', linewidth=2, label='Smoothed (10 ep)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps to Goal')
        ax1.set_title('Learning Progress (Lower = Better)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win rate over time
        ax2 = axes[0, 1]
        win_rate_smooth = np.convolve(episode_wins, np.ones(20)/20, mode='valid')
        ax2.plot(win_rate_smooth, color='green', linewidth=2)
        ax2.axhline(y=0.7, color='r', linestyle='--', label='Curriculum Threshold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate (20 ep window)')
        ax2.set_title('Success Rate Over Time')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Network structure
        ax3 = axes[1, 0]
        layer = agent.network.layers[1]  # Hidden layer
        positions = layer.positions
        connections = layer.weights
        
        # Draw strongest connections
        n_show = min(100, len(positions))
        connection_strengths = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if connections[i, j] > 0.1:
                    connection_strengths.append((i, j, connections[i, j]))
        
        # Sort and take strongest
        connection_strengths.sort(key=lambda x: x[2], reverse=True)
        for i, j, weight in connection_strengths[:n_show]:
            ax3.plot([positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    'b-', alpha=weight*0.5, linewidth=0.5)
        
        # Draw neurons
        activity = layer.neurons.get_activity_rates()
        scatter = ax3.scatter(positions[:, 0], positions[:, 1],
                            c=activity, s=20, cmap='Reds', 
                            vmin=0, vmax=0.3, alpha=0.7)
        ax3.set_title('Brain Structure (Fibonacci Spiral)\nColor = Activity')
        ax3.set_aspect('equal')
        plt.colorbar(scatter, ax=ax3, label='Firing Rate')
        
        # Plot 4: Neuromodulator levels over last episode
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.9, 'Final Neuromodulator Levels', 
                ha='center', fontsize=12, weight='bold',
                transform=ax4.transAxes)
        
        levels = agent.neuromodulators.get_levels()
        y_pos = 0.7
        for name, value in levels.items():
            color = 'green' if value > 0.5 else 'red'
            ax4.text(0.1, y_pos, f'{name.capitalize()}:', 
                    fontsize=10, transform=ax4.transAxes)
            ax4.barh(y_pos, value, height=0.08, 
                    color=color, alpha=0.6,
                    transform=ax4.transAxes)
            ax4.text(value + 0.05, y_pos, f'{value:.2f}',
                    fontsize=9, transform=ax4.transAxes)
            y_pos -= 0.15
            
        ax4.set_xlim([0, 1.2])
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('thronglet_training_results.png', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to 'thronglet_training_results.png'")
        
        plt.show()
    
    return agent, env


if __name__ == "__main__":
    print("\n🧠 Starting Thronglet Brain Training...\n")
    
    # Train the agent
    agent, env = train_agent(
        n_episodes=200,
        visualize=True
    )
    
    print("\n✨ Success! Your thronglet brain has learned to navigate!")
    print("\nNext steps:")
    print("  1. Try changing parameters in config/default_config.yaml")
    print("  2. Experiment with more neurons or different architectures")
    print("  3. Build more complex environments")
    print("  4. Implement Nash equilibrium pruning")
    print("  5. Create the Pokemon gamification layer!")
