"""
Simple 2D Grid World - Navigation environment with curriculum learning.

Ensures agent experiences success early so TD learning can work.
"""

import numpy as np
from typing import Tuple, List


class GridWorld:
    """
    2D grid navigation task.
    
    Features:
    - Configurable grid size
    - Curriculum learning (start easy, get harder)
    - Reward shaping (continuous feedback)
    - Obstacles (optional)
    """
    
    def __init__(self,
                 grid_size: int = 10,
                 max_steps: int = 100,
                 use_curriculum: bool = True,
                 initial_difficulty: int = 2):
        """
        Initialize environment.
        
        Args:
            grid_size: Size of grid (grid_size x grid_size)
            max_steps: Maximum steps before episode ends
            use_curriculum: Start with goal close, gradually move farther
            initial_difficulty: Initial distance to goal
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.use_curriculum = use_curriculum
        self.initial_difficulty = initial_difficulty
        
        # Current difficulty (distance to goal)
        self.current_difficulty = initial_difficulty
        
        # State
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([initial_difficulty, initial_difficulty])
        self.steps_taken = 0
        
        # Statistics
        self.total_episodes = 0
        self.recent_wins = []
        self.max_recent = 20
        
        # Action mapping
        self.actions = {
            0: np.array([1, 0]),   # Right
            1: np.array([-1, 0]),  # Left
            2: np.array([0, 1]),   # Up
            3: np.array([0, -1])   # Down
        }
        
    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for new episode.
        
        Returns:
            Initial observation and info dict
        """
        # Start at origin
        self.agent_pos = np.array([0, 0])
        
        # Place goal based on current difficulty
        if self.use_curriculum:
            # Goal within current_difficulty steps
            max_dist = min(self.current_difficulty, self.grid_size - 1)
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(max_dist * 0.5, max_dist)
            self.goal_pos = np.array([
                int(dist * np.cos(angle)),
                int(dist * np.sin(angle))
            ])
        else:
            # Random goal
            self.goal_pos = np.random.randint(0, self.grid_size, size=2)
            
        # Clip to grid
        self.goal_pos = np.clip(self.goal_pos, 0, self.grid_size - 1)
        
        self.steps_taken = 0
        self.total_episodes += 1
        
        obs = self._get_observation()
        info = {'difficulty': self.current_difficulty}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in environment.
        
        Args:
            action: 0=Right, 1=Left, 2=Up, 3=Down
            
        Returns:
            observation, reward, done, info
        """
        # Previous distance to goal
        prev_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        # Take action
        self.agent_pos = self.agent_pos + self.actions[action]
        
        # Clip to grid bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)
        
        self.steps_taken += 1
        
        # Calculate reward
        current_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        
        # Check if reached goal
        if current_dist < 0.5:
            reward = 5.0  # Big reward for success!
            done = True
            self.recent_wins.append(1)
        else:
            # Reward shaping: small reward for getting closer
            if current_dist < prev_dist:
                reward = 0.1 + 0.2 * (prev_dist - current_dist)
            else:
                reward = -0.05  # Small penalty for moving away
                
            # Episode ends if max steps reached
            done = self.steps_taken >= self.max_steps
            if done:
                self.recent_wins.append(0)
                
        # Update curriculum
        if done and self.use_curriculum:
            self._update_curriculum()
            
        # Limit recent wins memory
        if len(self.recent_wins) > self.max_recent:
            self.recent_wins.pop(0)
            
        obs = self._get_observation()
        info = {
            'steps': self.steps_taken,
            'distance': current_dist,
            'difficulty': self.current_difficulty
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation vector.
        
        Returns:
            [agent_x, agent_y, goal_x, goal_y, dx, dy, distance, angle]
        """
        # Normalize to 0-1
        agent_norm = self.agent_pos / self.grid_size
        goal_norm = self.goal_pos / self.grid_size
        
        # Relative position
        dx = goal_norm[0] - agent_norm[0]
        dy = goal_norm[1] - agent_norm[1]
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) / np.pi  # Normalize to -1, 1
        
        obs = np.array([
            agent_norm[0], agent_norm[1],
            goal_norm[0], goal_norm[1],
            dx, dy,
            distance,
            angle
        ])
        
        return obs
    
    def _update_curriculum(self):
        """
        Update difficulty based on performance.
        
        If winning consistently, make it harder.
        If losing consistently, make it easier.
        """
        if len(self.recent_wins) < 10:
            return
            
        win_rate = np.mean(self.recent_wins[-10:])
        
        # Winning 70%+ -> increase difficulty
        if win_rate > 0.7 and self.current_difficulty < self.grid_size - 1:
            self.current_difficulty += 1
            
        # Winning <30% -> decrease difficulty
        elif win_rate < 0.3 and self.current_difficulty > 2:
            self.current_difficulty -= 1
            
    def get_state_tuple(self) -> tuple:
        """Get discrete state for tabular methods."""
        return (tuple(self.agent_pos), tuple(self.goal_pos))
    
    def get_statistics(self) -> dict:
        """Get environment statistics."""
        if len(self.recent_wins) == 0:
            win_rate = 0.0
        else:
            win_rate = np.mean(self.recent_wins)
            
        return {
            'total_episodes': self.total_episodes,
            'current_difficulty': self.current_difficulty,
            'recent_win_rate': win_rate,
            'total_wins': sum(self.recent_wins)
        }
    
    def render_text(self) -> str:
        """Simple text rendering."""
        grid = [['.' for _ in range(self.grid_size)] 
                for _ in range(self.grid_size)]
        
        # Place goal
        gx, gy = self.goal_pos
        grid[gy][gx] = 'G'
        
        # Place agent
        ax, ay = self.agent_pos
        grid[ay][ax] = 'A'
        
        # Convert to string
        lines = [''.join(row) for row in grid]
        return '\n'.join(lines)
