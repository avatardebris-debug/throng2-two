"""
q_policy.py --- Stable Q-Network for z-space training.

Operates on the latent z-vectors produced by UniversalEncoder.
Includes built-in persistence (save/load) and exploration logic.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QPolicy:
    """
    DQN-style policy wrapper for Throng2.
    """
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 100000,
    ):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._steps = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(input_dim, n_actions).to(self.device)
        self.target_model = QNetwork(input_dim, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state_z: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_t = torch.as_tensor(state_z, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state_t)
            return int(torch.argmax(q_values).item())

    def select_batch(self, states_z: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Batched action selection for VectorizedImaginedEnv."""
        N = states_z.shape[0]
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions, size=N)
            
        with torch.no_grad():
            states_t = torch.as_tensor(states_z, dtype=torch.float32).to(self.device)
            q_values = self.model(states_t)
            return torch.argmax(q_values, dim=1).cpu().numpy()

    def update_epsilon(self):
        """Decay epsilon over time."""
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (1.0 - self.epsilon_end) / self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)

    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """Standard DQN update step."""
        states, actions, next_states, rewards, dones = batch
        
        # Current Q
        q_values = self.model(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q
            
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self._steps += 1
        if self._steps % 1000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        return loss.item()

    def save(self, path: str):
        """Save policy weights."""
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self._steps
        }, path)

    def load(self, path: str):
        """Load policy weights."""
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", 1.0)
        self._steps = checkpoint.get("steps", 0)
