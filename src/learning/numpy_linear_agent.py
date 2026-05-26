"""Lightweight numpy policy for low-dimensional gym envs in cross-game training."""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class SimpleNumpyAgent:
    """Tiny linear Q-style agent (pure numpy) for CartPole, MountainCar, etc."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        epsilon: float = 0.2,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.W = np.zeros((obs_dim, n_actions), dtype=np.float32)
        self.b = np.zeros(n_actions, dtype=np.float32)
        self._buf: List = []
        self._gamma = 0.99
        self.total_reward = 0.0
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None

    def _q(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self.W + self.b

    def step(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32).flatten()
        if np.random.random() < self.epsilon:
            action = int(np.random.randint(self.n_actions))
        else:
            action = int(np.argmax(self._q(obs)))
        self._prev_obs = obs
        self._prev_action = action
        return action

    def learn(self, reward: float, next_obs: np.ndarray, done: bool):
        if self._prev_obs is None:
            return
        self.total_reward += reward
        next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
        target = reward + (0 if done else self._gamma * float(self._q(next_obs).max()))
        td = target - float(self._q(self._prev_obs)[self._prev_action])
        grad_w = np.outer(self._prev_obs, np.eye(self.n_actions)[self._prev_action])
        self.W += self.lr * td * grad_w
        self.b[self._prev_action] += self.lr * td

    def reset(self):
        self._prev_obs = None
        self._prev_action = None
        self.total_reward = 0.0
