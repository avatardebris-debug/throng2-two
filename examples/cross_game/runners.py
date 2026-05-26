"""Environment runners for cross-game world model training."""
from __future__ import annotations

import numpy as np

from src.games.mario.mario_generator import MarioLevelGenerator
from src.games.mario.mario_adapter import MarioAdapter

_GYM_AVAILABLE = True
try:
    import gymnasium as gym
except ImportError:
    _GYM_AVAILABLE = False

# Observation dims for gym envs used in cross-game training (env_name substring → dim)
GYM_OBS_DIMS = {
    "cartpole": 4,
    "mountaincar": 2,
    "lunarlander": 8,
    "gridworld": 16,
}


def gym_obs_dim_for_env(env_name: str, default: int = 8) -> int:
    """Resolve agent obs_dim from a gymnasium env name."""
    name = env_name.lower()
    return next((dim for key, dim in GYM_OBS_DIMS.items() if key in name), default)


class MarioRunner:
    """Wraps MarioAdapter into a simple gym-like interface."""

    def __init__(self, tier: int = 2, seed: int = 42):
        self.gen = MarioLevelGenerator(seed=seed)
        self.adapter = MarioAdapter()
        self.tier = tier
        self._sim = None
        self.n_actions = 8

    def reset(self):
        for _ in range(10):
            sim = self.gen.generate(tier=self.tier)
            if sim is not None:
                self._sim = sim
                return self.adapter.reset(sim)
        raise RuntimeError("Mario level gen failed")

    def step(self, action: int):
        return self.adapter.step(action)

    def get_sim(self):
        return self._sim


class GymRunner:
    """Wraps a gymnasium environment."""

    def __init__(self, env_name: str, seed: int = 42):
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium not installed. Run: pip install gymnasium")
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.n_actions = self.env.action_space.n
        self._base_seed = seed
        self._episode = 0

    def reset(self):
        obs, _ = self.env.reset(seed=self._base_seed + self._episode)
        self._episode += 1
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return np.asarray(obs, dtype=np.float32), float(reward), done, info

    def close(self):
        self.env.close()


GAME_CONFIGS = {
    "mario": {
        "runner_cls": MarioRunner,
        "runner_kwargs": {"tier": 2},
        "n_actions": 8,
        "game_id": 0,
    },
    "cartpole": {
        "runner_cls": GymRunner,
        "runner_kwargs": {"env_name": "CartPole-v1"},
        "n_actions": 2,
        "game_id": 1,
    },
    "mountaincar": {
        "runner_cls": GymRunner,
        "runner_kwargs": {"env_name": "MountainCar-v0"},
        "n_actions": 3,
        "game_id": 2,
    },
    "lunarlander": {
        "runner_cls": GymRunner,
        "runner_kwargs": {"env_name": "LunarLander-v2"},
        "n_actions": 4,
        "game_id": 3,
    },
}
