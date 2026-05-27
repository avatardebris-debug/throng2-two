"""
Unified Mario agent — single entry point for PPO / PPO+ICM / torch backends.

Usage:
    agent = make_mario_agent(curiosity=True)          # numpy ICM (default)
    agent = make_mario_agent(backend="torch")         # GPU CoordConv + ICM
    agent = make_mario_agent(curiosity=False)         # numpy PPO only

Legacy names remain available:
    MarioRLAgent, MarioICMAgent, MarioTorchAgent
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Optional

import numpy as np

from .backends.numpy_ppo import MarioRLAgent as _NumpyPPOAgent

_TORCH_AVAILABLE = True
try:
    from .mario_torch_agent import MarioTorchAgent as _TorchAgent
except ImportError:
    _TORCH_AVAILABLE = False
    _TorchAgent = None

from .mario_icm_agent import MarioICMAgent as _NumpyICMAgent

BackendName = Literal["numpy", "torch"]


@dataclass
class MarioAgentConfig:
    obs_dim: int = 378
    n_actions: int = 8
    backend: BackendName = "numpy"
    curiosity: bool = True
    hidden1: int = 128
    hidden2: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_length: int = 128
    update_epochs: int = 4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    icm_feature_dim: int = 32
    icm_hidden_dim: int = 64
    icm_lr: float = 1e-3
    intrinsic_lambda: float = 0.5

    def ppo_kwargs(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "lr": self.lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "rollout_length": self.rollout_length,
            "update_epochs": self.update_epochs,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
        }

    def icm_kwargs(self) -> Dict[str, Any]:
        return {
            "icm_feature_dim": self.icm_feature_dim,
            "icm_hidden_dim": self.icm_hidden_dim,
            "icm_lr": self.icm_lr,
            "intrinsic_lambda": self.intrinsic_lambda,
        }


class MarioAgent:
    """
    Facade over numpy PPO, numpy PPO+ICM, or torch PPO+ICM.

    Trainers should call learn_with_next_obs when possible (required for ICM).
    """

    def __init__(
        self,
        config: Optional[MarioAgentConfig] = None,
        **kwargs: Any,
    ):
        cfg = config or MarioAgentConfig()
        if kwargs:
            valid = {f.name for f in fields(MarioAgentConfig)}
            cfg = MarioAgentConfig(
                **{**{k: getattr(cfg, k) for k in valid}, **{k: v for k, v in kwargs.items() if k in valid}}
            )

        self.config = cfg
        self._impl = self._build_impl(cfg)

    @classmethod
    def from_hpo_config(cls, hpo: Dict[str, float], **overrides: Any) -> "MarioAgent":
        """Build from mario_hpo search-space dict."""
        int_keys = {"hidden1", "hidden2", "rollout_length", "update_epochs", "n_actions", "obs_dim",
                    "icm_feature_dim", "icm_hidden_dim"}
        kw: Dict[str, Any] = {}
        for f in fields(MarioAgentConfig):
            if f.name not in hpo:
                continue
            v = hpo[f.name]
            kw[f.name] = int(v) if f.name in int_keys else float(v)
        kw.update(overrides)
        return cls(**kw)

    @staticmethod
    def _build_impl(cfg: MarioAgentConfig):
        if cfg.backend == "torch":
            if not _TORCH_AVAILABLE or _TorchAgent is None:
                raise ImportError("PyTorch required for backend='torch'. pip install torch")
            return _TorchAgent(
                obs_dim=cfg.obs_dim,
                n_actions=cfg.n_actions,
                hidden1=cfg.hidden1,
                hidden2=cfg.hidden2,
                lr=cfg.lr,
                gamma=cfg.gamma,
                rollout_length=cfg.rollout_length,
                icm_feature_dim=cfg.icm_feature_dim,
                icm_hidden_dim=cfg.icm_hidden_dim,
                icm_lr=cfg.icm_lr,
                intrinsic_lambda=cfg.intrinsic_lambda,
            )
        if cfg.curiosity:
            icm_ppo = {
                k: v
                for k, v in cfg.ppo_kwargs().items()
                if k
                in (
                    "obs_dim",
                    "n_actions",
                    "hidden1",
                    "hidden2",
                    "lr",
                    "gamma",
                    "rollout_length",
                )
            }
            return _NumpyICMAgent(**{**icm_ppo, **cfg.icm_kwargs()})
        return _NumpyPPOAgent(**cfg.ppo_kwargs())

    def step(self, obs: np.ndarray) -> int:
        return self._impl.step(obs)

    def learn(self, reward: float, done: bool) -> Optional[dict]:
        if hasattr(self._impl, "learn"):
            return self._impl.learn(reward, done)
        return None

    def learn_with_next_obs(
        self, reward: float, done: bool, next_obs: np.ndarray
    ) -> Optional[dict]:
        if hasattr(self._impl, "learn_with_next_obs"):
            return self._impl.learn_with_next_obs(reward, done, next_obs)
        return self.learn(reward, done)

    def reset(self) -> None:
        self._impl.reset()

    def save(self, path: str) -> None:
        if hasattr(self._impl, "save"):
            self._impl.save(path)
        elif hasattr(self._impl, "save_weights"):
            self._impl.save_weights(path)

    def load(self, path: str) -> None:
        if hasattr(self._impl, "load"):
            self._impl.load(path)
        elif hasattr(self._impl, "load_weights"):
            self._impl.load_weights(path)

    @property
    def backend(self) -> str:
        return self.config.backend

    @property
    def uses_curiosity(self) -> bool:
        return self.config.curiosity and self.config.backend == "numpy"

    def stats(self) -> dict:
        if hasattr(self._impl, "stats"):
            return self._impl.stats()
        return {}

    def __getattr__(self, name: str):
        return getattr(self._impl, name)


def make_mario_agent(
    *,
    backend: Optional[str] = None,
    curiosity: bool = True,
    obs_dim: int = 378,
    n_actions: int = 8,
    **kwargs: Any,
) -> MarioAgent:
    """
    Factory: auto-select torch when requested and available, else numpy.

    Args:
        backend: "numpy" | "torch" | None (None → numpy)
        curiosity: If True and backend is numpy/torch ICM, enable ICM
        obs_dim, n_actions: Mario ASCII observation size
        **kwargs: Passed to MarioAgentConfig (lr, intrinsic_lambda, ...)
    """
    if backend is None:
        backend = "numpy"
    if backend == "torch" and not _TORCH_AVAILABLE:
        backend = "numpy"
    return MarioAgent(
        MarioAgentConfig(
            obs_dim=obs_dim,
            n_actions=n_actions,
            backend=backend,  # type: ignore[arg-type]
            curiosity=curiosity,
            **{k: v for k, v in kwargs.items() if k in {f.name for f in fields(MarioAgentConfig)}},
        )
    )


# Backward-compatible aliases
MarioRLAgent = _NumpyPPOAgent
MarioICMAgent = _NumpyICMAgent
if _TorchAgent is not None:
    MarioTorchAgent = _TorchAgent
else:
    MarioTorchAgent = None  # type: ignore[misc, assignment]

__all__ = [
    "MarioAgent",
    "MarioAgentConfig",
    "make_mario_agent",
    "MarioRLAgent",
    "MarioICMAgent",
    "MarioTorchAgent",
]
