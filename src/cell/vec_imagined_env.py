"""
vec_imagined_env.py --- N imagined environments stepped in a single WM forward pass.

The Universal Fast-Imagination Backbone.

Architecture:
    N imagined z-states (N x z_dim)
        -> WM.predict_batch(states, actions)    [ONE forward pass]
        -> N next z-states + N rewards + N done_probs
        -> policy sees N x (z_dim + snn_features) observations
        -> ready for the same batched PPO that trains on real envs

This is the "sim2real flip" --- instead of:
    real_env.step()  xN  (serial Python calls, ~4ms each, 85% of budget)

We do:
    wm.predict_batch()  x1  (single matmul, ~0.5ms for any N)

Throughput math (N=12, z_dim=16):
    Real sim:     12 x 4ms = 48ms/step -> ~250 sps/env -> 3,000 total sps
    WM imagined:  1 x 0.5ms = 0.5ms/step -> ~24,000 sps/env -> 288,000 total sps
    (limited in practice by PPO update, ~13ms amortized -> ~10,000 sustained sps)

Usage:
    wm  = CellWorldModel(feature_dim=16, n_actions=8)
    vec = VectorizedImaginedEnv(wm, n_envs=64)

    obs = vec.reset(initial_z_batch)
    for step in range(100_000):
        actions = policy.select_batch(obs)
        obs, rewards, dones = vec.step(actions)
"""

from __future__ import annotations

import logging
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.cell.world_model import CellWorldModel

_log = logging.getLogger(__name__)


@dataclass
class VecImaginedStats:
    n_steps:           int   = 0
    total_elapsed_s:   float = 0.0
    n_reality_checks:  int   = 0
    n_sim2real_drifts: int   = 0
    last_sps:          float = 0.0
    peak_sps:          float = 0.0

    @property
    def avg_sps(self) -> float:
        if self.total_elapsed_s < 1e-6:
            return 0.0
        return self.n_steps / self.total_elapsed_s


class VectorizedImaginedEnv:
    """
    N imagined environments running in parallel via batched WM inference.
    """

    def __init__(
        self,
        world_model:               CellWorldModel,
        n_envs:                    int             = 12,
        z_dim:                     int             = 16,
        done_reward_threshold:     Optional[float] = None,
        done_prob_threshold:       float           = 0.5,
        reality_check_fn:          Optional[Any]   = None,
        reality_check_interval:    int             = 256,
        sim2real_drift_threshold:  float           = 0.20,
    ):
        self.wm                      = world_model
        self.n_envs                  = n_envs
        self.z_dim                   = z_dim
        self.done_threshold          = done_reward_threshold
        self.done_prob_threshold     = done_prob_threshold
        self.reality_check_fn        = reality_check_fn
        self.reality_check_interval  = reality_check_interval
        self.drift_threshold         = sim2real_drift_threshold

        self._states    = np.zeros((n_envs, z_dim), dtype=np.float32)
        self._dones     = np.zeros(n_envs, dtype=bool)
        self._ep_steps  = np.zeros(n_envs, dtype=np.int32)
        self._ep_returns = np.zeros(n_envs, dtype=np.float32)
        self._initial_states: Optional[np.ndarray] = None
        self._step_start: float = 0.0
        self._steps_since_check = 0
        self._stats = VecImaginedStats()
        self._sim2real_errors: deque = deque(maxlen=50)

    @property
    def active_count(self) -> int:
        return self.n_envs

    @property
    def obs_dim(self) -> int:
        return self.z_dim

    def reset(self, initial_states: np.ndarray) -> np.ndarray:
        if initial_states.ndim == 1:
            initial_states = np.tile(initial_states, (self.n_envs, 1))
        self._states         = initial_states.astype(np.float32).copy()
        self._initial_states = self._states.copy()
        self._dones[:]       = False
        self._ep_steps[:]    = 0
        self._ep_returns[:]  = 0.0
        self._steps_since_check = 0
        self._step_start     = time.perf_counter()
        return self._states.copy()

    def reset_env(self, env_idx: int, new_state: Optional[np.ndarray] = None) -> np.ndarray:
        if new_state is not None:
            self._states[env_idx] = new_state.astype(np.float32)
        elif self._initial_states is not None:
            self._states[env_idx] = self._initial_states[env_idx].copy()
        self._dones[env_idx]     = False
        self._ep_steps[env_idx]  = 0
        self._ep_returns[env_idx] = 0.0
        return self._states[env_idx].copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        result = self.wm.predict_batch(self._states, actions)
        
        if len(result) == 3:
            next_states, rewards, done_probs = result
        else:
            next_states, rewards = result
            done_probs = np.zeros(self.n_envs, dtype=np.float32)

        dones = done_probs > self.done_prob_threshold
        if self.done_threshold is not None:
            dones = dones | (rewards > self.done_threshold)

        for i in np.where(dones)[0]:
            next_states[i] = self.reset_env(i)

        self._states     = next_states
        self._dones      = dones
        self._ep_steps  += 1
        self._ep_returns += rewards

        self._steps_since_check += 1
        if (self.reality_check_fn is not None and self._steps_since_check >= self.reality_check_interval):
            self._run_reality_check()
            self._steps_since_check = 0

        elapsed = time.perf_counter() - t0
        self._stats.n_steps += self.n_envs
        self._stats.total_elapsed_s += elapsed
        step_sps = self.n_envs / elapsed
        self._stats.last_sps  = step_sps
        self._stats.peak_sps  = max(self._stats.peak_sps, step_sps)
        return next_states.copy(), rewards.copy(), dones.copy()

    def _run_reality_check(self):
        try:
            real_next_z = self.reality_check_fn(self._states)
            errors = np.mean(np.abs(self._states - real_next_z), axis=1)
            mean_err = float(np.mean(errors))
            self._sim2real_errors.append(mean_err)
            self._stats.n_reality_checks += 1
            if mean_err > self.drift_threshold:
                self._stats.n_sim2real_drifts += 1
        except Exception:
            _log.debug("Reality check failed; skipping drift update", exc_info=True)

    @property
    def sim2real_accuracy(self) -> float:
        if not self._sim2real_errors: return 0.0
        return float(max(0.0, 1.0 - np.mean(self._sim2real_errors)))

    def status(self) -> dict:
        return {
            "active": self.n_envs,
            "imagined": True,
            "sps": round(self._stats.last_sps),
            "avg_sps": round(self._stats.avg_sps),
        }

    def stats(self) -> dict:
        return {
            "n_envs": self.n_envs,
            "n_steps": self._stats.n_steps,
            "avg_sps": round(self._stats.avg_sps),
            "sim2real_accuracy": round(self.sim2real_accuracy, 3),
            "wm_confidence": round(self.wm.confidence, 3),
        }

    def close(self):
        pass
