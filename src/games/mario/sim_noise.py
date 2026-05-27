"""
sim_noise.py — Domain randomization wrapper for MarioSimulator.

Wraps any MarioSimulator with per-episode physics perturbation and
per-step observation noise. Makes the policy robust to sim-to-real
gaps by training on a distribution of physics, not one exact setting.

KEY DESIGN: Noise is SCHEDULED, not constant.
  Following Go-Explore (Ecoffet 2021):
    Phase 1 (Explore):  zero noise → learn clean mechanics first
    Phase 2 (Ramp):     linearly increase noise as success_rate improves
    Phase 3 (Full):     full noise for robustification

  noise_scale = 0.0 until success_rate > ramp_start_threshold (default 0.3)
  then linearly ramps to 1.0 as success_rate → ramp_full_threshold (default 0.8)

Gym-compatible: has reset() → obs and step(action) → (obs, rew, done, info).

Noise categories:
  - Action noise:    random action substitution (controller lag/error)
  - Observation noise: gaussian, tile corruption, frame drops
  - Reward noise:    ±10% uniform (reward shaping differences)
"""

from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np

from .mario_simulator import MarioSimulator, N_ACTIONS


# ══════════════════════════════════════════════════════════════════════
#  Default noise config (at FULL scale=1.0)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_NOISE: Dict[str, float] = {
    # Per-step noise (scaled by noise_scale)
    'action_flip_prob':   0.05,   # 5% chance action replaced with random
    'obs_gaussian_std':   0.02,   # Gaussian noise on obs values
    'tile_corrupt_prob':  0.02,   # 2% of viewport tiles randomly corrupted
    'frame_drop_prob':    0.03,   # 3% chance obs is stale (previous frame)
    'reward_noise_range': 0.10,   # ±10% reward perturbation
}

# Zero noise config for clean evaluation
ZERO_NOISE: Dict[str, float] = {k: 0.0 for k in DEFAULT_NOISE}


# ══════════════════════════════════════════════════════════════════════
#  NoisyMarioSim
# ══════════════════════════════════════════════════════════════════════

class NoisyMarioSim:
    """
    Gymnasium-compatible wrapper with Go-Explore-style noise scheduling.

    Noise is scaled by noise_scale ∈ [0, 1]:
      - noise_scale = 0.0 → clean environment (Phase 1: explore)
      - noise_scale = 0.5 → half noise (Phase 2: ramp)
      - noise_scale = 1.0 → full noise (Phase 3: robustify)

    noise_scale is controlled externally via set_noise_scale() or
    automatically by the NoiseScheduler.

    Usage:
        sim = NoisyMarioSim(lambda: some_simulator)
        sim.set_noise_scale(0.0)   # clean for exploration
        obs = sim.reset()
        obs, rew, done, info = sim.step(action)
        # Later, when agent is competent:
        sim.set_noise_scale(0.5)   # start robustifying
    """

    def __init__(
        self,
        sim_factory: Callable[[], MarioSimulator],
        noise_config: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        initial_noise_scale: float = 0.0,
    ):
        self.factory = sim_factory
        self.cfg = {**DEFAULT_NOISE, **(noise_config or {})}
        self.rng = np.random.RandomState(seed)
        self.sim: Optional[MarioSimulator] = None
        self._prev_obs: Optional[np.ndarray] = None
        self._noise_scale: float = initial_noise_scale

        # Interface
        self.obs_dim = 378
        self.n_actions = N_ACTIONS

    @property
    def noise_scale(self) -> float:
        return self._noise_scale

    def set_noise_scale(self, scale: float) -> None:
        """Set noise intensity: 0.0 = clean, 1.0 = full noise."""
        self._noise_scale = max(0.0, min(1.0, scale))

    def _scaled(self, key: str) -> float:
        """Return noise parameter scaled by current noise_scale."""
        return self.cfg[key] * self._noise_scale

    def reset(self) -> np.ndarray:
        """Reset with fresh sim."""
        self.sim = self.factory()
        obs = self.sim.get_obs()
        self._prev_obs = obs.copy()
        return self._apply_obs_noise(obs)

    def step(self, action: int):
        """Step with scaled noise."""
        assert self.sim is not None, "Call reset() first"

        # Action noise (scaled)
        if self.rng.random() < self._scaled('action_flip_prob'):
            action = self.rng.randint(0, N_ACTIONS)

        obs, reward, done, info = self.sim.step(int(action))

        # Reward noise (scaled)
        noise_range = self._scaled('reward_noise_range')
        if noise_range > 0 and abs(reward) > 0.001:
            reward *= self.rng.uniform(1.0 - noise_range, 1.0 + noise_range)

        obs = self._apply_obs_noise(obs)
        return obs, reward, done, info

    def _apply_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        """Apply per-step observation noise (scaled)."""
        if self._noise_scale < 0.001:
            self._prev_obs = obs.copy()
            return obs  # Skip entirely when noise is off

        obs = obs.copy()

        # Frame drop
        if self.rng.random() < self._scaled('frame_drop_prob'):
            if self._prev_obs is not None:
                return self._prev_obs.copy()

        # Gaussian noise
        std = self._scaled('obs_gaussian_std')
        if std > 0:
            obs += self.rng.normal(0, std, obs.shape).astype(np.float32)

        # Tile corruption in viewport
        corrupt_prob = self._scaled('tile_corrupt_prob')
        if corrupt_prob > 0:
            n_corrupt = int(320 * corrupt_prob)
            if n_corrupt > 0:
                idx = self.rng.choice(320, n_corrupt, replace=False)
                obs[idx] = self.rng.uniform(0, 1, n_corrupt).astype(np.float32)

        obs = np.clip(obs, -1.0, 2.0)
        self._prev_obs = obs.copy()
        return obs

    def render_ascii(self) -> str:
        if self.sim is None:
            return "(not initialized)"
        return self.sim.render_ascii()

    def close(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════
#  Noise Scheduler — Go-Explore-style ramp
# ══════════════════════════════════════════════════════════════════════

class NoiseScheduler:
    """
    Automatically adjusts noise_scale based on agent success rate.

    Follows Go-Explore's insight:
      Phase 1: success_rate < ramp_start → noise_scale = 0.0 (clean explore)
      Phase 2: ramp_start ≤ success_rate < ramp_full → linear ramp 0→1
      Phase 3: success_rate ≥ ramp_full → noise_scale = 1.0 (full robustify)

    Usage:
        scheduler = NoiseScheduler()
        # Each episode:
        scheduler.report(won=True)   # or False
        noisy_sim.set_noise_scale(scheduler.noise_scale)
    """

    def __init__(
        self,
        ramp_start: float = 0.30,    # start adding noise at 30% success
        ramp_full:  float = 0.80,    # full noise at 80% success
        window: int = 100,            # success rate measured over last N episodes
    ):
        self.ramp_start = ramp_start
        self.ramp_full = ramp_full
        self.window = window
        self._results: list = []
        self._noise_scale: float = 0.0

    @property
    def noise_scale(self) -> float:
        return self._noise_scale

    @property
    def success_rate(self) -> float:
        if not self._results:
            return 0.0
        recent = self._results[-self.window:]
        return sum(recent) / len(recent)

    def report(self, won: bool) -> float:
        """Report episode result. Returns updated noise_scale."""
        self._results.append(1.0 if won else 0.0)

        rate = self.success_rate
        if rate < self.ramp_start:
            self._noise_scale = 0.0
        elif rate >= self.ramp_full:
            self._noise_scale = 1.0
        else:
            # Linear ramp between ramp_start and ramp_full
            frac = (rate - self.ramp_start) / (self.ramp_full - self.ramp_start)
            self._noise_scale = frac

        return self._noise_scale

    def stats(self) -> dict:
        return {
            'noise_scale': round(self._noise_scale, 3),
            'success_rate': round(self.success_rate, 3),
            'episodes': len(self._results),
            'phase': (
                'explore' if self._noise_scale < 0.01 else
                'ramp' if self._noise_scale < 0.99 else
                'robustify'
            ),
        }
