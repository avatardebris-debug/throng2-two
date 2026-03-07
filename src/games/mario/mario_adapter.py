"""
mario_adapter.py — Connects MarioSimulator to the ThrongletCell pipeline.

Provides gymnasium-style interface (reset/step) and feature extraction
for use with PPO, DQN, or any RL agent.

Usage:
    from src.games.mario.mario_adapter import MarioAdapter
    from src.games.mario.mario_generator import MarioLevelGenerator

    gen = MarioLevelGenerator(seed=42)
    adapter = MarioAdapter()

    level = gen.generate(tier=1)
    obs = adapter.reset(level)

    for step in range(1000):
        action = agent.act(obs)
        obs, reward, done, info = adapter.step(action)
        if done:
            break
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .mario_simulator import Action, MarioSimulator, N_ACTIONS


class MarioAdapter:
    """
    Gymnasium-style wrapper for MarioSimulator.

    Provides:
      - reset(sim) → features
      - step(action) → (features, reward, done, info)
      - Feature extraction: grid viewport + state variables
      - Fake RAM generation (128 bytes, like LOLO/Montezuma pattern)
    """

    def __init__(self, feature_dim: Optional[int] = None):
        """
        Args:
            feature_dim: If set, pad/truncate features to this size.
                         If None, use native obs_size (378).
        """
        self.feature_dim = feature_dim
        self.sim: Optional[MarioSimulator] = None
        self._step_count = 0

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def obs_dim(self) -> int:
        if self.feature_dim:
            return self.feature_dim
        return 378  # Native obs_size

    def reset(self, sim: MarioSimulator) -> np.ndarray:
        """Reset with a new level."""
        self.sim = sim
        self._step_count = 0
        return self._get_features()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step."""
        assert self.sim is not None, "Call reset() first"
        obs, reward, done, info = self.sim.step(action)
        self._step_count += 1
        return self._get_features(), reward, done, info

    def _get_features(self) -> np.ndarray:
        """Get feature vector from current state."""
        raw = self.sim.get_obs()
        if self.feature_dim is None:
            return raw
        # Pad or truncate to feature_dim
        if len(raw) >= self.feature_dim:
            return raw[:self.feature_dim]
        return np.pad(raw, (0, self.feature_dim - len(raw)))

    def grid_to_ram(self, sim: Optional[MarioSimulator] = None) -> np.ndarray:
        """
        Generate fake 128-byte RAM representation.

        Matches the pattern used for Montezuma/LOLO:
          RAM[0]  = mario_col (x position)
          RAM[1]  = mario_row (y position)
          RAM[2]  = scroll_x (camera offset)
          RAM[3]  = on_ground (1/0)
          RAM[4]  = jump_timer
          RAM[5]  = coins
          RAM[6]  = score (clamped to 255)
          RAM[7]  = alive (1/0)
          RAM[8]  = won (1/0)
          RAM[9]  = max_x_reached
          RAM[10-17] = enemy positions (col, row pairs)
          RAM[18+] = viewport tile hashes
        """
        s = sim or self.sim
        ram = np.zeros(128, dtype=np.uint8)

        ram[0] = min(255, s.mario_col)
        ram[1] = min(255, s.mario_row)
        ram[2] = min(255, s.scroll_x)
        ram[3] = int(s.on_ground)
        ram[4] = s.jump_timer
        ram[5] = min(255, s.coins)
        ram[6] = min(255, int(s.score))
        ram[7] = int(s.alive)
        ram[8] = int(s.won)
        ram[9] = min(255, s.max_x_reached)

        # Enemy positions
        idx = 10
        for e in s.enemies[:4]:
            if e.alive:
                ram[idx] = min(255, e.col)
                ram[idx + 1] = min(255, e.row)
            idx += 2

        # Viewport tile hash (compact representation)
        vp_start = max(0, min(s.mario_col - 10, s.width - 20))
        vp_end = min(vp_start + 20, s.width)
        idx = 18
        for col in range(vp_start, min(vp_end, vp_start + 20)):
            if idx >= 128:
                break
            # Hash each column into a single byte
            col_hash = 0
            for row in range(min(16, s.GRID_H)):
                col_hash ^= (int(s.grid[row, col]) * (row + 1)) & 0xFF
            ram[idx] = col_hash
            idx += 1

        return ram

    def render(self, viewport: bool = True) -> str:
        """Render current state as ASCII."""
        if self.sim is None:
            return "(no level loaded)"
        return self.sim.render_ascii(viewport=viewport)

    def stats(self) -> dict:
        """Current episode statistics."""
        if self.sim is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "mario_pos": (self.sim.mario_row, self.sim.mario_col),
            "coins": self.sim.coins,
            "score": self.sim.score,
            "alive": self.sim.alive,
            "won": self.sim.won,
            "steps": self._step_count,
            "progress": self.sim.max_x_reached / max(1, self.sim.width),
        }
