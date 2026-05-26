"""
atari_adapter.py — ASCII adapter for Atari games via Gymnasium ALE.

Follows the same pattern as MuJoCoAdapter:
    - Graceful fallback when gymnasium[atari] is not installed
    - Compatible with UniversalEncoder / MultiGameWorldModel pipeline
    - Uses AsciiEncoder to convert Atari frames (210×160×3) → density grids

Supported via gymnasium: any ROM accessible via `gym.make("ALE/Pong-v5")`.

ASCII Encoding:
    - Full frame (15×20 grid): 300 values
    - Colour-filtered: removes background, keeps moving objects
    - Compatible obs_dim: 300 (flat RAM-like vector)

Usage:
    adapter = AtariAdapter("ALE/Pong-v5")          # real rom
    adapter = AtariAdapter("AtariFallback/Pong")   # fallback

    obs = adapter.reset()          # (300,) float32
    obs2, rew, done, info = adapter.step(action_idx)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

import numpy as np

_ALE_AVAILABLE = True
try:
    import gymnasium as gym
    gym.make("ALE/Pong-v5")     # probe
except Exception:
    _ALE_AVAILABLE = False

try:
    from src.encoder.ascii_encoder import AsciiEncoder
except ImportError:
    from encoder.ascii_encoder import AsciiEncoder  # type: ignore


# ═══════════════════════════════════════════════════════════════
# SUPPORTED ATARI GAME SPECS
# ═══════════════════════════════════════════════════════════════

ATARI_SPECS: Dict[str, Dict] = {
    "Pong": {
        "ale_name": "ALE/Pong-v5",
        "n_actions": 6,
        "ram_obs_dim": 128,
        "description": "Pong — 2-paddle ball sport",
    },
    "Breakout": {
        "ale_name": "ALE/Breakout-v5",
        "n_actions": 4,
        "ram_obs_dim": 128,
        "description": "Breakout — brick-breaking",
    },
    "SpaceInvaders": {
        "ale_name": "ALE/SpaceInvaders-v5",
        "n_actions": 6,
        "ram_obs_dim": 128,
        "description": "Space Invaders — vertical shooter",
    },
    "MontezumaRevenge": {
        "ale_name": "ALE/MontezumaRevenge-v5",
        "n_actions": 18,
        "ram_obs_dim": 128,
        "description": "Montezuma's Revenge — hard exploration",
    },
    "Qbert": {
        "ale_name": "ALE/Qbert-v5",
        "n_actions": 6,
        "ram_obs_dim": 128,
        "description": "Qbert — isometric platformer",
    },
}

GRID_ROWS = 15
GRID_COLS = 20
GRID_DIM = GRID_ROWS * GRID_COLS  # 300


# ═══════════════════════════════════════════════════════════════
# ATARI ADAPTER
# ═══════════════════════════════════════════════════════════════

class AtariAdapter:
    """
    ASCII adapter for Atari environments.

    When ALE is available: renders real Atari frames → 15×20 ASCII grid (300-dim).
    When ALE is not available: uses AtariFallbackSim (simple RAM-based simulation).
    """

    def __init__(
        self,
        game_name: str = "Pong",
        grid_rows: int = GRID_ROWS,
        grid_cols: int = GRID_COLS,
        seed: int = 42,
    ):
        self.game_name = game_name
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.seed = seed

        spec = ATARI_SPECS.get(game_name, ATARI_SPECS["Pong"])
        self._spec = spec
        self.n_actions = spec["n_actions"]
        self._obs_dim = grid_rows * grid_cols

        self._enc = AsciiEncoder(rows=grid_rows, cols=grid_cols, color_channels=False)
        self._env = None
        self._episode = 0
        self._total_reward = 0.0

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def _make_env(self):
        if not _ALE_AVAILABLE:
            raise RuntimeError(
                "gymnasium[atari] not installed. "
                "Install with: pip install gymnasium[atari] ale-py"
            )
        self._env = gym.make(
            self._spec["ale_name"],
            render_mode="rgb_array",
            obs_type="rgb",
        )

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if self._env is None:
            self._make_env()
        seed_val = seed if seed is not None else self.seed + self._episode
        obs, _ = self._env.reset(seed=seed_val)
        self._episode += 1
        self._total_reward = 0.0
        return self._frame_to_obs(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._env is None:
            raise RuntimeError("Call reset() first")
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._total_reward += float(reward)
        done = terminated or truncated
        return self._frame_to_obs(obs), float(reward), done, info

    def _frame_to_obs(self, frame: np.ndarray) -> np.ndarray:
        """Convert (H,W,3) uint8 frame to flattened (grid_dim,) float32."""
        if frame is None:
            return np.zeros(self._obs_dim, dtype=np.float32)
        grid = self._enc.encode(frame)
        return grid.flatten().astype(np.float32) / 9.0

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def stats(self) -> Dict[str, Any]:
        return {
            "game": self.game_name,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "total_reward": round(self._total_reward, 3),
        }

    def __repr__(self) -> str:
        return (
            f"AtariAdapter({self.game_name!r}, "
            f"obs_dim={self.obs_dim}, n_actions={self.n_actions})"
        )


# ═══════════════════════════════════════════════════════════════
# FALLBACK SIMULATOR — no ALE install needed
# ═══════════════════════════════════════════════════════════════

class AtariFallbackSim:
    """
    Pure numpy Pong-like simulation for testing without ALE.

    State: (ball_x, ball_y, ball_vx, ball_vy, paddle_y, opp_y)  → obs_dim = 6
    Action: 0=noop, 1=fire, 2=up, 3=down, 4=upfire, 5=downfire
    """

    def __init__(
        self,
        game_name: str = "Pong",
        obs_dim: int = GRID_DIM,   # matches AtariAdapter obs_dim
        max_steps: int = 500,
        seed: int = 42,
    ):
        self.game_name = f"Fallback/{game_name}"
        self._obs_dim = obs_dim
        self.max_steps = max_steps
        self.n_actions = ATARI_SPECS.get(game_name, ATARI_SPECS["Pong"])["n_actions"]
        self._rng = np.random.RandomState(seed)
        self._step_count = 0
        self._total_reward = 0.0
        self._episode = 0

        # Simple pong physics state
        self._state = np.zeros(6, dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._state = self._rng.uniform(-1, 1, 6).astype(np.float32)
        self._step_count = 0
        self._total_reward = 0.0
        self._episode += 1
        return self._state_to_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Simple physics: ball moves, paddle adjusts to action
        self._state[0] += self._state[2] * 0.1    # ball_x
        self._state[1] += self._state[3] * 0.1    # ball_y

        # Paddle moves with action
        if action in (2, 4):
            self._state[4] = min(1.0, self._state[4] + 0.1)
        elif action in (3, 5):
            self._state[4] = max(-1.0, self._state[4] - 0.1)

        # Bounce off walls
        for i in [0, 1]:
            if abs(self._state[i]) > 1.0:
                self._state[i + 2] *= -1

        # Score reward
        reward = 0.0
        if abs(self._state[0]) > 0.95:  # ball near paddle
            diff = abs(self._state[1] - self._state[4])
            reward = 1.0 if diff < 0.2 else -1.0
            # Reset ball
            self._state[:4] = self._rng.uniform(-0.5, 0.5, 4)

        self._step_count += 1
        self._total_reward += reward
        done = self._step_count >= self.max_steps
        return self._state_to_obs(), reward, done, {}

    def _state_to_obs(self) -> np.ndarray:
        """
        Render physics state as a GRID_ROWS × GRID_COLS ASCII-style density grid.

        Maps ball (x,y) and paddle (y) onto a binary occupancy grid so the
        agent can distinguish game states — unlike zero-padding which makes
        all states look identical to a world model.

        Grid layout (15 rows × 20 cols):
            Top half (rows 0-6): ball position (binary)
            Bottom half (rows 8-14): paddle position (binary)
            Row 7: separator (always 0)
        """
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # Render into a (GRID_ROWS, GRID_COLS) grid
        rows, cols = 7, 20   # half-grid for ball
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

        # Ball position: state[0]=x in [-1,1], state[1]=y in [-1,1]
        ball_col = int((self._state[0] + 1.0) / 2.0 * (cols - 1))
        ball_row = int((self._state[1] + 1.0) / 2.0 * (rows - 1))
        ball_col = np.clip(ball_col, 0, cols - 1)
        ball_row = np.clip(ball_row, 0, rows - 1)
        grid[ball_row, ball_col] = 1.0

        # Paddle: state[4]=y in [-1,1] → bottom half of grid
        paddle_row = GRID_ROWS - 1 - int((self._state[4] + 1.0) / 2.0 * (rows - 1))
        paddle_row = np.clip(paddle_row, GRID_ROWS - rows, GRID_ROWS - 1)
        # Paddle spans 3 columns centred at col 18 (right side)
        for dc in [-1, 0, 1]:
            pc = np.clip(18 + dc, 0, cols - 1)
            grid[paddle_row, pc] = 0.8

        # Flatten grid into obs, pad remainder with zeros
        flat = grid.flatten()
        obs[:len(flat)] = flat
        return obs

    def close(self):
        pass

    def stats(self) -> Dict[str, Any]:
        return {
            "game": self.game_name,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "total_reward": round(self._total_reward, 3),
        }


def make_atari_adapter(
    game_name: str = "Pong",
    seed: int = 42,
) -> "AtariAdapter | AtariFallbackSim":
    """
    Factory: returns real AtariAdapter if ALE is available, else AtariFallbackSim.
    """
    if not _ALE_AVAILABLE:
        return AtariFallbackSim(game_name=game_name, seed=seed)
    return AtariAdapter(game_name=game_name, seed=seed)
