"""
mario_real_adapter.py -- Adapter between gym-super-mario-bros and our ASCII agent.

Wraps the real NES Mario game (via gym-super-mario-bros) so our agent
trained on ASCII simulators can play the actual game. The conversion
pipeline:

  Real NES frame (240x256 RGB pixels)
    ↓  image_to_ascii()
  Tile grid (14x16)
    ↓  flatten + normalize
  Observation vector (same format as MarioAdapter)
    ↓
  Agent picks action (0-5)
    ↓  ACTION_MAP
  NES joypad buttons → env.step()

Requirements (install on cloud):
  pip install gym-super-mario-bros nes-py

Usage:
  from mario_real_adapter import MarioRealAdapter

  adapter = MarioRealAdapter()
  obs = adapter.reset()
  action = agent.step(obs)
  obs, reward, done, info = adapter.step(action)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Lazy imports for gym-super-mario-bros (may not be installed locally)
_gym_loaded = False
_gym_mario = None
_JoypadSpace = None


def _ensure_gym():
    """Lazy-load gym-super-mario-bros."""
    global _gym_loaded, _gym_mario, _JoypadSpace
    if _gym_loaded:
        return
    try:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        _gym_mario = gym_super_mario_bros
        _JoypadSpace = JoypadSpace
        _gym_loaded = True
    except ImportError:
        raise ImportError(
            "gym-super-mario-bros not installed. "
            "Run: pip install gym-super-mario-bros"
        )


# ═══════════════════════════════════════════════════════════════
# ACTION MAPPING
# ═══════════════════════════════════════════════════════════════

# Our simulator uses 6 actions (matching MarioAdapter):
#   0: NOOP, 1: RIGHT, 2: RIGHT+JUMP, 3: LEFT, 4: JUMP, 5: RUN_RIGHT
#
# gym-super-mario-bros uses NES joypad buttons:
#   RIGHT_ONLY:  [NOOP, right]
#   SIMPLE:      [NOOP, right, right+A, right+B, right+A+B, A, left]
#   COMPLEX:     [all 256 combinations]
#
# We use SIMPLE_MOVEMENT and map our 6 actions to the 7 SIMPLE actions.

# Our action → SIMPLE_MOVEMENT index
ACTION_MAP = {
    0: 0,   # NOOP  → NOOP
    1: 1,   # RIGHT → right
    2: 2,   # RIGHT+JUMP → right + A (jump)
    3: 6,   # LEFT  → left
    4: 5,   # JUMP  → A
    5: 3,   # RUN_RIGHT → right + B (run)
}

SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]


# ═══════════════════════════════════════════════════════════════
# REAL GAME ADAPTER
# ═══════════════════════════════════════════════════════════════

class MarioRealAdapter:
    """
    Gymnasium-compatible adapter for real NES Super Mario Bros.

    Converts pixel frames to ASCII tile grids, producing the same
    observation format as our MarioAdapter. Agents trained on ASCII
    simulators can play the real game seamlessly.
    """

    HUD_ROWS = 2         # Top 2 tile rows are HUD (score/time)
    GAME_ROWS = 14        # 16 - 2 HUD = 14 playable rows
    VIEWPORT_COLS = 16    # Visible tile columns

    def __init__(
        self,
        world: str = "SuperMarioBros-v0",
        render_mode: Optional[str] = None,
        frame_skip: int = 4,
    ):
        """
        Args:
            world: gym-super-mario-bros environment ID
            render_mode: 'human' to see the game window, None for headless
            frame_skip: repeat each action for N frames (NES runs at 60fps)
        """
        _ensure_gym()

        self._env = _gym_mario.make(world, render_mode=render_mode,
                                     apply_api_compatibility=True)
        self._env = _JoypadSpace(self._env, SIMPLE_MOVEMENT)
        self._frame_skip = frame_skip

        # Import our converter (relative import won't work, use absolute)
        from src.games.mario.screen_to_ascii import image_to_ascii

        self._image_to_ascii = image_to_ascii
        self._last_frame = None
        self._last_grid = None
        self._last_entities = None
        self._step_count = 0

        # Observation dimensions (match MarioAdapter)
        # grid (14*16=224) + mario_pos(2) + velocity_placeholder(2) +
        # enemy_slots(5*3=15) + coins(1) + score(1) + alive(1) = 246
        # Pad to 378 to match MarioAdapter exactly
        self.obs_dim = 378
        self.n_actions = 6

    def reset(self) -> np.ndarray:
        """Reset environment, return initial observation."""
        frame, info = self._env.reset()
        self._last_frame = frame
        self._step_count = 0
        return self._frame_to_obs(frame)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in real game.

        Args:
            action: our action index (0-5)

        Returns:
            obs: observation vector (378,)
            reward: game reward
            done: episode over
            info: game info dict
        """
        # Map our action to NES joypad
        nes_action = ACTION_MAP.get(action, 0)

        # Frame skip: repeat action N times
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self._frame_skip):
            frame, reward, terminated, truncated, info = self._env.step(nes_action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        self._last_frame = frame
        self._step_count += 1

        obs = self._frame_to_obs(frame)

        return obs, total_reward, done, info

    def _frame_to_obs(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert NES pixel frame to our observation vector.

        Frame: (240, 256, 3) RGB → crop HUD → image_to_ascii → flatten
        """
        h, w = frame.shape[:2]

        # Crop HUD (top ~32 pixels for 240-height, scale proportionally)
        hud_px = int(h * 32 / 240)
        game_frame = frame[hud_px:]

        # Convert to ASCII grid
        grid, entities = self._image_to_ascii(
            game_frame,
            grid_rows=self.GAME_ROWS,
            grid_cols=self.VIEWPORT_COLS,
        )

        self._last_grid = grid
        self._last_entities = entities

        # Build observation vector (matching MarioAdapter format)
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # Grid tiles (normalized to 0-1)
        grid_flat = grid.flatten().astype(np.float32) / 10.0
        obs[:len(grid_flat)] = grid_flat

        # Mario position
        idx = len(grid_flat)
        if entities["mario_pos"]:
            obs[idx] = entities["mario_pos"][0] / self.GAME_ROWS
            obs[idx + 1] = entities["mario_pos"][1] / self.VIEWPORT_COLS
        idx += 2

        # Enemy positions (up to 5 enemies, 3 values each: type/row/col)
        for i, enemy in enumerate(entities.get("enemies", [])[:5]):
            base = idx + i * 3
            type_map = {"goomba": 0.2, "turtle": 0.5, "piranha": 0.8}
            obs[base] = type_map.get(enemy["type"], 0.1)
            obs[base + 1] = enemy["row"] / self.GAME_ROWS
            obs[base + 2] = enemy["col"] / self.VIEWPORT_COLS
        idx += 15

        # Coins, score, alive flag from info (if available)
        obs[idx] = 1.0  # alive

        return obs

    def render(self) -> str:
        """Render current game state as ASCII string."""
        if self._last_grid is None:
            return "No frame captured yet"

        from src.games.mario.mario_simulator import TILE_CHAR

        lines = []
        for row in range(self._last_grid.shape[0]):
            line = ""
            for col in range(self._last_grid.shape[1]):
                tile = self._last_grid[row, col]
                line += TILE_CHAR.get(tile, "?")
            lines.append(line)
        return "\n".join(lines)

    def stats(self) -> dict:
        """Return current game stats."""
        return {
            "mario_pos": self._last_entities.get("mario_pos", (0, 0))
                         if self._last_entities else (0, 0),
            "enemies": self._last_entities.get("enemies", [])
                       if self._last_entities else [],
            "step": self._step_count,
        }

    def close(self):
        """Close the environment."""
        self._env.close()
