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

# Our simulator uses 8 actions:
#   0: NOOP, 1: LEFT, 2: RIGHT, 3: JUMP, 4: JUMP_LEFT, 5: JUMP_RIGHT
#   6: RUN_RIGHT, 7: RUN_JUMP
#
# gym-super-mario-bros SIMPLE_MOVEMENT:
#   0: NOOP, 1: right, 2: right+A, 3: right+B, 4: right+A+B, 5: A, 6: left
#
# We map our 8 actions to the 7 SIMPLE actions:

# Our action → SIMPLE_MOVEMENT index
ACTION_MAP = {
    0: 0,   # NOOP       → NOOP
    1: 6,   # LEFT       → left
    2: 1,   # RIGHT      → right
    3: 5,   # JUMP       → A
    4: 6,   # JUMP_LEFT  → left (no left+jump in SIMPLE)
    5: 2,   # JUMP_RIGHT → right + A (jump right)
    6: 3,   # RUN_RIGHT  → right + B (run)
    7: 4,   # RUN_JUMP   → right + A + B (run + jump)
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
        Convert NES pixel frame to observation matching simulator's get_obs().

        Simulator layout (378 dims):
          [0:320]   viewport grid 16×20, normalized by N_TILE_TYPES
          [320:322] mario_vp_row, mario_vp_col
          [322:325] on_ground, vy, jump_timer
          [325:327] coins, progress
          [327:375] 8 enemies × 6 (etype, row, col, dangerous, shell, dir)
          [375:378] alive, won, step_count
        """
        from src.games.mario.mario_simulator import N_TILE_TYPES

        h, w = frame.shape[:2]

        # Crop HUD (top ~32 pixels)
        hud_px = int(h * 32 / 240)
        game_frame = frame[hud_px:]

        # Convert to ASCII grid (14×16)
        grid, entities = self._image_to_ascii(
            game_frame,
            grid_rows=self.GAME_ROWS,
            grid_cols=self.VIEWPORT_COLS,
        )

        self._last_grid = grid
        self._last_entities = entities

        # === Build 378-dim observation matching simulator exactly ===
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # [0:320] Viewport grid: resize 14×16 → 16×20, normalized
        viewport = np.zeros((16, 20), dtype=np.float32)
        # Place the 14×16 game grid centered in the 16×20 viewport
        r_off = 1  # offset 1 row down (HUD gap)
        c_off = 2  # offset 2 cols right (center)
        gr, gc = grid.shape
        viewport[r_off:r_off+min(gr,15), c_off:c_off+min(gc,18)] = \
            grid[:min(gr,15), :min(gc,18)].astype(np.float32)
        viewport /= max(N_TILE_TYPES - 1, 1)
        obs[0:320] = viewport.flatten()

        # [320:322] Mario position relative to viewport
        mario_pos = entities.get("mario_pos")
        if mario_pos:
            obs[320] = (mario_pos[0] + r_off) / 16.0   # row
            obs[321] = (mario_pos[1] + c_off) / 20.0   # col
        else:
            obs[320] = 0.75  # default: near ground
            obs[321] = 0.5   # default: center

        # [322:325] Physics (estimated from frame changes)
        obs[322] = 1.0   # on_ground (assume grounded)
        obs[323] = 0.0   # vy (unknown from single frame)
        obs[324] = 0.0   # jump_timer

        # [325:327] Coins, progress
        obs[325] = 0.0   # coins (could extract from HUD)
        obs[326] = self._step_count / 2000.0  # rough progress proxy

        # [327:375] Enemies (8 × 6 values each)
        enemies = entities.get("enemies", [])[:8]
        for i in range(8):
            base = 327 + i * 6
            if i < len(enemies):
                e = enemies[i]
                type_map = {"goomba": 1.0, "turtle": 2.0, "piranha": 3.0}
                obs[base + 0] = type_map.get(e.get("type", ""), 1.0) / 3.0
                obs[base + 1] = (e.get("row", 0) + r_off) / 16.0
                obs[base + 2] = (e.get("col", 0) + c_off) / 20.0
                obs[base + 3] = 1.0  # dangerous
                obs[base + 4] = 0.0  # not shell
                obs[base + 5] = -1.0  # moving left

        # [375:378] Status
        obs[375] = 1.0  # alive
        obs[376] = 0.0  # not won
        obs[377] = self._step_count / 1000.0

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
