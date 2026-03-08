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
# NES RAM ADDRESSES (Super Mario Bros.)
# ═══════════════════════════════════════════════════════════════
# Verified against FCEUX RAM watch / tcrf.net SMB1 RAM map

RAM = {
    # Mario
    "mario_x":        0x006D,  # Mario page (column in level / 16)
    "mario_x_pixel":  0x0086,  # Mario X pixel within the page
    "mario_y_pixel":  0x00CE,  # Mario Y pixel (top of sprite, 0=top)

    # Camera scroll
    "scroll_x":       0x071C,  # Scroll X low byte (level pixel offset)
    "scroll_x_page":  0x071D,  # Scroll X page

    # Enemy slots (5 enemies max on screen)
    "enemy_active":   [0x000F, 0x0010, 0x0011, 0x0012, 0x0013],  # nonzero = active
    "enemy_type":     [0x0016, 0x0017, 0x0018, 0x0019, 0x001A],  # enemy type ID
    "enemy_x_page":   [0x006E, 0x006F, 0x0070, 0x0071, 0x0072],  # enemy X page
    "enemy_x_pixel":  [0x0087, 0x0088, 0x0089, 0x008A, 0x008B],  # enemy X pixel
    "enemy_y_pixel":  [0x00CF, 0x00D0, 0x00D1, 0x00D2, 0x00D3],  # enemy Y pixel

    # Game state
    "lives":          0x075A,  # Lives remaining
    "coins":          0x075E,  # Coins (BCD)
    "world":          0x075F,  # World number (0-indexed)
    "level":          0x0760,  # Level number (0-indexed)
    "player_state":   0x000E,  # 0=alive, 6=dying, 11=dead
    "game_state":     0x0770,  # Global game state
    "flag_get":       0x001D,  # Nonzero when flagpole grabbed
}

# NES enemy type IDs → our EnemyType names
ENEMY_TYPE_MAP = {
    0x00: "goomba",
    0x01: "goomba",
    0x02: "turtle",
    0x03: "turtle",
    0x04: "piranha",
    0x05: "turtle",    # Buzzy Beetle
    0x06: "turtle",    # Hammer Bro
    0x07: "goomba",    # Lakitu
}

# NES tile height and width in pixels
NES_TILE_PX = 16
NES_SCREEN_H = 240
NES_SCREEN_W = 256


class MarioRealAdapter:
    """
    Gymnasium-compatible adapter for real NES Super Mario Bros.

    Uses NES RAM for exact game state (Mario position, enemy positions)
    instead of unreliable pixel-to-color matching. Produces the same
    observation format as MarioAdapter (378-dim vector).
    """

    def __init__(
        self,
        world: str = "SuperMarioBros-v0",
        render_mode: Optional[str] = None,
        frame_skip: int = 4,
    ):
        """
        Args:
            world: gym-super-mario-bros environment ID
            render_mode: 'human' to show the game window, None for headless
            frame_skip: repeat each action N NES frames (default 4 = 15fps)
        """
        _ensure_gym()

        self._env = _gym_mario.make(world, render_mode=render_mode,
                                     apply_api_compatibility=True)
        self._env = _JoypadSpace(self._env, SIMPLE_MOVEMENT)
        self._frame_skip = frame_skip

        self._last_frame = None
        self._last_obs_data = {}   # Cache of last parsed RAM state
        self._step_count = 0
        self.obs_dim = 378
        self.n_actions = 8

    def _ram(self) -> Optional[np.ndarray]:
        """Get NES RAM array from environment."""
        env = self._env
        while hasattr(env, "env"):
            env = env.env
        return getattr(env, "ram", None)

    def reset(self) -> np.ndarray:
        """Reset environment, return initial observation."""
        frame, info = self._env.reset()
        self._last_frame = frame
        self._step_count = 0
        self._last_obs_data = {}
        return self._build_obs(info)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in real game.

        Args:
            action: simulator action index (0-7)

        Returns:
            obs, reward, done, info
        """
        nes_action = ACTION_MAP.get(action, 0)

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
        obs = self._build_obs(info)
        return obs, total_reward, done, info

    def _read_ram_state(self) -> dict:
        """
        Read all relevant game state from NES RAM.

        Returns dict with:
          mario_x_tile, mario_y_tile: Mario's tile position in viewport
          mario_x_pixel, mario_y_pixel: pixel coords
          enemies: list of {type, x_tile, y_tile}
          coins, lives, player_state, flag_get
        """
        ram = self._ram()
        state = {
            "mario_x_tile": 7,   # defaults (Mario center-left of screen)
            "mario_y_tile": 11,
            "mario_x_pixel": 112,
            "mario_y_pixel": 176,
            "enemies": [],
            "coins": 0,
            "lives": 3,
            "player_state": 0,
            "flag_get": False,
            "scroll_x": 0,
        }

        if ram is None:
            return state

        # Mario pixel coords on screen
        # NES draws Mario at pixel coords relative to SCREEN, not level
        mario_y_raw = int(ram[RAM["mario_y_pixel"]])
        mario_x_raw = int(ram[RAM["mario_x_pixel"]])

        # Y: 0=top of screen. Subtract HUD (32px = 2 tiles) and convert to tile
        mario_y_screen = mario_y_raw - 32  # remove HUD
        mario_y_tile = max(0, min(15, mario_y_screen // NES_TILE_PX))

        # X: pixel position on screen (0-255), convert to tile column (0-15)
        mario_x_tile = max(0, min(15, mario_x_raw // NES_TILE_PX))

        state["mario_x_tile"] = mario_x_tile
        state["mario_y_tile"] = mario_y_tile
        state["mario_x_pixel"] = mario_x_raw
        state["mario_y_pixel"] = mario_y_raw

        # Scroll position
        scroll_lo = int(ram[RAM["scroll_x"]])
        scroll_hi = int(ram[RAM["scroll_x_page"]])
        state["scroll_x"] = scroll_hi * 256 + scroll_lo

        # Enemies (5 slots)
        for i in range(5):
            active = int(ram[RAM["enemy_active"][i]])
            if active == 0:
                continue
            etype_raw = int(ram[RAM["enemy_type"][i]])
            etype = ENEMY_TYPE_MAP.get(etype_raw, "goomba")

            ey_raw = int(ram[RAM["enemy_y_pixel"][i]])
            ex_raw = int(ram[RAM["enemy_x_pixel"][i]])

            ey_screen = ey_raw - 32  # remove HUD
            ey_tile = max(0, min(15, ey_screen // NES_TILE_PX))
            ex_tile = max(0, min(15, ex_raw // NES_TILE_PX))

            state["enemies"].append({
                "type": etype,
                "x_tile": ex_tile,
                "y_tile": ey_tile,
            })

        # Game state
        state["coins"] = int(ram[RAM["coins"]])
        state["lives"] = int(ram[RAM["lives"]])
        state["player_state"] = int(ram[RAM["player_state"]])
        state["flag_get"] = int(ram[RAM["flag_get"]]) != 0

        return state

    def _build_obs(self, info: dict = None) -> np.ndarray:
        """
        Build 378-dim observation from NES RAM state.

        Layout matches MarioSimulator.get_obs() exactly:
          [0:320]   viewport grid 16×20 (normalized)
          [320:322] mario row, col (normalized)
          [322:325] on_ground, vy, jump_timer
          [325:327] coins, progress
          [327:375] 8 enemies × 6 (type, row, col, dangerous, shell, dir)
          [375:378] alive, won, step_count
        """
        from src.games.mario.mario_simulator import N_TILE_TYPES, Tile

        state = self._read_ram_state()
        self._last_obs_data = state

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # ── [0:320] Viewport grid ──────────────────────────────────────
        # Build a 16×20 grid. We fill ground at row 14 and sky elsewhere.
        # If we have the info dict, use x_pos to infer tile layout later.
        # For now: ground row and sky, with enemy positions marked.
        grid = np.full((16, 20), Tile.EMPTY, dtype=np.uint8)

        # Ground (rows 13-15 are usually solid)
        grid[13, :] = Tile.GROUND
        grid[14, :] = Tile.GROUND
        grid[15, :] = Tile.GROUND

        # Mark enemies in the grid
        for e in state["enemies"]:
            er, ec = e["y_tile"], e["x_tile"]
            if 0 <= er < 16 and 0 <= ec < 20:
                grid[er, ec] = Tile.ENEMY if hasattr(Tile, "ENEMY") else Tile.PIT

        # Mark Mario
        mr, mc = state["mario_y_tile"], state["mario_x_tile"]
        if 0 <= mr < 16 and 0 <= mc < 20:
            grid[mr, mc] = Tile.PLAYER

        obs[0:320] = grid.flatten().astype(np.float32) / max(N_TILE_TYPES - 1, 1)

        # ── [320:322] Mario position ───────────────────────────────────
        obs[320] = state["mario_y_tile"] / 16.0
        obs[321] = state["mario_x_tile"] / 20.0

        # ── [322:325] Physics ──────────────────────────────────────────
        # on_ground: Mario tile is on or above ground row
        on_ground = 1.0 if state["mario_y_tile"] >= 11 else 0.0
        obs[322] = on_ground
        obs[323] = 0.0   # vy unknown from RAM (would need prev frame diff)
        obs[324] = 0.0   # jump_timer

        # ── [325:327] Coins, progress ──────────────────────────────────
        obs[325] = state["coins"] / 100.0
        # Progress: x_pos from info dict if available
        x_pos = info.get("x_pos", state["scroll_x"]) if info else state["scroll_x"]
        obs[326] = min(x_pos / 3200.0, 1.0)  # W1-1 is ~3200 pixels wide

        # ── [327:375] Enemies 8×6 ──────────────────────────────────────
        type_map = {"goomba": 1.0, "turtle": 2.0, "piranha": 3.0}
        for i in range(8):
            base = 327 + i * 6
            if i < len(state["enemies"]):
                e = state["enemies"][i]
                obs[base + 0] = type_map.get(e["type"], 1.0) / 3.0
                obs[base + 1] = e["y_tile"] / 16.0
                obs[base + 2] = e["x_tile"] / 20.0
                obs[base + 3] = 1.0   # dangerous
                obs[base + 4] = 0.0   # not shell
                obs[base + 5] = 0.5   # direction unknown

        # ── [375:378] Status ───────────────────────────────────────────
        alive = 0.0 if state["player_state"] in (6, 11) else 1.0
        obs[375] = alive
        obs[376] = 1.0 if state["flag_get"] else 0.0
        obs[377] = self._step_count / 1000.0

        return obs

    def render(self) -> str:
        """Render current game state as ASCII showing RAM-derived data."""
        from src.games.mario.mario_simulator import TILE_CHAR, Tile

        state = self._last_obs_data
        if not state:
            return "(no data yet)"

        # Build ASCII grid
        grid = [["." for _ in range(16)] for _ in range(16)]

        # Ground
        for c in range(16):
            grid[13][c] = "#"
            grid[14][c] = "#"
            grid[15][c] = "#"

        # Enemies
        for e in state.get("enemies", []):
            r, c = e["y_tile"], e["x_tile"]
            if 0 <= r < 16 and 0 <= c < 16:
                t = e["type"]
                grid[r][c] = "G" if t == "goomba" else "K" if t == "turtle" else "P"

        # Mario
        mr = state.get("mario_y_tile", 11)
        mc = state.get("mario_x_tile", 7)
        if 0 <= mr < 16 and 0 <= mc < 16:
            grid[mr][mc] = "M"

        lines = ["".join(row) for row in grid]
        enemies = state.get("enemies", [])
        mario_info = f"Mario at tile ({mr},{mc}) pixel ({state.get('mario_y_pixel',0)},{state.get('mario_x_pixel',0)})"
        enemy_info = f"Enemies: {len(enemies)} " + \
                     " ".join(f"{e['type'][0].upper()}@({e['y_tile']},{e['x_tile']})"
                              for e in enemies)
        return "\n".join(lines) + f"\n{mario_info}\n{enemy_info}"

    def stats(self) -> dict:
        """Return current game stats."""
        state = self._last_obs_data
        return {
            "mario_pos": (state.get("mario_y_tile", 0), state.get("mario_x_tile", 0)),
            "enemies": state.get("enemies", []),
            "step": self._step_count,
        }

    def close(self):
        """Close the environment."""
        self._env.close()
