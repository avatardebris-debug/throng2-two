"""
mario_simulator.py — Pure Python Mario side-scroller engine.

Simulates core Super Mario Bros platformer mechanics without any emulator.
Runs ~100,000x faster than real NES, enabling massive GAN-based training.

Grid: 16 rows × W columns (W = n_screens × 20)
Actions: NOOP(0), LEFT(1), RIGHT(2), JUMP(3), JUMP_LEFT(4), JUMP_RIGHT(5)
Win: Reach the flagpole (rightmost column with FLAG tile)
Death: Fall in pit, touch enemy from non-stomp angle

Physics (tile-level):
  - Gravity: Mario falls 1 tile/step when airborne
  - Jump: rises 1 tile/step for 4 steps, then falls
  - Horizontal: moves 1 tile/step

Usage:
    from src.games.mario.mario_simulator import MarioSimulator, Tile, Action
    sim = MarioSimulator.from_flat_ground(n_screens=1)
    obs, r, done, info = sim.step(Action.RIGHT)
    print(sim.render_ascii())
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# TILE TYPES
# ═══════════════════════════════════════════════════════════════════════

class Tile(IntEnum):
    EMPTY     = 0   # Air / sky
    GROUND    = 1   # Solid ground
    BRICK     = 2   # Breakable brick
    QUESTION  = 3   # ? block (coin / powerup)
    PIPE_L    = 4   # Pipe left
    PIPE_R    = 5   # Pipe right
    COIN      = 6   # Collectible coin
    PIT       = 7   # Deadly pit (no floor)
    PLATFORM  = 8   # Floating platform
    FLAG      = 9   # Level end / flagpole
    PLAYER    = 10  # Mario start position (converted to EMPTY)
    # Enemies are tracked separately, not in the tile grid

N_TILE_TYPES = 11

# Display characters for ASCII rendering
TILE_CHAR = {
    Tile.EMPTY:    '.',
    Tile.GROUND:   '#',
    Tile.BRICK:    'B',
    Tile.QUESTION: '?',
    Tile.PIPE_L:   '[',
    Tile.PIPE_R:   ']',
    Tile.COIN:     'o',
    Tile.PIT:      '_',
    Tile.PLATFORM: '=',
    Tile.FLAG:     'F',
    Tile.PLAYER:   'M',
}


# ═══════════════════════════════════════════════════════════════════════
# ACTIONS
# ═══════════════════════════════════════════════════════════════════════

class Action(IntEnum):
    NOOP       = 0
    LEFT       = 1
    RIGHT      = 2
    JUMP       = 3
    JUMP_LEFT  = 4
    JUMP_RIGHT = 5
    RUN_RIGHT  = 6  # Right + B (run speed)
    RUN_JUMP   = 7  # Right + B + A (run speed + jump)

N_ACTIONS = 8


# ═══════════════════════════════════════════════════════════════════════
# ENEMIES
# ═══════════════════════════════════════════════════════════════════════

class EnemyType(IntEnum):
    GOOMBA   = 0   # Walks left, dies on stomp
    TURTLE   = 1   # Walks left, becomes shell on stomp
    PIRANHA  = 2   # Bobs up/down from pipe, can't be stomped
    LAKITU   = 3   # Flies in sky row, drops spinies


@dataclass
class Enemy:
    """A Mario enemy with type, position, and state."""
    etype: EnemyType
    row: int
    col: int
    alive: bool = True
    direction: int = -1  # -1 = moving left, +1 = moving right
    is_shell: bool = False
    shell_moving: bool = False
    shell_dir: int = 1
    bob_timer: int = 0   # Piranha: bob cycle counter
    bob_up: bool = True   # Piranha: currently above pipe?

    @property
    def is_dangerous(self) -> bool:
        if not self.alive:
            return False
        if self.is_shell and not self.shell_moving:
            return False  # Stationary shell is safe
        return True


# ═══════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

SOLID_TILES = {Tile.GROUND, Tile.BRICK, Tile.QUESTION, Tile.PIPE_L,
               Tile.PIPE_R, Tile.PLATFORM}

JUMP_DURATION = 4     # Steps Mario rises during a jump
GRAVITY = 1           # Tiles Mario falls per step when airborne
MAX_FALL_SPEED = 2    # Terminal velocity in tiles/step


# ═══════════════════════════════════════════════════════════════════════
# SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

class MarioSimulator:
    """
    Pure Python Super Mario Bros side-scroller engine.

    Operates on a tile grid. No pixels — just tile-level physics.
    Supports save/load, ASCII rendering, and completability checking.
    """

    GRID_H = 16  # rows (top = 0 = sky, bottom = 15 = ground level)
    GROUND_ROW = 13  # Default ground level (rows 13-15 are ground)

    # ── Reward shaping ─────────────────────────────────────────
    # Following Mario RL best practices (OpenAI, Go-Explore, MarioAI)
    R_STEP       = -0.01     # Small cost of living
    R_COIN       = 0.5       # Coin pickup
    R_BRICK_BREAK = 0.1      # Breaking bricks
    R_QUESTION_HIT = 0.3     # Hitting ? block
    R_ENEMY_STOMP = 1.0      # Stomping enemy
    R_PROGRESS   = 0.1       # Per NEW rightmost column reached
    R_VELOCITY   = 0.05      # Per rightward tile moved (even if not new max)
    R_LEFTWARD   = -0.05     # Penalty for moving left (discourages backtracking)
    R_STALL      = -0.10     # Penalty when standing still too long
    R_WIN        = 10.0      # Level clear base reward
    R_DEATH      = -1.0      # Death penalty (moderate — don't prefer paralysis)
    STALL_THRESHOLD = 5      # Steps without rightward progress before stall penalty
    TIME_PRESSURE_RATE = 0.005  # Accelerating time cost: -rate * (step/200)

    def __init__(
        self,
        grid: np.ndarray,
        enemies: Optional[List[Enemy]] = None,
    ):
        """
        Args:
            grid: (GRID_H, width) uint8 array of Tile values
            enemies: List of Enemy objects
        """
        assert grid.shape[0] == self.GRID_H, \
            f"Grid height must be {self.GRID_H}, got {grid.shape[0]}"

        self.grid = grid.astype(np.uint8).copy()
        self.width = grid.shape[1]
        self.enemies = deepcopy(enemies) if enemies else []

        # Find and remove player marker from grid
        player_pos = np.argwhere(self.grid == Tile.PLAYER)
        if len(player_pos) > 0:
            self.mario_row = int(player_pos[0][0])
            self.mario_col = int(player_pos[0][1])
            self.grid[self.mario_row, self.mario_col] = Tile.EMPTY
        else:
            # Default: stand on ground, column 2
            self.mario_row = self.GROUND_ROW - 1
            self.mario_col = 2

        # Physics state
        self.vy = 0              # Vertical velocity: <0 = rising, >0 = falling
        self.jump_timer = 0      # Remaining rise steps
        self.on_ground = True    # Is Mario on solid surface?

        # Game state
        self.alive = True
        self.won = False
        self.coins = 0
        self.score = 0
        self.step_count = 0
        self.max_x_reached = self.mario_col  # Rightmost position ever reached
        self._stall_count = 0                # Steps since last rightward progress
        self._prev_col = self.mario_col       # For velocity reward

        # Viewport (for multi-screen scrolling)
        self.scroll_x = 0

    # ── Factory methods ────────────────────────────────────────────

    @classmethod
    def from_flat_ground(cls, n_screens: int = 1, width: int = None) -> 'MarioSimulator':
        """Create a simple flat ground level for testing."""
        w = width or n_screens * 20
        grid = np.full((cls.GRID_H, w), Tile.EMPTY, dtype=np.uint8)
        # Ground rows 13-15
        grid[13:, :] = Tile.GROUND
        # Player start
        grid[12, 2] = Tile.PLAYER
        # Flag at end
        grid[12, w - 2] = Tile.FLAG
        grid[11, w - 2] = Tile.FLAG
        grid[10, w - 2] = Tile.FLAG
        return cls(grid)

    # ── Main Step ─────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one game step.

        Returns:
            obs: flat observation vector
            reward: scalar reward
            done: episode over?
            info: diagnostic dict
        """
        if not self.alive or self.won:
            return self.get_obs(), 0.0, True, {"reason": "already_done"}

        self.step_count += 1
        reward = self.R_STEP
        # Accelerating time pressure: gets worse the longer you take
        reward -= self.TIME_PRESSURE_RATE * (self.step_count / 200.0)
        info: Dict[str, Any] = {}
        action = Action(action)
        col_before = self.mario_col

        # ── 1. Compute desired movement ────────────────────────────
        dx = 0
        want_jump = False
        is_running = False

        if action == Action.LEFT:
            dx = -1
        elif action == Action.RIGHT:
            dx = 1
        elif action == Action.JUMP:
            want_jump = True
        elif action == Action.JUMP_LEFT:
            dx = -1
            want_jump = True
        elif action == Action.JUMP_RIGHT:
            dx = 1
            want_jump = True
        elif action == Action.RUN_RIGHT:
            dx = 2  # Run speed = 2 tiles/step
            is_running = True
        elif action == Action.RUN_JUMP:
            dx = 2  # Run speed
            want_jump = True
            is_running = True

        # ── 2. Initiate jump ──────────────────────────────────────
        if want_jump and self.on_ground:
            # Running jump lasts longer (farther arc)
            self.jump_timer = JUMP_DURATION + (1 if is_running else 0)
            self.vy = -1  # Rising
            self.on_ground = False

        # ── 3. Vertical movement ─────────────────────────────────
        if self.jump_timer > 0:
            # Rising phase
            self.jump_timer -= 1
            new_row = self.mario_row - 1  # Move up

            if new_row >= 0 and not self._is_solid(new_row, self.mario_col):
                # Check for head bump (hitting block from below)
                if new_row >= 0 and self._is_solid(new_row, self.mario_col):
                    self.jump_timer = 0  # Cancel jump
                    self.vy = 1  # Start falling
                else:
                    self.mario_row = new_row
                    # Check head-bump on block above
                    head_row = self.mario_row - 1
                    if head_row >= 0:
                        head_tile = self.grid[head_row, self.mario_col]
                        if head_tile == Tile.BRICK:
                            self.grid[head_row, self.mario_col] = Tile.EMPTY
                            reward += self.R_BRICK_BREAK
                            info["brick_break"] = True
                        elif head_tile == Tile.QUESTION:
                            self.grid[head_row, self.mario_col] = Tile.BRICK  # Used ? block
                            self.coins += 1
                            reward += self.R_QUESTION_HIT
                            info["question_hit"] = True
            else:
                # Blocked above — cancel jump
                self.jump_timer = 0
                self.vy = 1
        elif not self.on_ground:
            # Falling phase
            fall_speed = min(GRAVITY, MAX_FALL_SPEED)
            new_row = self.mario_row + fall_speed

            if new_row < self.GRID_H and not self._is_solid(new_row, self.mario_col):
                self.mario_row = new_row
            elif new_row >= self.GRID_H:
                # Fell off the bottom — check for pit
                self.alive = False
                reward += self.R_DEATH
                info["death"] = "fell_off_screen"
            else:
                # Landed on solid ground
                self.on_ground = True
                self.vy = 0

        # ── 4. Check if still on ground ──────────────────────────
        if self.on_ground and self.jump_timer == 0:
            below = self.mario_row + 1
            if below >= self.GRID_H:
                # Off the bottom edge
                self.alive = False
                reward += self.R_DEATH
                info["death"] = "fell_off_screen"
            elif not self._is_solid(below, self.mario_col):
                # Ground disappeared — start falling
                self.on_ground = False
                self.vy = 1

        # ── 5. Horizontal movement ───────────────────────────────
        if dx != 0 and self.alive:
            new_col = self.mario_col + dx
            if 0 <= new_col < self.width:
                if not self._is_solid(self.mario_row, new_col):
                    self.mario_col = new_col

                    # Collect coin
                    if self.grid[self.mario_row, self.mario_col] == Tile.COIN:
                        self.grid[self.mario_row, self.mario_col] = Tile.EMPTY
                        self.coins += 1
                        reward += self.R_COIN
                        info["coin"] = True

                    # Check for flag (win)
                    if self.grid[self.mario_row, self.mario_col] == Tile.FLAG:
                        self.won = True
                        reward += self.R_WIN
                        info["won"] = True

                    # Check for pit tile
                    if self.grid[self.mario_row, self.mario_col] == Tile.PIT:
                        self.alive = False
                        reward += self.R_DEATH
                        info["death"] = "pit"

                    # Progress reward
                    if self.mario_col > self.max_x_reached:
                        progress = self.mario_col - self.max_x_reached
                        reward += self.R_PROGRESS * progress
                        self.max_x_reached = self.mario_col
                        info["progress"] = progress

        # ── 6. Collect coins at current position ─────────────────
        if self.alive and self.grid[self.mario_row, self.mario_col] == Tile.COIN:
            self.grid[self.mario_row, self.mario_col] = Tile.EMPTY
            self.coins += 1
            reward += self.R_COIN

        # Check flag at current position (could land on it)
        if self.alive and self.grid[self.mario_row, self.mario_col] == Tile.FLAG:
            self.won = True
            reward += self.R_WIN
            info["won"] = True

        # ── 7. Update enemies ────────────────────────────────────
        if self.alive:
            enemy_info = self._update_enemies()
            if enemy_info.get("mario_died"):
                self.alive = False
                reward += self.R_DEATH
                info["death"] = enemy_info.get("death_reason", "enemy")
            if enemy_info.get("stomps", 0) > 0:
                reward += self.R_ENEMY_STOMP * enemy_info["stomps"]
                info["stomps"] = enemy_info["stomps"]

        # ── 8. Velocity + Stall rewards ────────────────────────
        if self.alive:
            dx_actual = self.mario_col - col_before
            if dx_actual > 0:
                reward += self.R_VELOCITY * dx_actual
                self._stall_count = 0
            elif dx_actual < 0:
                reward += self.R_LEFTWARD * abs(dx_actual)
                self._stall_count += 1
            else:
                self._stall_count += 1

            if self._stall_count >= self.STALL_THRESHOLD:
                reward += self.R_STALL
                info["stalled"] = self._stall_count

            self._prev_col = self.mario_col

        # ── 9. Speed bonus on clear ──────────────────────────────
        if self.won:
            max_steps = self.width * 4  # generous time budget
            speed_bonus = max(0.0, 1.0 - self.step_count / max(max_steps, 1))
            reward += self.R_WIN * speed_bonus * 0.5  # up to +5.0 extra
            info["speed_bonus"] = round(speed_bonus, 3)

        # ── 10. Update scroll ────────────────────────────────────
        self.scroll_x = max(0, min(self.mario_col - 10,
                                    self.width - 20))

        self.score += max(0, reward)
        done = not self.alive or self.won
        return self.get_obs(), float(reward), done, info

    # ── Enemy Logic ───────────────────────────────────────────────

    def _update_enemies(self) -> Dict[str, Any]:
        """Move enemies and check collisions with Mario."""
        result: Dict[str, Any] = {"mario_died": False, "stomps": 0}

        for e in self.enemies:
            if not e.alive:
                continue

            # ── Move enemy ────────────────────────────────────────
            if e.etype == EnemyType.GOOMBA:
                self._ai_walk(e)
            elif e.etype == EnemyType.TURTLE:
                if e.is_shell and e.shell_moving:
                    self._ai_shell(e)
                elif not e.is_shell:
                    self._ai_walk(e)
            elif e.etype == EnemyType.PIRANHA:
                self._ai_piranha(e)
            elif e.etype == EnemyType.LAKITU:
                self._ai_lakitu(e)

            # ── Collision with Mario ──────────────────────────────
            if not e.alive or not e.is_dangerous:
                continue

            if e.row == self.mario_row and e.col == self.mario_col:
                # Same cell — kill Mario (side contact)
                result["mario_died"] = True
                result["death_reason"] = f"touched_{e.etype.name if hasattr(e.etype, 'name') else e.etype}"

            elif (e.row == self.mario_row + 1 and e.col == self.mario_col
                  and self.vy >= 0 and not self.on_ground):
                # Mario is above enemy and falling — STOMP
                if e.etype == EnemyType.PIRANHA:
                    # Can't stomp piranhas
                    result["mario_died"] = True
                    result["death_reason"] = "touched_PIRANHA"
                elif e.etype == EnemyType.TURTLE and not e.is_shell:
                    e.is_shell = True
                    e.shell_moving = False
                    result["stomps"] += 1
                    self.jump_timer = 2  # Bounce
                    self.vy = -1
                elif e.etype == EnemyType.TURTLE and e.is_shell and not e.shell_moving:
                    e.shell_moving = True
                    e.shell_dir = 1 if self.mario_col < e.col else -1
                    result["stomps"] += 1
                    self.jump_timer = 2
                    self.vy = -1
                else:
                    e.alive = False
                    result["stomps"] += 1
                    self.jump_timer = 2  # Bounce off stomp
                    self.vy = -1

            # Side contact with Mario (enemy same row, adjacent or same col)
            elif (abs(e.row - self.mario_row) <= 0
                  and abs(e.col - self.mario_col) <= 0):
                if e.is_dangerous:
                    result["mario_died"] = True
                    result["death_reason"] = f"touched_{e.etype.name}"

        return result

    def _ai_walk(self, e: Enemy):
        """Goomba/Turtle: walk in direction, reverse at walls."""
        new_col = e.col + e.direction
        below = e.row + 1

        # Check wall or edge
        if (new_col < 0 or new_col >= self.width
                or self._is_solid(e.row, new_col)):
            e.direction *= -1
            return

        # Check for floor under next position
        if below < self.GRID_H and not self._is_solid(below, new_col):
            # No floor ahead — reverse (enemies don't walk off edges)
            e.direction *= -1
            return

        e.col = new_col

        # Apply gravity to enemy
        if below < self.GRID_H and not self._is_solid(below, e.col):
            e.row = below

    def _ai_shell(self, e: Enemy):
        """Moving turtle shell: slides fast, kills other enemies."""
        new_col = e.col + e.shell_dir * 2  # Shells move faster
        if new_col < 0 or new_col >= self.width or self._is_solid(e.row, new_col):
            e.shell_dir *= -1
            return
        e.col = new_col

        # Shell kills other enemies it hits
        for other in self.enemies:
            if other is not e and other.alive and not other.is_shell:
                if other.row == e.row and other.col == e.col:
                    other.alive = False

    def _ai_piranha(self, e: Enemy):
        """Piranha plant: bobs up and down from pipe position."""
        e.bob_timer += 1
        if e.bob_timer >= 8:  # Change every 8 steps
            e.bob_timer = 0
            if e.bob_up:
                e.row += 1  # Go back down into pipe
                e.bob_up = False
            else:
                e.row -= 1  # Pop up
                e.bob_up = True

    def _ai_lakitu(self, e: Enemy):
        """Lakitu: follows Mario in the sky, stays at row 2."""
        e.row = 2  # Always in sky
        if self.mario_col > e.col:
            e.col = min(e.col + 1, self.width - 1)
        elif self.mario_col < e.col:
            e.col = max(e.col - 1, 0)

    # ── Helpers ───────────────────────────────────────────────────

    def _is_solid(self, row: int, col: int) -> bool:
        """Is this tile solid (blocks movement)?"""
        if row < 0 or row >= self.GRID_H or col < 0 or col >= self.width:
            return row < 0  # Above screen = not solid; sides/bottom = solid
        return self.grid[row, col] in SOLID_TILES

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.GRID_H and 0 <= col < self.width

    # ── Observation ───────────────────────────────────────────────

    def get_obs(self) -> np.ndarray:
        """
        Return game state as flat numpy array.

        Layout: viewport grid (16×20=320) + mario_pos (2) + physics (3)
                + coins/score (2) + enemy_states (6 × max_enemies) + flags (3)
        """
        # Viewport: 16×20 window centered on Mario
        vp_start = max(0, min(self.mario_col - 10, self.width - 20))
        vp_end = min(vp_start + 20, self.width)
        vp_width = vp_end - vp_start

        viewport = np.zeros((self.GRID_H, 20), dtype=np.float32)
        viewport[:, :vp_width] = self.grid[:, vp_start:vp_end].astype(np.float32)
        viewport /= max(N_TILE_TYPES - 1, 1)  # Normalize to [0, 1]

        # Mario position relative to viewport
        mario_vp_col = (self.mario_col - vp_start) / 20.0
        mario_vp_row = self.mario_row / self.GRID_H

        obs_parts = [
            viewport.flatten(),                                      # 320
            np.array([mario_vp_row, mario_vp_col], np.float32),     # 2
            np.array([                                               # 3
                float(self.on_ground),
                self.vy / MAX_FALL_SPEED,
                self.jump_timer / JUMP_DURATION,
            ], np.float32),
            np.array([                                               # 2
                self.coins / 100.0,
                self.mario_col / max(self.width, 1),  # Progress
            ], np.float32),
        ]

        # Enemy states (up to 8 enemies in viewport)
        enemies_in_view = [e for e in self.enemies if e.alive
                           and vp_start <= e.col < vp_end][:8]
        for i in range(8):
            if i < len(enemies_in_view):
                e = enemies_in_view[i]
                obs_parts.append(np.array([
                    e.etype / 3.0,
                    e.row / self.GRID_H,
                    (e.col - vp_start) / 20.0,
                    float(e.is_dangerous),
                    float(e.is_shell),
                    float(e.direction),
                ], np.float32))
            else:
                obs_parts.append(np.zeros(6, np.float32))

        obs_parts.append(np.array([
            float(self.alive),
            float(self.won),
            self.step_count / 1000.0,
        ], np.float32))

        return np.concatenate(obs_parts)

    @property
    def obs_size(self) -> int:
        """Expected observation vector size."""
        return 320 + 2 + 3 + 2 + 8 * 6 + 3  # = 378

    # ── Rendering ─────────────────────────────────────────────────

    def render_ascii(self, viewport: bool = True) -> str:
        """
        Render the level as ASCII text.

        Args:
            viewport: If True, show 20-column viewport around Mario.
                      If False, show entire level.
        """
        if viewport:
            vp_start = max(0, min(self.mario_col - 10, self.width - 20))
            vp_end = min(vp_start + 20, self.width)
        else:
            vp_start = 0
            vp_end = self.width

        lines = []
        for row in range(self.GRID_H):
            chars = []
            for col in range(vp_start, vp_end):
                # Check for Mario
                if row == self.mario_row and col == self.mario_col:
                    chars.append('M')
                    continue

                # Check for enemies
                enemy_here = False
                for e in self.enemies:
                    if e.alive and e.row == row and e.col == col:
                        if e.etype == EnemyType.GOOMBA:
                            chars.append('G')
                        elif e.etype == EnemyType.TURTLE:
                            chars.append('s' if e.is_shell else 'T')
                        elif e.etype == EnemyType.PIRANHA:
                            chars.append('P')
                        elif e.etype == EnemyType.LAKITU:
                            chars.append('L')
                        else:
                            chars.append('E')
                        enemy_here = True
                        break

                if not enemy_here:
                    tile = self.grid[row, col]
                    chars.append(TILE_CHAR.get(tile, '?'))

            lines.append(''.join(chars))

        # Add status bar
        status = (f"  Mario({self.mario_row},{self.mario_col}) "
                  f"Coins:{self.coins} Score:{self.score:.0f} "
                  f"Step:{self.step_count} "
                  f"{'ALIVE' if self.alive else 'DEAD'} "
                  f"{'WON!' if self.won else ''}")
        lines.append(status)

        return '\n'.join(lines)

    def render_full_ascii(self) -> str:
        """Render the full level (all screens)."""
        return self.render_ascii(viewport=False)

    # ── Save / Load ───────────────────────────────────────────────

    def save(self) -> Dict[str, Any]:
        """Serialize full state for checkpointing."""
        return {
            "grid": self.grid.copy(),
            "width": self.width,
            "mario_row": self.mario_row,
            "mario_col": self.mario_col,
            "vy": self.vy,
            "jump_timer": self.jump_timer,
            "on_ground": self.on_ground,
            "alive": self.alive,
            "won": self.won,
            "coins": self.coins,
            "score": self.score,
            "step_count": self.step_count,
            "max_x_reached": self.max_x_reached,
            "scroll_x": self.scroll_x,
            "enemies": [
                {
                    "etype": int(e.etype), "row": e.row, "col": e.col,
                    "alive": e.alive, "direction": e.direction,
                    "is_shell": e.is_shell, "shell_moving": e.shell_moving,
                    "shell_dir": e.shell_dir, "bob_timer": e.bob_timer,
                    "bob_up": e.bob_up,
                }
                for e in self.enemies
            ],
        }

    def load(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self.grid = state["grid"].copy()
        self.width = state["width"]
        self.mario_row = state["mario_row"]
        self.mario_col = state["mario_col"]
        self.vy = state["vy"]
        self.jump_timer = state["jump_timer"]
        self.on_ground = state["on_ground"]
        self.alive = state["alive"]
        self.won = state["won"]
        self.coins = state["coins"]
        self.score = state["score"]
        self.step_count = state["step_count"]
        self.max_x_reached = state["max_x_reached"]
        self.scroll_x = state.get("scroll_x", 0)
        self.enemies = []
        for ed in state["enemies"]:
            self.enemies.append(Enemy(
                etype=EnemyType(ed["etype"]),
                row=ed["row"], col=ed["col"],
                alive=ed["alive"], direction=ed["direction"],
                is_shell=ed["is_shell"], shell_moving=ed["shell_moving"],
                shell_dir=ed["shell_dir"], bob_timer=ed["bob_timer"],
                bob_up=ed["bob_up"],
            ))

    # ── Completability Check ──────────────────────────────────────

    def is_completable(self) -> bool:
        """
        Check if Mario can reach the flag from start position.

        Uses BFS with jump physics simulation:
        - State: (row, col, airborne_ticks)
        - Transitions: all valid actions from each state
        """
        # Find flag position
        flag_positions = set()
        for r in range(self.GRID_H):
            for c in range(self.width):
                if self.grid[r, c] == Tile.FLAG:
                    flag_positions.add((r, c))

        if not flag_positions:
            return False

        # BFS state: (row, col, jump_timer, on_ground)
        # Simplified: just (row, col) reachability with jump consideration
        visited = set()
        queue = deque()

        start = (self.mario_row, self.mario_col, 0, True)
        queue.append(start)
        visited.add((self.mario_row, self.mario_col))

        while queue:
            row, col, jtimer, on_gnd = queue.popleft()

            # Win check
            if (row, col) in flag_positions:
                return True

            # Generate successors for each action
            for action in [Action.LEFT, Action.RIGHT, Action.JUMP,
                           Action.JUMP_LEFT, Action.JUMP_RIGHT, Action.NOOP]:
                nr, nc = row, col
                n_jtimer = jtimer
                n_on_gnd = on_gnd

                # Horizontal
                dx = 0
                if action in (Action.LEFT, Action.JUMP_LEFT):
                    dx = -1
                elif action in (Action.RIGHT, Action.JUMP_RIGHT):
                    dx = 1

                # Jump initiation
                if action in (Action.JUMP, Action.JUMP_LEFT, Action.JUMP_RIGHT):
                    if on_gnd:
                        n_jtimer = JUMP_DURATION
                        n_on_gnd = False

                # Vertical: rising
                if n_jtimer > 0:
                    n_jtimer -= 1
                    up_row = nr - 1
                    if up_row >= 0 and not self._is_solid(up_row, nc):
                        nr = up_row
                    else:
                        n_jtimer = 0  # Bump head
                elif not n_on_gnd:
                    # Falling
                    down_row = nr + 1
                    if down_row < self.GRID_H and not self._is_solid(down_row, nc):
                        nr = down_row
                    elif down_row >= self.GRID_H:
                        continue  # Fell to death
                    else:
                        n_on_gnd = True  # Landed

                # Apply horizontal
                if dx != 0:
                    new_c = nc + dx
                    if 0 <= new_c < self.width and not self._is_solid(nr, new_c):
                        nc = new_c

                # Check ground below
                if n_on_gnd and n_jtimer == 0:
                    below = nr + 1
                    if below >= self.GRID_H or not self._is_solid(below, nc):
                        if below >= self.GRID_H:
                            continue  # Death
                        n_on_gnd = False

                # Skip pit/death tiles
                if self._in_bounds(nr, nc) and self.grid[nr, nc] == Tile.PIT:
                    continue

                state_key = (nr, nc)
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((nr, nc, n_jtimer, n_on_gnd))

        return False
