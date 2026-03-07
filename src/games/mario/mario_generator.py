"""
mario_generator.py — Procedural level generation for Mario ASCII simulator.

Generates completable Mario levels at controllable difficulty tiers.
Each tier introduces new obstacles and enemies:

  Tier 1: Flat ground, run right to flag. 1 screen.
  Tier 2: Ground + platforms + gaps (no pits, no enemies). 1 screen.
  Tier 3: Goombas + obstacles. 1 screen.
  Tier 4: Turtles, goombas, varied terrain. 1-3 screens.
  Tier 5: Pipes, pipe monsters, question blocks. 3 screens.
  Tier 6: Pits, sky enemies, full complexity. 3-5 screens.
  Tier 7: Long levels, all mechanics, high density. 5-7 screens.

Usage:
    gen = MarioLevelGenerator(seed=42)
    level = gen.generate(tier=1)
    print(level.render_ascii())

    batch = gen.generate_batch(100, tier=3)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .mario_simulator import (
    Action, Enemy, EnemyType, MarioSimulator, Tile,
)


# ── Tier configurations ───────────────────────────────────────────────

TIER_CONFIG = {
    1: {
        "name": "flat_ground",
        "n_screens": (1, 1),
        "platforms": (0, 0),
        "gaps": (0, 0),         # Gaps where ground is missing (not pits)
        "pits": (0, 0),
        "enemies_goomba": (0, 0),
        "enemies_turtle": (0, 0),
        "enemies_piranha": (0, 0),
        "enemies_lakitu": (0, 0),
        "pipes": (0, 0),
        "coins": (2, 5),
        "questions": (0, 0),
        "bricks": (0, 0),
        "desc": "Run right on flat ground",
    },
    2: {
        "name": "platforms_and_gaps",
        "n_screens": (1, 1),
        "platforms": (2, 5),
        "gaps": (1, 3),
        "pits": (0, 0),
        "enemies_goomba": (0, 0),
        "enemies_turtle": (0, 0),
        "enemies_piranha": (0, 0),
        "enemies_lakitu": (0, 0),
        "pipes": (0, 0),
        "coins": (3, 8),
        "questions": (1, 3),
        "bricks": (1, 4),
        "desc": "Jump over gaps and onto platforms",
    },
    3: {
        "name": "goombas",
        "n_screens": (1, 1),
        "platforms": (2, 4),
        "gaps": (1, 3),
        "pits": (0, 0),
        "enemies_goomba": (2, 4),
        "enemies_turtle": (0, 0),
        "enemies_piranha": (0, 0),
        "enemies_lakitu": (0, 0),
        "pipes": (0, 1),
        "coins": (3, 8),
        "questions": (1, 3),
        "bricks": (2, 5),
        "desc": "Goombas + obstacles",
    },
    4: {
        "name": "turtles_variety",
        "n_screens": (1, 3),
        "platforms": (3, 7),
        "gaps": (2, 5),
        "pits": (0, 1),
        "enemies_goomba": (2, 4),
        "enemies_turtle": (1, 3),
        "enemies_piranha": (0, 0),
        "enemies_lakitu": (0, 0),
        "pipes": (1, 2),
        "coins": (5, 12),
        "questions": (2, 5),
        "bricks": (3, 8),
        "desc": "Turtles, goombas, varied terrain",
    },
    5: {
        "name": "pipes_and_blocks",
        "n_screens": (3, 3),
        "platforms": (4, 8),
        "gaps": (3, 6),
        "pits": (1, 3),
        "enemies_goomba": (3, 5),
        "enemies_turtle": (1, 3),
        "enemies_piranha": (1, 2),
        "enemies_lakitu": (0, 0),
        "pipes": (2, 4),
        "coins": (8, 16),
        "questions": (3, 6),
        "bricks": (4, 10),
        "desc": "Pipes, pipe monsters, question blocks",
    },
    6: {
        "name": "pits_and_sky",
        "n_screens": (3, 5),
        "platforms": (5, 10),
        "gaps": (3, 7),
        "pits": (2, 5),
        "enemies_goomba": (3, 6),
        "enemies_turtle": (2, 4),
        "enemies_piranha": (1, 3),
        "enemies_lakitu": (1, 1),
        "pipes": (2, 5),
        "coins": (10, 20),
        "questions": (3, 7),
        "bricks": (5, 12),
        "desc": "Pits, sky enemies, full complexity",
    },
    7: {
        "name": "full_mario",
        "n_screens": (5, 7),
        "platforms": (8, 15),
        "gaps": (5, 10),
        "pits": (3, 7),
        "enemies_goomba": (5, 8),
        "enemies_turtle": (3, 5),
        "enemies_piranha": (2, 4),
        "enemies_lakitu": (1, 2),
        "pipes": (3, 6),
        "coins": (15, 30),
        "questions": (5, 10),
        "bricks": (8, 16),
        "desc": "Long levels, all mechanics, high density",
    },
}

SCREEN_WIDTH = 20
GRID_H = MarioSimulator.GRID_H  # 16
GROUND_ROW = MarioSimulator.GROUND_ROW  # 13


class MarioLevelGenerator:
    """
    Generate completable Mario levels at controllable difficulty.

    Uses randomized placement with constraint checking:
    1. Create ground profile (with gaps for jumpable sections)
    2. Place platforms at various heights
    3. Place pipes, bricks, question blocks
    4. Place enemies on valid surfaces
    5. Place coins and flag
    6. Verify completability
    7. If not completable, regenerate
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.complexity_tier = 1
        self._total_generated = 0
        self._total_attempts = 0

    def _safe_randint(self, low: int, high: int) -> int:
        """Safe randint that handles (0, 0) ranges."""
        if low >= high:
            return low
        return int(self.rng.randint(low, high))

    def generate(
        self,
        tier: Optional[int] = None,
        max_attempts: int = 100,
    ) -> Optional[MarioSimulator]:
        """Generate a single completable level."""
        tier = tier or self.complexity_tier
        config = TIER_CONFIG.get(tier, TIER_CONFIG[1])

        for attempt in range(max_attempts):
            self._total_attempts += 1
            sim = self._try_generate(config, tier)
            if sim is not None and sim.is_completable():
                self._total_generated += 1
                return sim

        return None

    def generate_batch(
        self,
        n: int,
        tier: Optional[int] = None,
    ) -> List[MarioSimulator]:
        """Generate n completable levels."""
        results = []
        for _ in range(n):
            sim = self.generate(tier)
            if sim is not None:
                results.append(sim)
        return results

    def advance_tier(self) -> int:
        self.complexity_tier = min(self.complexity_tier + 1, 7)
        return self.complexity_tier

    def _try_generate(self, config: dict, tier: int) -> Optional[MarioSimulator]:
        """Single attempt to generate a level."""
        n_min, n_max = config["n_screens"]
        n_screens = self.rng.randint(n_min, n_max + 1)
        width = n_screens * SCREEN_WIDTH
        grid = np.full((GRID_H, width), Tile.EMPTY, dtype=np.uint8)
        enemies: List[Enemy] = []

        # ── 1. Ground profile ─────────────────────────────────────
        # Fill ground rows (13, 14, 15) by default
        for col in range(width):
            grid[GROUND_ROW:, col] = Tile.GROUND

        # ── 2. Gaps (ground removed, jumpable over) ───────────────
        n_gaps = self._safe_randint(*config["gaps"])
        gap_cols = set()
        for _ in range(n_gaps):
            gap_start = self.rng.randint(4, width - 4)
            gap_len = self.rng.randint(2, 4)  # 2-3 tiles wide
            for c in range(gap_start, min(gap_start + gap_len, width - 2)):
                grid[GROUND_ROW:, c] = Tile.EMPTY
                gap_cols.add(c)

        # ── 3. Pits (deadly holes) ────────────────────────────────
        n_pits = self._safe_randint(*config["pits"])
        pit_cols = set()
        for _ in range(n_pits):
            pit_start = self.rng.randint(5, width - 5)
            # Don't overlap with gaps
            if pit_start in gap_cols:
                continue
            pit_len = self.rng.randint(2, 4)
            for c in range(pit_start, min(pit_start + pit_len, width - 3)):
                grid[GROUND_ROW:, c] = Tile.PIT
                pit_cols.add(c)

        # ── 4. Platforms ──────────────────────────────────────────
        n_platforms = self._safe_randint(*config["platforms"])
        for _ in range(n_platforms):
            plat_col = self.rng.randint(3, width - 5)
            plat_row = self.rng.randint(6, GROUND_ROW - 2)  # Not too high, not on ground
            plat_len = self.rng.randint(3, 7)
            for c in range(plat_col, min(plat_col + plat_len, width - 1)):
                if grid[plat_row, c] == Tile.EMPTY:
                    grid[plat_row, c] = Tile.PLATFORM

        # ── 5. Pipes ─────────────────────────────────────────────
        n_pipes = self._safe_randint(*config["pipes"])
        pipe_positions = []
        for _ in range(n_pipes):
            pc = self.rng.randint(5, width - 5)
            # Pipe is 2 wide, 2-3 tall, sitting on ground
            if (grid[GROUND_ROW, pc] in (Tile.GROUND,)
                    and grid[GROUND_ROW, pc + 1] in (Tile.GROUND,)
                    and pc not in gap_cols and pc not in pit_cols
                    and (pc + 1) not in gap_cols and (pc + 1) not in pit_cols):
                pipe_h = self.rng.randint(2, 4)
                for pr in range(GROUND_ROW - pipe_h, GROUND_ROW):
                    grid[pr, pc] = Tile.PIPE_L
                    grid[pr, pc + 1] = Tile.PIPE_R
                pipe_positions.append((GROUND_ROW - pipe_h, pc))

        # ── 6. Bricks ────────────────────────────────────────────
        n_bricks = self._safe_randint(*config["bricks"])
        for _ in range(n_bricks):
            bc = self.rng.randint(3, width - 2)
            br = self.rng.randint(7, GROUND_ROW - 2)
            if grid[br, bc] == Tile.EMPTY:
                grid[br, bc] = Tile.BRICK

        # ── 7. Question blocks ────────────────────────────────────
        n_questions = self._safe_randint(*config["questions"])
        for _ in range(n_questions):
            qc = self.rng.randint(3, width - 2)
            qr = self.rng.randint(7, GROUND_ROW - 2)
            if grid[qr, qc] == Tile.EMPTY:
                grid[qr, qc] = Tile.QUESTION

        # ── 8. Coins ─────────────────────────────────────────────
        n_coins = self._safe_randint(*config["coins"])
        for _ in range(n_coins):
            cc = self.rng.randint(3, width - 2)
            cr = self.rng.randint(4, GROUND_ROW - 1)
            if grid[cr, cc] == Tile.EMPTY:
                grid[cr, cc] = Tile.COIN

        # ── 9. Player start ──────────────────────────────────────
        # Find leftmost ground position
        start_col = 2
        for c in range(2, min(6, width)):
            if grid[GROUND_ROW, c] == Tile.GROUND:
                start_col = c
                break
        grid[GROUND_ROW - 1, start_col] = Tile.PLAYER

        # ── 10. Flag at end ───────────────────────────────────────
        flag_col = width - 2
        # Find a ground position near the end
        for c in range(width - 2, width - 6, -1):
            if grid[GROUND_ROW, c] == Tile.GROUND:
                flag_col = c
                break
        # Flag pole: 3 tiles tall
        for fr in range(GROUND_ROW - 3, GROUND_ROW):
            if grid[fr, flag_col] == Tile.EMPTY:
                grid[fr, flag_col] = Tile.FLAG

        # ── 11. Enemies ───────────────────────────────────────────
        def _place_enemies(etype: EnemyType, count_range: Tuple[int, int]):
            n = self._safe_randint(*count_range)
            for _ in range(n):
                ec = self.rng.randint(6, width - 4)  # Not too close to start/end
                if etype == EnemyType.PIRANHA:
                    # Place on a pipe
                    if pipe_positions:
                        pr, pc = pipe_positions[self.rng.randint(len(pipe_positions))]
                        enemies.append(Enemy(etype=etype, row=pr - 1, col=pc))
                elif etype == EnemyType.LAKITU:
                    enemies.append(Enemy(etype=etype, row=2, col=ec))
                else:
                    # Place on ground
                    er = GROUND_ROW - 1
                    if (grid[GROUND_ROW, ec] == Tile.GROUND
                            and grid[er, ec] == Tile.EMPTY):
                        enemies.append(Enemy(
                            etype=etype, row=er, col=ec,
                            direction=-1 if self.rng.random() < 0.5 else 1
                        ))

        _place_enemies(EnemyType.GOOMBA, config["enemies_goomba"])
        _place_enemies(EnemyType.TURTLE, config["enemies_turtle"])
        _place_enemies(EnemyType.PIRANHA, config["enemies_piranha"])
        _place_enemies(EnemyType.LAKITU, config["enemies_lakitu"])

        # ── Build simulator ───────────────────────────────────────
        try:
            sim = MarioSimulator(grid, enemies)
            return sim
        except Exception:
            return None

    def report(self) -> dict:
        return {
            "current_tier": self.complexity_tier,
            "tier_name": TIER_CONFIG.get(
                self.complexity_tier, {}
            ).get("name", "unknown"),
            "total_generated": self._total_generated,
            "total_attempts": self._total_attempts,
            "success_rate": (
                round(self._total_generated / max(1, self._total_attempts), 3)
            ),
        }
