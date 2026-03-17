"""
mario_difficulty_analyzer.py — Death Hotspot Analysis and Targeted Practice.

Identifies where Mario dies most frequently and generates focused practice
levels around those locations to accelerate mastery of difficult spots.

Three components:
  1. DifficultyAnalyzer — clusters death events into hotspots
  2. Practice level generator — carves a short window from an existing level
     and trains the agent exclusively on that window until success climbs
  3. HotspotDrillCurriculum — wrapper that cycles through hotspots until
     each reaches a success threshold, then releases back to main training

Design:
  - Death data comes from ZoneCurriculum.record_death() / _death_log
  - Practice level: 25-column window around the hotspot column
  - Mario spawns 8 columns left of the hotspot (approach run)
  - Success = reaching 10 columns right of the hotspot
  - Integrates with MarioAdapter (reset/step interface)

Usage:
    # After some training with ZoneCurriculum:
    hotspots = curriculum.get_death_hotspots()
    analyzer = DifficultyAnalyzer(death_log=curriculum._death_log)
    drill = HotspotDrillCurriculum(level=sim, analyzer=analyzer, adapter=adapter)

    while not drill.all_mastered():
        sim_slice, info = drill.get_episode()
        obs = adapter.reset(sim_slice)
        # ... train agent ...
        drill.report_result(success=..., final_col=...)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mario_simulator import (
    Enemy, MarioSimulator, Tile, N_TILE_TYPES,
    JUMP_DURATION, SOLID_TILES,
)

# Module-level aliases for convenience
GROUND_ROW = MarioSimulator.GROUND_ROW


# ═══════════════════════════════════════════════════════════════
# HOTSPOT DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════

@dataclass
class Hotspot:
    """A geographic death cluster in the level."""
    col: int                          # Center column
    col_range: Tuple[int, int]        # (min_col, max_col) of cluster
    death_count: int                  # Total deaths in cluster
    most_common_action: int           # Action most often taken before death
    action_counts: Dict[int, int]     # {action: count}
    death_rows: List[int]             # Row distribution of deaths
    mastered: bool = False            # Has success threshold been reached?
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.attempts)

    def __repr__(self):
        return (
            f"Hotspot(col={self.col}, range={self.col_range}, "
            f"deaths={self.death_count}, "
            f"success={self.success_rate:.0%}/{self.attempts} attempts)"
        )


# ═══════════════════════════════════════════════════════════════
# DIFFICULTY ANALYZER
# ═══════════════════════════════════════════════════════════════

class DifficultyAnalyzer:
    """
    Clusters death events into geographic hotspots.

    Accepts death_log from ZoneCurriculum: list of (col, row, action) tuples.
    Produces ranked list of Hotspot objects sorted by death_count descending.
    """

    def __init__(
        self,
        death_log: Optional[List[Tuple[int, int, int]]] = None,
        cluster_radius: int = 4,
        min_deaths_for_hotspot: int = 3,
    ):
        """
        Args:
            death_log: List of (col, row, last_action) tuples from training.
            cluster_radius: Columns within this range are merged into one hotspot.
            min_deaths_for_hotspot: Minimum deaths to qualify as a hotspot.
        """
        self.cluster_radius = cluster_radius
        self.min_deaths_for_hotspot = min_deaths_for_hotspot
        self._death_log: List[Tuple[int, int, int]] = list(death_log or [])
        self._hotspots: Optional[List[Hotspot]] = None

    def record_death(self, col: int, row: int, last_action: int):
        """Add a single death event to the log."""
        self._death_log.append((col, row, last_action))
        self._hotspots = None  # Invalidate cache

    def update(self, death_log: List[Tuple[int, int, int]]):
        """Bulk-update with a new death log (replaces existing log)."""
        self._death_log = list(death_log)
        self._hotspots = None

    def analyze(self) -> List[Hotspot]:
        """
        Run hotspot analysis on accumulated death events.

        Returns:
            List of Hotspot objects sorted by death_count descending.
        """
        if not self._death_log:
            return []

        if self._hotspots is not None:
            return self._hotspots

        # Count deaths per column
        col_deaths: Counter = Counter(d[0] for d in self._death_log)

        # Build col → [rows] and col → [actions] maps for detail extraction
        col_rows: Dict[int, List[int]] = defaultdict(list)
        col_actions: Dict[int, List[int]] = defaultdict(list)
        for col, row, action in self._death_log:
            col_rows[col].append(row)
            col_actions[col].append(action)

        # Merge nearby columns into clusters (1D greedy)
        sorted_cols = sorted(col_deaths.keys())
        clusters: List[List[int]] = []
        current: List[int] = [sorted_cols[0]]

        for c in sorted_cols[1:]:
            if c - current[-1] <= self.cluster_radius:
                current.append(c)
            else:
                clusters.append(current)
                current = [c]
        clusters.append(current)

        # Build Hotspot objects
        hotspots = []
        for cluster_cols in clusters:
            total = sum(col_deaths[c] for c in cluster_cols)
            if total < self.min_deaths_for_hotspot:
                continue

            # Center = column with most deaths in cluster
            center = max(cluster_cols, key=lambda c: col_deaths[c])

            # Aggregate rows and actions
            all_rows: List[int] = []
            all_actions: List[int] = []
            for c in cluster_cols:
                all_rows.extend(col_rows[c])
                all_actions.extend(col_actions[c])

            action_counts = dict(Counter(all_actions))
            most_common = max(action_counts, key=action_counts.get) if action_counts else 0

            hotspots.append(Hotspot(
                col=center,
                col_range=(cluster_cols[0], cluster_cols[-1]),
                death_count=total,
                most_common_action=most_common,
                action_counts=action_counts,
                death_rows=all_rows,
            ))

        self._hotspots = sorted(hotspots, key=lambda h: h.death_count, reverse=True)
        return self._hotspots

    def report(self) -> str:
        """Human-readable hotspot report."""
        hotspots = self.analyze()
        if not hotspots:
            return "No hotspots found (need more deaths)"

        lines = [f"Difficulty Analysis — {len(self._death_log)} total deaths, "
                 f"{len(hotspots)} hotspots:"]

        action_names = {0: "NOOP", 1: "LEFT", 2: "RIGHT", 3: "JUMP",
                        4: "JLEFT", 5: "JRIGHT", 6: "RUN", 7: "RJUMP"}

        for i, h in enumerate(hotspots):
            top_action = action_names.get(h.most_common_action, str(h.most_common_action))
            avg_row = sum(h.death_rows) / max(1, len(h.death_rows))
            lines.append(
                f"  #{i+1} col {h.col_range[0]}-{h.col_range[1]} "
                f"({h.death_count} deaths, "
                f"avg row {avg_row:.1f}, "
                f"last action: {top_action})"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# PRACTICE LEVEL GENERATOR
# ═══════════════════════════════════════════════════════════════

def make_practice_level(
    source_sim: MarioSimulator,
    center_col: int,
    approach_cols: int = 8,
    challenge_cols: int = 12,
) -> Optional[MarioSimulator]:
    """
    Carve a focused practice window from an existing level.

    Mario spawns `approach_cols` before center_col so he gets a running
    approach to the hotspot, and the level ends `challenge_cols` after.

    Args:
        source_sim: Full level to extract the window from.
        center_col: The hotspot column (death cluster center).
        approach_cols: How many columns before center to include.
        challenge_cols: How many columns after center to include.

    Returns:
        A new MarioSimulator covering the windowed slice, or None if
        the window would be invalid (too small, off level bounds).
    """
    total_window = approach_cols + challenge_cols
    if total_window < 8:
        return None

    # Clamp window to level bounds
    start_col = max(0, center_col - approach_cols)
    end_col = min(source_sim.width - 1, center_col + challenge_cols)
    window_width = end_col - start_col + 1

    if window_width < 8:
        return None

    # Extract grid slice
    grid_slice = source_sim.grid[:, start_col:end_col + 1].copy()

    # Find a valid spawn row in the approach section
    spawn_col_in_slice = min(2, window_width - 3)
    spawn_row = GROUND_ROW - 1  # Default: one above ground

    # Walk up from GROUND_ROW to find first empty tile in spawn column
    for r in range(GROUND_ROW - 1, 0, -1):
        if grid_slice[r, spawn_col_in_slice] == Tile.EMPTY:
            below = r + 1
            if below < MarioSimulator.GRID_H and grid_slice[below, spawn_col_in_slice] in SOLID_TILES:
                spawn_row = r
                break

    # Place Mario marker
    grid_slice[spawn_row, spawn_col_in_slice] = Tile.PLAYER

    # Place flag at end of window (if not already there)
    flag_placed = False
    end_in_slice = window_width - 2
    for r in range(GROUND_ROW - 3, GROUND_ROW):
        if grid_slice[r, end_in_slice] == Tile.EMPTY:
            grid_slice[r, end_in_slice] = Tile.FLAG
            flag_placed = True
            break
    if not flag_placed:
        # Force-place flag by clearing the column top
        grid_slice[GROUND_ROW - 3, end_in_slice] = Tile.FLAG

    # Extract enemies within the window (offset their columns)
    practice_enemies = []
    for e in source_sim.enemies:
        if start_col <= e.col <= end_col:
            ec = deepcopy(e)
            ec.col = e.col - start_col
            practice_enemies.append(ec)

    try:
        return MarioSimulator(grid_slice, practice_enemies)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# HOTSPOT DRILL CURRICULUM
# ═══════════════════════════════════════════════════════════════

class HotspotDrillCurriculum:
    """
    Targeted drill curriculum that cycles through death hotspots.

    For each hotspot (sorted by severity):
      - Generates a short practice level window around the hotspot
      - Trains until success_threshold is reached
      - Marks hotspot as mastered and moves on

    Use alongside ZoneCurriculum: run drills between zone episodes to
    break through specific difficult spots faster.

    Usage:
        drill = HotspotDrillCurriculum(source_sim, analyzer, pass_threshold=0.7)
        while not drill.all_mastered():
            practice_sim, info = drill.get_episode()
            obs = adapter.reset(practice_sim)
            # ... train agent ...
            drill.report_result(success, final_col)
    """

    def __init__(
        self,
        source_sim: MarioSimulator,
        analyzer: DifficultyAnalyzer,
        pass_threshold: float = 0.70,
        min_attempts: int = 20,
        approach_cols: int = 8,
        challenge_cols: int = 12,
        max_hotspots: int = 5,
    ):
        """
        Args:
            source_sim: Full level the hotspots belong to.
            analyzer: DifficultyAnalyzer with populated death log.
            pass_threshold: Success rate needed to mark a hotspot mastered.
            min_attempts: Minimum attempts before evaluating pass threshold.
            approach_cols: Run-up distance before hotspot center.
            challenge_cols: Distance after hotspot center.
            max_hotspots: Only drill the top-N hotspots by death count.
        """
        self.source_sim = source_sim
        self.analyzer = analyzer
        self.pass_threshold = pass_threshold
        self.min_attempts = min_attempts
        self.approach_cols = approach_cols
        self.challenge_cols = challenge_cols
        self.max_hotspots = max_hotspots

        self._hotspots: Optional[List[Hotspot]] = None
        self._current_idx: int = 0
        self._practice_sims: Dict[int, Optional[MarioSimulator]] = {}
        self._practice_states: Dict[int, Any] = {}  # saved sim states
        self._total_drill_episodes: int = 0

    def _load_hotspots(self):
        """Refresh hotspot list from analyzer."""
        all_hotspots = self.analyzer.analyze()
        self._hotspots = all_hotspots[:self.max_hotspots]
        self._current_idx = 0

        # Pre-generate practice levels
        for h in self._hotspots:
            if h.col not in self._practice_sims:
                psim = make_practice_level(
                    self.source_sim, h.col,
                    self.approach_cols, self.challenge_cols,
                )
                self._practice_sims[h.col] = psim
                if psim is not None:
                    self._practice_states[h.col] = psim.save()

    def _current_hotspot(self) -> Optional[Hotspot]:
        if self._hotspots is None:
            self._load_hotspots()
        while self._current_idx < len(self._hotspots):
            h = self._hotspots[self._current_idx]
            if h.mastered:
                self._current_idx += 1
                continue
            if self._practice_sims.get(h.col) is None:
                # Can't make a practice level here — skip
                h.mastered = True
                self._current_idx += 1
                continue
            return h
        return None

    @property
    def active_hotspot(self) -> Optional[Hotspot]:
        return self._current_hotspot()

    def all_mastered(self) -> bool:
        """True if all hotspots have been mastered (or are undrillable)."""
        h = self._current_hotspot()
        return h is None

    def get_episode(self) -> Tuple[Optional[MarioSimulator], Dict[str, Any]]:
        """
        Get a practice simulator for the current active hotspot.

        Returns:
            (practice_sim, info) or (None, info) if all mastered.
        """
        self._total_drill_episodes += 1
        h = self._current_hotspot()
        if h is None:
            return None, {"all_mastered": True}

        sim = self._practice_sims[h.col]
        # Reset to pristine state
        sim.load(self._practice_states[h.col])

        window_width = sim.width
        success_col = min(
            sim.mario_col + self.approach_cols + self.challenge_cols - 2,
            window_width - 2,
        )

        info = {
            "hotspot_col": h.col,
            "hotspot_range": h.col_range,
            "death_count": h.death_count,
            "success_rate": round(h.success_rate, 3),
            "attempts": h.attempts,
            "practice_width": window_width,
            "success_col": success_col,
            "drill_episode": self._total_drill_episodes,
        }
        return sim, info

    def report_result(self, success: bool, final_col: int = 0):
        """
        Report result of a drill episode.

        Args:
            success: Did Mario clear the challenge section?
            final_col: Final column reached (within the practice sim).
        """
        h = self._current_hotspot()
        if h is None:
            return

        h.attempts += 1
        if success:
            h.successes += 1

        # Check mastery
        if h.attempts >= self.min_attempts and h.success_rate >= self.pass_threshold:
            h.mastered = True
            self._current_idx += 1

    def stats(self) -> Dict[str, Any]:
        """Drill curriculum statistics."""
        if self._hotspots is None:
            return {"loaded": False}

        mastered = sum(1 for h in self._hotspots if h.mastered)
        active = self._current_hotspot()

        return {
            "total_hotspots": len(self._hotspots),
            "mastered": mastered,
            "remaining": len(self._hotspots) - mastered,
            "all_mastered": active is None,
            "active_hotspot_col": active.col if active else None,
            "active_hotspot_deaths": active.death_count if active else None,
            "active_success_rate": round(active.success_rate, 3) if active else None,
            "active_attempts": active.attempts if active else None,
            "total_drill_episodes": self._total_drill_episodes,
        }

    def report(self) -> str:
        """Human-readable drill progress."""
        if self._hotspots is None:
            return "No hotspots loaded yet"

        lines = [
            f"Hotspot Drill Curriculum — "
            f"{self._total_drill_episodes} total drill episodes"
        ]
        for h in self._hotspots:
            bar = "█" * int(h.success_rate * 10) + "░" * (10 - int(h.success_rate * 10))
            status = "✓ MASTERED" if h.mastered else f"{h.success_rate:.0%}/{self.pass_threshold:.0%}"
            lines.append(
                f"  col {h.col_range[0]}-{h.col_range[1]} "
                f"({h.death_count} deaths): "
                f"|{bar}| {status} ({h.attempts} attempts)"
            )
        return "\n".join(lines)
