"""
mario_zone_curriculum.py — Zone-Based Go-Explore Curriculum for Mario W1-1.

Breaks the level into geographic zones and trains each to a success threshold
before advancing. Combines with checkpoint management and DiscoRL-style
column-novelty bonuses for efficient exploration.

Zone decomposition (for generated Tier 5+ levels, ~60-100 columns):
  Zone 0: columns 0–20   (flat approach + first obstacles)
  Zone 1: columns 0–35   (first gap + Goombas)
  Zone 2: columns 0–50   (pipe section)
  Zone 3: columns 0–65   (brick/question zone)
  Zone 4: columns 0–80   (second gap cluster)
  Zone 5: columns 0–end  (full level — staircase + flagpole)

Key design:
  - Zones are cumulative (Zone N includes all of Zone 0..N-1)
  - Simulator state is checkpointed at each zone boundary
  - On reset: 50% spawn at zone start, 50% from beginning (anti-forgetting)
  - Promotion after 100 evals at ≥threshold; demotion if <50% for 50 evals
  - Column-novelty bonus (DiscoRL): +bonus for first visit to new columns

Usage:
    gen = MarioLevelGenerator(seed=42)
    curriculum = ZoneCurriculum(generator=gen, tier=5)

    agent = MarioICMAgent(obs_dim=378, n_actions=8)
    adapter = MarioAdapter()

    for episode in range(5000):
        sim, zone_info = curriculum.get_episode()
        obs = adapter.reset(sim)
        done = False
        while not done:
            action = agent.step(obs)
            obs, reward, done, info = adapter.step(action)
            # Add column novelty bonus
            novelty = curriculum.column_visited(sim.mario_col)
            agent.learn(reward + novelty, done)
        curriculum.report_result(sim.mario_col, sim.won, sim.alive)
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mario_simulator import MarioSimulator
from .mario_generator import MarioLevelGenerator


# ═══════════════════════════════════════════════════════════════════════
# ZONE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Zone:
    """A geographic zone within a Mario level."""
    id: int
    name: str
    target_col: int         # Episode succeeds if mario_col reaches this
    pass_threshold: float   # Success rate needed to advance (0-1)
    eval_window: int        # Number of recent episodes to evaluate

    def __repr__(self):
        return f"Zone({self.id}: {self.name}, target={self.target_col}, pass={self.pass_threshold:.0%})"


def make_zones(level_width: int) -> List[Zone]:
    """
    Create zone decomposition scaled to the actual level width.

    Zones are proportional: Zone 0 is the first ~20%, Zone 5 is 100%.
    """
    w = max(level_width, 20)  # Guard against tiny levels

    # Proportional column milestones
    milestones = [
        (0.20, "approach",     0.90, 100),  # First 20%: easy start
        (0.35, "first_gap",    0.80, 100),  # 35%: first real obstacle
        (0.50, "pipes",        0.80, 100),  # 50%: mid-level complexity
        (0.70, "blocks",       0.75, 100),  # 70%: bricks/questions
        (0.85, "gap_cluster",  0.75, 100),  # 85%: late-game gauntlet
        (1.00, "full_level",   0.70, 100),  # 100%: clear the level
    ]

    zones = []
    for i, (frac, name, thresh, window) in enumerate(milestones):
        target = min(int(w * frac), w - 1)
        zones.append(Zone(
            id=i,
            name=name,
            target_col=target,
            pass_threshold=thresh,
            eval_window=window,
        ))
    return zones


# ═══════════════════════════════════════════════════════════════════════
# ZONE CURRICULUM
# ═══════════════════════════════════════════════════════════════════════

class ZoneCurriculum:
    """
    Go-Explore style zone-based curriculum for Mario training.

    Manages:
      - Zone progression (promotion/demotion)
      - Simulator state checkpoints at zone boundaries
      - Column-novelty bonuses (DiscoRL)
      - Episode statistics and logging
    """

    def __init__(
        self,
        generator: MarioLevelGenerator,
        tier: int = 5,
        novelty_bonus: float = 0.1,
        novelty_decay_visits: int = 50,
        spawn_from_start_ratio: float = 0.5,
        demotion_threshold: float = 0.50,
        demotion_window: int = 50,
        seed: Optional[int] = None,
    ):
        """
        Args:
            generator: Level generator to produce new levels.
            tier: Generator tier (difficulty) for levels.
            novelty_bonus: Reward for visiting a new column for the first time.
            novelty_decay_visits: Visits before a column loses its novelty.
            spawn_from_start_ratio: Fraction of episodes that start from col 0.
            demotion_threshold: Drop back a zone if success < this for demotion_window.
            demotion_window: Number of episodes over which to check demotion.
            seed: Random seed for reproducibility.
        """
        self.generator = generator
        self.tier = tier
        self.novelty_bonus = novelty_bonus
        self.novelty_decay_visits = novelty_decay_visits
        self.spawn_from_start_ratio = spawn_from_start_ratio
        self.demotion_threshold = demotion_threshold
        self.demotion_window = demotion_window
        self.rng = np.random.RandomState(seed)

        # Generate the first level and set up zones
        self._current_level: Optional[MarioSimulator] = None
        self._level_state: Optional[Dict] = None  # saved pristine state
        self._new_level()

        self.zones = make_zones(self._current_level.width)
        self.current_zone_idx = 0

        # Zone boundary checkpoints: zone_id → simulator save state
        self._zone_checkpoints: Dict[int, Dict] = {}

        # Statistics per zone
        self._zone_results: Dict[int, deque] = {
            z.id: deque(maxlen=z.eval_window) for z in self.zones
        }

        # Column novelty tracking (per-episode, reset each episode)
        self._episode_columns_visited: set = set()
        # Global column visit counts (across episodes, for decay)
        self._global_column_visits: Dict[int, int] = {}

        # Logging
        self._total_episodes = 0
        self._promotions: List[Tuple[int, int]] = []  # (episode, zone_id)
        self._demotions: List[Tuple[int, int]] = []
        self._best_col_reached = 0

    @property
    def current_zone(self) -> Zone:
        return self.zones[self.current_zone_idx]

    @property
    def target_col(self) -> int:
        return self.current_zone.target_col

    def _new_level(self):
        """Generate a fresh level and cache its pristine state."""
        for _ in range(10):
            sim = self.generator.generate(tier=self.tier)
            if sim is not None:
                self._current_level = sim
                self._level_state = sim.save()
                return
        raise RuntimeError(f"Failed to generate level at tier {self.tier}")

    # ── Episode lifecycle ─────────────────────────────────────────────

    def get_episode(self) -> Tuple[MarioSimulator, Dict[str, Any]]:
        """
        Get a simulator ready for a new episode.

        Returns:
            sim: MarioSimulator in the correct start state
            info: dict with zone info for logging
        """
        self._total_episodes += 1
        self._episode_columns_visited.clear()

        zone = self.current_zone

        # Decide spawn point: beginning vs. zone checkpoint
        use_checkpoint = (
            self.current_zone_idx > 0
            and self.current_zone_idx - 1 in self._zone_checkpoints
            and self.rng.random() > self.spawn_from_start_ratio
        )

        # Restore level from pristine state
        self._current_level.load(self._level_state)

        if use_checkpoint:
            # Load checkpoint from the previous zone boundary
            prev_zone_id = self.current_zone_idx - 1
            self._current_level.load(self._zone_checkpoints[prev_zone_id])

        info = {
            "zone_id": zone.id,
            "zone_name": zone.name,
            "target_col": zone.target_col,
            "spawned_from_checkpoint": use_checkpoint,
            "episode": self._total_episodes,
        }

        return self._current_level, info

    def report_result(self, final_col: int, won: bool, alive: bool):
        """
        Report the result of an episode. Handles promotion/demotion logic.

        Args:
            final_col: The rightmost column Mario reached.
            won: Whether Mario reached the flag.
            alive: Whether Mario was alive at episode end.
        """
        zone = self.current_zone
        success = final_col >= zone.target_col or won

        self._zone_results[zone.id].append(1 if success else 0)
        self._best_col_reached = max(self._best_col_reached, final_col)

        # Save checkpoint if Mario reached a zone boundary for the first time
        if success and zone.id not in self._zone_checkpoints:
            # Checkpoint at the zone target column position
            # To create a checkpoint: reset the level, walk Mario to zone boundary
            # In practice, we save the current sim state at this point
            self._zone_checkpoints[zone.id] = self._current_level.save()

        # Check promotion (uses full eval window)
        results = self._zone_results[zone.id]
        if len(results) >= zone.eval_window:
            success_rate = sum(results) / len(results)
            if success_rate >= zone.pass_threshold:
                self._promote()

        # Check demotion (uses only the most recent demotion_window results)
        # Re-read current zone since promotion may have changed current_zone_idx
        cur_zone = self.current_zone
        cur_results = self._zone_results[cur_zone.id]
        if len(cur_results) >= self.demotion_window and self.current_zone_idx > 0:
            recent = list(cur_results)[-self.demotion_window:]
            recent_rate = sum(recent) / len(recent)
            if recent_rate < self.demotion_threshold:
                self._demote()

    def _promote(self):
        """Advance to the next zone."""
        if self.current_zone_idx < len(self.zones) - 1:
            self.current_zone_idx += 1
            self._promotions.append((self._total_episodes, self.current_zone_idx))
            # Generate a new level periodically to prevent overfitting
            if len(self._promotions) % 3 == 0:
                self._new_level()
                self.zones = make_zones(self._current_level.width)
                self._zone_checkpoints.clear()
                # Rebuild zone results deques for new zone definitions
                # Keep existing data where zone IDs overlap
                old_results = self._zone_results
                self._zone_results = {
                    z.id: deque(old_results.get(z.id, deque()), maxlen=z.eval_window)
                    for z in self.zones
                }

    def _demote(self):
        """Drop back one zone."""
        if self.current_zone_idx > 0:
            self.current_zone_idx -= 1
            self._demotions.append((self._total_episodes, self.current_zone_idx))
            # Clear results for the zone we dropped from (fresh start)
            old_zone_id = self.current_zone_idx + 1
            self._zone_results[old_zone_id].clear()

    # ── Column Novelty (DiscoRL) ──────────────────────────────────────

    def column_visited(self, col: int) -> float:
        """
        Record a column visit and return the novelty bonus.

        First visit to a column in this episode: full bonus.
        Column visited many times globally: decayed bonus.
        Already visited this episode: zero.

        Args:
            col: Mario's current column.

        Returns:
            Novelty bonus reward (float, 0 to novelty_bonus).
        """
        if col in self._episode_columns_visited:
            return 0.0

        self._episode_columns_visited.add(col)

        # Global visit tracking
        self._global_column_visits[col] = self._global_column_visits.get(col, 0) + 1
        visits = self._global_column_visits[col]

        # Decay: full bonus for first few visits, then linear decay to 0
        if visits >= self.novelty_decay_visits:
            return 0.0

        decay = 1.0 - (visits / self.novelty_decay_visits)
        return self.novelty_bonus * decay

    # ── Death Tracking ────────────────────────────────────────────────

    def record_death(self, col: int, row: int, last_action: int):
        """Record a death for hotspot analysis (Phase 1B)."""
        if not hasattr(self, '_death_log'):
            self._death_log: List[Tuple[int, int, int]] = []
        self._death_log.append((col, row, last_action))

    def get_death_hotspots(self, cluster_radius: int = 3) -> List[Dict]:
        """
        Cluster deaths into geographic hotspots.

        Returns list of dicts: {col, row, count, actions}
        """
        if not hasattr(self, '_death_log') or not self._death_log:
            return []

        # Simple 1D clustering by column
        from collections import Counter
        col_deaths = Counter(d[0] for d in self._death_log)

        # Merge adjacent columns into clusters
        sorted_cols = sorted(col_deaths.keys())
        hotspots = []
        current_cluster = [sorted_cols[0]]

        for c in sorted_cols[1:]:
            if c - current_cluster[-1] <= cluster_radius:
                current_cluster.append(c)
            else:
                # Finish this cluster
                center = current_cluster[len(current_cluster) // 2]
                total = sum(col_deaths[x] for x in current_cluster)
                hotspots.append({
                    "col": center,
                    "col_range": (current_cluster[0], current_cluster[-1]),
                    "count": total,
                })
                current_cluster = [c]

        # Flush last cluster
        if current_cluster:
            center = current_cluster[len(current_cluster) // 2]
            total = sum(col_deaths[x] for x in current_cluster)
            hotspots.append({
                "col": center,
                "col_range": (current_cluster[0], current_cluster[-1]),
                "count": total,
            })

        return sorted(hotspots, key=lambda h: h["count"], reverse=True)

    # ── Stats and Logging ─────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Full curriculum statistics."""
        zone = self.current_zone
        results = self._zone_results[zone.id]
        success_rate = (sum(results) / len(results)) if results else 0.0

        return {
            "current_zone": zone.id,
            "zone_name": zone.name,
            "target_col": zone.target_col,
            "success_rate": round(success_rate, 3),
            "eval_count": len(results),
            "pass_threshold": zone.pass_threshold,
            "total_episodes": self._total_episodes,
            "best_col_reached": self._best_col_reached,
            "promotions": len(self._promotions),
            "demotions": len(self._demotions),
            "checkpoints_saved": len(self._zone_checkpoints),
            "unique_columns_visited": len(self._global_column_visits),
            "death_hotspots": len(self.get_death_hotspots()) if hasattr(self, '_death_log') else 0,
        }

    def zone_summary(self) -> str:
        """Human-readable multi-line zone progress summary."""
        lines = [f"Zone Curriculum — Episode {self._total_episodes}"]
        for z in self.zones:
            results = self._zone_results[z.id]
            rate = (sum(results) / len(results)) if results else 0.0
            arrow = " ◄" if z.id == self.current_zone_idx else ""
            ckpt = " [✓ckpt]" if z.id in self._zone_checkpoints else ""
            bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
            lines.append(
                f"  Zone {z.id} ({z.name:>12s}): "
                f"col≤{z.target_col:>3d}  "
                f"|{bar}| {rate:.0%}/{z.pass_threshold:.0%}"
                f"{ckpt}{arrow}"
            )
        lines.append(f"  Best column reached: {self._best_col_reached}")
        if self._promotions:
            lines.append(f"  Promotions: {[f'ep{ep}→z{z}' for ep, z in self._promotions[-5:]]}")
        if self._demotions:
            lines.append(f"  Demotions:  {[f'ep{ep}→z{z}' for ep, z in self._demotions[-5:]]}")
        return "\n".join(lines)
