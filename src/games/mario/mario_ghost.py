"""
mario_ghost.py -- Ghost Racing Self-Play for Mario ASCII.

The ghost is a replay of the agent's own personal best run.
Each episode, the agent races against the ghost:
  - Reward += (agent_x - ghost_x) * ghost_reward_scale
  - If agent finishes before ghost, big bonus
  - Ghost auto-updates to personal best after each win

This creates self-curriculum: as the agent improves, the ghost
gets faster, always pushing the agent to go faster and further.

Usage:
    ghost = GhostRacer(seed=42)

    # During episode:
    ghost.start_race(level_id)
    for step in range(max_steps):
        ghost_x = ghost.get_ghost_position(step)
        ghost_reward = ghost.compute_reward(agent_x, step)
        ...
    ghost.end_race(agent_x_history, won=True)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class GhostReplay:
    """
    A recorded run — stores the agent's x-position at each step.
    """

    __slots__ = ("level_id", "x_positions", "won", "total_steps",
                 "max_x", "avg_speed")

    def __init__(
        self,
        level_id: str,
        x_positions: List[int],
        won: bool = False,
    ):
        self.level_id = level_id
        self.x_positions = list(x_positions)
        self.won = won
        self.total_steps = len(x_positions)
        self.max_x = max(x_positions) if x_positions else 0
        self.avg_speed = self.max_x / max(1, self.total_steps)

    def x_at(self, step: int) -> int:
        """Get ghost's x position at a given step."""
        if step < len(self.x_positions):
            return self.x_positions[step]
        # After ghost finishes, stay at final position
        return self.x_positions[-1] if self.x_positions else 0


class GhostRacer:
    """
    Ghost racing self-play system.

    Maintains a library of "personal best" ghost replays per level/tier.
    During racing, provides shaped reward based on relative position
    to the ghost.

    Reward shaping:
      - ahead_reward: agent is ahead of ghost → positive (proportional to gap)
      - behind_penalty: agent is behind ghost → negative
      - overtake_bonus: agent passes ghost → one-time bonus
      - finish_bonus: agent finishes before ghost → big bonus
    """

    def __init__(
        self,
        ghost_reward_scale: float = 0.1,
        behind_penalty_scale: float = 0.05,
        overtake_bonus: float = 1.0,
        finish_bonus: float = 5.0,
        ghost_speed_multiplier: float = 1.0,
        seed: int = 42,
    ):
        self.ghost_reward_scale = ghost_reward_scale
        self.behind_penalty_scale = behind_penalty_scale
        self.overtake_bonus = overtake_bonus
        self.finish_bonus = finish_bonus
        self.ghost_speed_multiplier = ghost_speed_multiplier

        # Ghost library: tier → best GhostReplay
        self._ghosts: Dict[int, GhostReplay] = {}
        # Per-tier stats
        self._race_count: Dict[int, int] = {}
        self._wins_vs_ghost: Dict[int, int] = {}

        # Current race state
        self._active = False
        self._current_tier = 0
        self._current_ghost: Optional[GhostReplay] = None
        self._agent_x_history: List[int] = []
        self._prev_ahead = False  # Was agent ahead last step?
        self._step = 0

        self._rng = np.random.RandomState(seed)

    # ═══════════════════════════════════════════════════════════
    # RACE LIFECYCLE
    # ═══════════════════════════════════════════════════════════

    def start_race(self, tier: int, level_width: int = 40):
        """Begin a new race against the ghost for this tier."""
        self._active = True
        self._current_tier = tier
        self._agent_x_history = []
        self._prev_ahead = False
        self._step = 0

        # Get ghost for this tier (or create a "slow walker" default)
        if tier in self._ghosts:
            self._current_ghost = self._ghosts[tier]
        else:
            # Default ghost: walks right at 0.5 tiles/step (beatable)
            default_steps = min(level_width * 2, 400)
            default_xs = [min(int(i * 0.5), level_width - 1)
                          for i in range(default_steps)]
            self._current_ghost = GhostReplay(
                level_id=f"tier{tier}_default",
                x_positions=default_xs,
                won=False,
            )

        self._race_count[tier] = self._race_count.get(tier, 0) + 1

    def compute_reward(self, agent_x: int, step: int) -> float:
        """
        Get ghost-based shaped reward for this step.

        Args:
            agent_x: agent's current x position (tile column)
            step: current step number

        Returns:
            shaped reward from ghost comparison
        """
        if not self._active or self._current_ghost is None:
            return 0.0

        self._agent_x_history.append(agent_x)
        self._step = step

        # Get ghost position (optionally sped up)
        ghost_step = int(step * self.ghost_speed_multiplier)
        ghost_x = self._current_ghost.x_at(ghost_step)

        gap = agent_x - ghost_x  # positive = ahead, negative = behind
        reward = 0.0

        if gap > 0:
            # Agent is ahead — reward proportional to gap
            reward += gap * self.ghost_reward_scale
        elif gap < 0:
            # Agent is behind — gentle penalty
            reward += gap * self.behind_penalty_scale

        # Overtake bonus (one-time when passing the ghost)
        is_ahead = (gap > 0)
        if is_ahead and not self._prev_ahead:
            reward += self.overtake_bonus
        self._prev_ahead = is_ahead

        return reward

    def end_race(
        self,
        won: bool,
        final_x: int,
        total_steps: int,
    ) -> Dict[str, float]:
        """
        End the race and potentially update the ghost.

        Returns:
            dict with race stats and any bonus reward
        """
        if not self._active:
            return {"bonus": 0.0}

        self._active = False
        tier = self._current_tier
        ghost = self._current_ghost

        bonus = 0.0
        beat_ghost = False

        if ghost is not None:
            # Did agent beat the ghost?
            ghost_final_x = ghost.x_at(total_steps)
            beat_ghost = final_x > ghost_final_x

            # Finish bonus if agent won AND ghost didn't
            if won and not ghost.won:
                bonus += self.finish_bonus
            elif won and ghost.won and total_steps < ghost.total_steps:
                # Won faster than ghost
                bonus += self.finish_bonus * 0.5

            if beat_ghost:
                self._wins_vs_ghost[tier] = self._wins_vs_ghost.get(tier, 0) + 1

        # Update ghost if this run was better
        should_update = False
        if tier not in self._ghosts:
            should_update = True
        elif won and not self._ghosts[tier].won:
            should_update = True
        elif won and self._ghosts[tier].won and total_steps < self._ghosts[tier].total_steps:
            should_update = True
        elif final_x > self._ghosts[tier].max_x:
            should_update = True

        if should_update and self._agent_x_history:
            self._ghosts[tier] = GhostReplay(
                level_id=f"tier{tier}_ep{self._race_count.get(tier, 0)}",
                x_positions=self._agent_x_history,
                won=won,
            )

        return {
            "bonus": bonus,
            "beat_ghost": beat_ghost,
            "ghost_final_x": ghost.x_at(total_steps) if ghost else 0,
            "agent_final_x": final_x,
            "gap": final_x - (ghost.x_at(total_steps) if ghost else 0),
            "ghost_updated": should_update,
            "races_at_tier": self._race_count.get(tier, 0),
            "ghost_wins_at_tier": self._wins_vs_ghost.get(tier, 0),
        }

    # ═══════════════════════════════════════════════════════════
    # GHOST INFO
    # ═══════════════════════════════════════════════════════════

    def get_ghost_position(self, step: int) -> int:
        """Get ghost's x position at current step."""
        if self._current_ghost is None:
            return 0
        ghost_step = int(step * self.ghost_speed_multiplier)
        return self._current_ghost.x_at(ghost_step)

    def has_ghost(self, tier: int) -> bool:
        """Is there a recorded ghost for this tier?"""
        return tier in self._ghosts

    def ghost_stats(self, tier: int) -> dict:
        """Get stats about the ghost for a tier."""
        if tier not in self._ghosts:
            return {"exists": False}
        g = self._ghosts[tier]
        return {
            "exists": True,
            "max_x": g.max_x,
            "won": g.won,
            "steps": g.total_steps,
            "avg_speed": g.avg_speed,
            "races": self._race_count.get(tier, 0),
            "ghost_beaten": self._wins_vs_ghost.get(tier, 0),
        }

    def all_stats(self) -> dict:
        """Summary stats for all tiers."""
        return {
            tier: self.ghost_stats(tier)
            for tier in sorted(set(list(self._ghosts.keys()) +
                                   list(self._race_count.keys())))
        }
