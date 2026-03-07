"""
mario_selfplay.py -- Dual-Rollout Self-Play for Mario ASCII.

Two runs of the same agent on the same level, competing head-to-head.
The winner (farther progress) gets bonus reward, the loser gets penalty.
Both rollouts are used for training — doubling sample efficiency.

This creates automatic curriculum: the agent always has a ~50/50
opponent (itself), so it's always pushed to improve. No ghost recording
or replay needed — the competition is live.

Usage:
    racer = DualRacer()

    # Generates shaped rewards for both rollouts:
    results = racer.race(agent, adapter, curriculum, max_steps=400)
    # results has both rollouts' experiences for training
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mario_adapter import MarioAdapter
from .mario_curriculum import MarioCurriculum


class RolloutData:
    """
    Stores one rollout's experience for training.
    """
    __slots__ = ("obs", "actions", "rewards", "dones", "next_obs",
                 "final_x", "won", "steps", "ep_reward")

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.next_obs: List[np.ndarray] = []
        self.final_x: int = 0
        self.won: bool = False
        self.steps: int = 0
        self.ep_reward: float = 0.0


class DualRacer:
    """
    Dual-rollout self-play racing.

    Each race:
      1. Generate one level
      2. Run the agent on it TWICE (independent rollouts)
      3. Compare results: winner gets bonus, loser gets penalty
      4. Both rollouts are fed to the agent for training

    The agent races itself — always ~50/50, always improving.
    """

    def __init__(
        self,
        win_bonus: float = 3.0,
        lose_penalty: float = -1.0,
        margin_scale: float = 0.2,
        speed_bonus: float = 1.0,
    ):
        """
        Args:
            win_bonus: Reward for the rollout that went farther
            lose_penalty: Penalty for the rollout that went less far
            margin_scale: Extra reward per tile of margin (winner - loser)
            speed_bonus: Bonus for winning AND finishing faster
        """
        self.win_bonus = win_bonus
        self.lose_penalty = lose_penalty
        self.margin_scale = margin_scale
        self.speed_bonus = speed_bonus

        # Stats
        self.total_races = 0
        self.ties = 0
        self.avg_margin = 0.0

    def race(
        self,
        agent,
        adapter: MarioAdapter,
        level,
        max_steps: int = 400,
    ) -> Tuple[RolloutData, RolloutData, Dict[str, Any]]:
        """
        Run two rollouts on the same level and compare.

        Args:
            agent: the RL agent (must have step() and reset())
            adapter: MarioAdapter instance
            level: a MarioSimulator level (will be copied for rollout 2)
            max_steps: max steps per rollout

        Returns:
            rollout_a: experience from first run
            rollout_b: experience from second run
            race_info: dict with comparison results
        """
        import copy

        # Deep copy the level so both rollouts start from identical state
        level_a = level
        level_b = copy.deepcopy(level)

        # Run rollout A
        rollout_a = self._run_rollout(agent, adapter, level_a, max_steps)

        # Run rollout B (same agent, same level layout, different random actions)
        rollout_b = self._run_rollout(agent, adapter, level_b, max_steps)

        # Compare and assign bonus/penalty
        race_info = self._compare_and_shape(rollout_a, rollout_b)

        self.total_races += 1
        return rollout_a, rollout_b, race_info

    def _run_rollout(
        self,
        agent,
        adapter: MarioAdapter,
        level,
        max_steps: int,
    ) -> RolloutData:
        """Run one complete rollout, collecting experience."""
        rollout = RolloutData()

        obs = adapter.reset(level)
        agent.reset()

        for step in range(max_steps):
            action = agent.step(obs)
            next_obs, reward, done, info = adapter.step(action)

            rollout.obs.append(obs)
            rollout.actions.append(action)
            rollout.rewards.append(reward)
            rollout.dones.append(done)
            rollout.next_obs.append(next_obs)
            rollout.ep_reward += reward

            obs = next_obs
            if done:
                break

        rollout.final_x = level.max_x_reached
        rollout.won = level.won
        rollout.steps = step + 1

        return rollout

    def _compare_and_shape(
        self,
        a: RolloutData,
        b: RolloutData,
    ) -> Dict[str, Any]:
        """Compare two rollouts and add shaped rewards."""

        margin = a.final_x - b.final_x  # positive = A went farther

        if margin > 0:
            # A won
            a_bonus = self.win_bonus + abs(margin) * self.margin_scale
            b_bonus = self.lose_penalty
            winner = "A"
        elif margin < 0:
            # B won
            a_bonus = self.lose_penalty
            b_bonus = self.win_bonus + abs(margin) * self.margin_scale
            winner = "B"
        else:
            # Tie — small bonus if both won, nothing otherwise
            tie_bonus = 0.5 if (a.won and b.won) else 0.0
            a_bonus = tie_bonus
            b_bonus = tie_bonus
            winner = "tie"
            self.ties += 1

        # Speed bonus: if winner also won the level, extra reward
        if winner == "A" and a.won:
            a_bonus += self.speed_bonus
        elif winner == "B" and b.won:
            b_bonus += self.speed_bonus

        # Add bonuses to the LAST step reward of each rollout
        if a.rewards:
            a.rewards[-1] += a_bonus
            a.ep_reward += a_bonus
        if b.rewards:
            b.rewards[-1] += b_bonus
            b.ep_reward += b_bonus

        # Track moving average of margin
        self.avg_margin = 0.95 * self.avg_margin + 0.05 * abs(margin)

        return {
            "winner": winner,
            "margin": margin,
            "a_final_x": a.final_x,
            "b_final_x": b.final_x,
            "a_won": a.won,
            "b_won": b.won,
            "a_bonus": a_bonus,
            "b_bonus": b_bonus,
            "a_steps": a.steps,
            "b_steps": b.steps,
            "avg_margin": self.avg_margin,
        }

    def stats(self) -> Dict[str, Any]:
        """Racing stats."""
        return {
            "total_races": self.total_races,
            "ties": self.ties,
            "tie_rate": self.ties / max(1, self.total_races),
            "avg_margin": round(self.avg_margin, 1),
        }
