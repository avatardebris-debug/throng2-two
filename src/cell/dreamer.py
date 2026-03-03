"""
CellDreamer — Dream coordinator for the ThrongletCell.

Manages when to dream and how to blend dream results with PPO's action selection.
Adapted from throng5's BasalGanglia dream scheduling pattern.

Key idea: dreams are ADVISORY, not mandatory. The PPO policy is the primary
decision-maker. Dream values are blended in with a weight proportional to
the WorldModel's confidence. Early in training, dreams have zero influence.
"""

import numpy as np
from typing import Optional


class CellDreamer:
    """
    Dream coordinator that schedules and blends world model predictions.

    Runs dream_all_actions() periodically and produces action value
    adjustments that bias PPO's action selection toward predicted-good actions.
    """

    def __init__(
        self,
        n_actions: int,
        dream_interval: int = 10,
        dream_depth: int = 3,
        max_blend_weight: float = 0.15,
        warmup_dreams: int = 50,
    ):
        """
        Args:
            n_actions: Number of discrete actions.
            dream_interval: Dream every N steps (not every step).
            dream_depth: How many steps to dream ahead.
            max_blend_weight: Maximum blend weight when WM fully confident.
            warmup_dreams: Number of dreams before dreamer can override PPO.
        """
        self.n_actions = n_actions
        self.dream_interval = dream_interval
        self.dream_depth = dream_depth
        self.max_blend_weight = max_blend_weight
        self.warmup_dreams = warmup_dreams

        self._step = 0
        self._last_dream_values = None
        self._total_dreams = 0
        self._dream_episodes_helped = 0

    def should_dream(self) -> bool:
        """Check if we should dream this step."""
        return self._step % self.dream_interval == 0

    def dream(self, features: np.ndarray, world_model) -> Optional[np.ndarray]:
        """
        Run a dream if conditions are met.

        Args:
            features: Current combined feature vector.
            world_model: CellWorldModel instance.

        Returns:
            Dream action values (n_actions,) or None if no dream.
        """
        self._step += 1

        if not self.should_dream():
            return self._last_dream_values  # Reuse last dream

        if world_model is None or not world_model.is_ready:
            return None

        # Dream all actions from current state
        dream_values = world_model.dream_all_actions(
            features,
            depth=self.dream_depth,
            gamma=0.99,
        )

        self._last_dream_values = dream_values
        self._total_dreams += 1
        return dream_values

    def blend_action(
        self,
        ppo_action: int,
        ppo_log_prob: float,
        dream_values: Optional[np.ndarray],
        wm_confidence: float,
        epsilon: float = 0.1,
    ) -> int:
        """
        Optionally override PPO's action using dream values.

        Only overrides if:
        1. Dream values are available
        2. WM confidence is above threshold
        3. Dream strongly favors a different action than PPO chose

        Args:
            ppo_action: PPO's selected action.
            ppo_log_prob: PPO's log probability for the action.
            dream_values: Dream action values (or None).
            wm_confidence: WorldModel confidence (0-1).
            epsilon: Exploration rate (don't override during exploration).

        Returns:
            Final action (may differ from ppo_action).
        """
        if dream_values is None or wm_confidence < 0.2:
            return ppo_action

        # Warmup: don't override until we've done enough dreams
        if self._total_dreams < self.warmup_dreams:
            return ppo_action

        # Blend weight scales with WM confidence
        blend = min(self.max_blend_weight, wm_confidence * self.max_blend_weight)

        # Dream's best action
        dream_best = int(np.argmax(dream_values))

        if dream_best == ppo_action:
            return ppo_action  # Agreement — proceed

        # Check if dream STRONGLY prefers its action
        dream_range = dream_values.max() - dream_values.min()
        if dream_range < 1e-6:
            return ppo_action  # Dream has no opinion

        dream_advantage = (dream_values[dream_best] - dream_values[ppo_action]) / dream_range

        # Override only if dream advantage is very significant
        if dream_advantage > 0.7 and np.random.random() < blend:
            self._dream_episodes_helped += 1
            return dream_best

        return ppo_action

    def reset(self):
        """Reset between episodes."""
        self._last_dream_values = None
        self._step = 0

    def stats(self) -> dict:
        """Dreamer statistics."""
        return {
            "total_dreams": self._total_dreams,
            "dream_overrides": self._dream_episodes_helped,
            "dream_interval": self.dream_interval,
            "dream_depth": self.dream_depth,
        }
