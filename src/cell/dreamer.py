"""
CellDreamer — Dream coordinator for the ThrongletCell.

Manages when to dream and how to blend dream results with PPO's action selection.
Adapted from throng5's BasalGanglia dream scheduling pattern.

Key idea: dreams are ADVISORY, not mandatory. The PPO policy is the primary
decision-maker. Dream values are blended in with a weight proportional to
the WorldModel's confidence. Early in training, dreams have zero influence.

Multi-game extension (Phase 2D):
  - set_game_id(game_id) switches the dreamer to condition on a specific game.
  - dream() passes game_id to world_model.dream_all_actions() so per-game heads
    generate game-appropriate predictions.
  - dream_log captures (dream_action, actual_action, was_correct) for offline
    accuracy analysis.
"""

import numpy as np
from collections import deque
from typing import Optional

from .world_model.protocol import (
    dream_all_actions_for_game,
    has_horizon_dreaming,
    is_ready_for_game,
)


class CellDreamer:
    """
    Dream coordinator that schedules and blends world model predictions.

    Runs dream_all_actions() periodically and produces action value
    adjustments that bias PPO's action selection toward predicted-good actions.

    Multi-game: call set_game_id() when switching environments.
    """

    def __init__(
        self,
        n_actions: int,
        dream_interval: int = 10,
        dream_depth: int = 3,
        max_blend_weight: float = 0.15,
        warmup_dreams: int = 50,
        log_dreams: bool = False,
        horizon_interval: int = 20,
        horizon_alpha: float = 0.4,
    ):
        """
        Args:
            n_actions: Number of discrete actions.
            dream_interval: Dream every N steps (not every step).
            dream_depth: How many steps to dream ahead (fast path).
            max_blend_weight: Maximum blend weight when WM fully confident.
            warmup_dreams: Number of dreams before dreamer can override PPO.
            log_dreams: If True, record dream decisions for accuracy analysis.
            horizon_interval: Refresh the slow (N-step) path every K steps.
                              Between refreshes the cached prediction is reused.
                              Default 20 (every 20 env steps).
            horizon_alpha: Weight of the slow path in the combined dream value.
                           0 = fast only, 1 = slow only, 0.4 = 40% slow.
                           Applied only when the slow path is available and fresh.
        """
        self.n_actions = n_actions
        self.dream_interval = dream_interval
        self.dream_depth = dream_depth
        self.max_blend_weight = max_blend_weight
        self.warmup_dreams = warmup_dreams
        self.log_dreams = log_dreams

        # ── Multi-timescale slow path ──────────────────────
        # Refresh the slow (N-step horizon) prediction every K steps.
        # Between refreshes, reuse the cached values — free lookahead.
        self.horizon_interval = horizon_interval  # refresh slow path every K steps
        self.horizon_alpha = horizon_alpha        # weight for slow path [0, 1]

        self._step = 0
        self._last_dream_values = None
        self._last_dream_step = -1          # step when last fast dream fired
        self._total_dreams = 0
        self._dream_episodes_helped = 0
        self._game_id: int = 0  # Current game (default: 0 = Mario)

        # Slow-path state (horizon head cache)
        self._slow_values: Optional[np.ndarray] = None
        self._slow_values_step: int = -1    # step when slow path last ran
        self._current_horizon_n: int = 0    # adaptive N reported at last slow refresh


        # Dream log: list of {dream_action, ppo_action, final_action, game_id}
        self._dream_log: deque = deque(maxlen=1000)

    def set_game_id(self, game_id: int):
        """
        Switch the dreamer to condition on a specific game.

        Call this when the training loop switches between environments.
        The next dream will use the game_id-specific world model head.

        Args:
            game_id: Integer game ID (from EncoderRegistry or GAME_CONFIGS).
        """
        self._game_id = game_id

    def get_game_id(self) -> int:
        """Return the current game ID."""
        return self._game_id

    def should_dream(self) -> bool:
        """Check if we should dream this step."""
        return self._step % self.dream_interval == 0

    def dream(self, features: np.ndarray, world_model) -> Optional[np.ndarray]:
        """
        Run a dream if conditions are met.

        Multi-timescale version:
          - FAST path: dream_all_actions(depth=1) every dream_interval steps
          - SLOW path: dream_horizon() every horizon_interval steps, cached between

        Combined: (1 - horizon_alpha) * fast + horizon_alpha * slow

        Args:
            features: Current combined feature vector.
            world_model: CellWorldModel or MultiGameWorldModel instance.

        Returns:
            Dream action values (n_actions,) or None if no dream.
        """
        self._step += 1

        # Return stale fast-dream values if within TTL
        stale_ttl = self.dream_interval * 2
        if not self.should_dream():
            if (self._last_dream_values is not None
                    and (self._step - self._last_dream_step) <= stale_ttl):
                return self._last_dream_values
            return None  # Stale beyond TTL

        if not is_ready_for_game(world_model, self._game_id):
            return None

        fast_values = dream_all_actions_for_game(
            world_model,
            features,
            depth=self.dream_depth,
            game_id=self._game_id,
        )

        slow_ttl = self.horizon_interval * 3
        if (
            self.horizon_alpha > 0
            and has_horizon_dreaming(world_model)
            and self._step % self.horizon_interval == 0
        ):
            self._slow_values = world_model.dream_horizon(
                features, game_id=self._game_id,
            )
            self._slow_values_step = self._step
            self._current_horizon_n = world_model.adaptive_horizon_n(self._game_id)

        # Blend fast + slow
        slow_fresh = (
            self._slow_values is not None
            and (self._step - self._slow_values_step) <= slow_ttl
            and np.any(self._slow_values != 0)   # skip if horizon head returned zeros (not ready)
        )
        if slow_fresh:
            dream_values = (
                (1.0 - self.horizon_alpha) * fast_values
                + self.horizon_alpha * self._slow_values
            )
        else:
            dream_values = fast_values

        self._last_dream_values = dream_values
        self._last_dream_step = self._step
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
        if dream_values is None:
            self._log_dream(None, ppo_action, ppo_action)
            return ppo_action

        # Early-training minimum blend: even with low confidence, contribute a
        # tiny advisory signal once the world model has trained enough steps.
        # This avoids the common failure mode where confidence=0 keeps dreams
        # permanently disabled while the model is already making useful predictions.
        effective_confidence = wm_confidence
        if effective_confidence < 0.2 and self._total_dreams >= 20:
            # Small fixed advisory weight (5% of max blend) as warmup floor
            effective_confidence = 0.05 / self.max_blend_weight

        if effective_confidence < 0.2:
            self._log_dream(None, ppo_action, ppo_action)
            return ppo_action

        # Warmup: don't override until we've done enough dreams
        if self._total_dreams < self.warmup_dreams:
            self._log_dream(int(np.argmax(dream_values)), ppo_action, ppo_action)
            return ppo_action

        # Blend weight scales with WM confidence
        blend = min(self.max_blend_weight, wm_confidence * self.max_blend_weight)

        # Dream's best action
        dream_best = int(np.argmax(dream_values))

        if dream_best == ppo_action:
            self._log_dream(dream_best, ppo_action, ppo_action)
            return ppo_action  # Agreement — proceed

        # Check if dream STRONGLY prefers its action
        dream_range = dream_values.max() - dream_values.min()
        if dream_range < 1e-6:
            self._log_dream(dream_best, ppo_action, ppo_action)
            return ppo_action  # Dream has no opinion

        dream_advantage = (dream_values[dream_best] - dream_values[ppo_action]) / dream_range

        # Override only if dream advantage is very significant
        if dream_advantage > 0.7 and np.random.random() < blend:
            self._dream_episodes_helped += 1
            self._log_dream(dream_best, ppo_action, dream_best)
            return dream_best

        self._log_dream(dream_best, ppo_action, ppo_action)
        return ppo_action

    def _log_dream(
        self,
        dream_action: Optional[int],
        ppo_action: int,
        final_action: int,
    ):
        """Record dream decision for accuracy analysis."""
        if self.log_dreams and dream_action is not None:
            self._dream_log.append({
                "dream_action": dream_action,
                "ppo_action": ppo_action,
                "final_action": final_action,
                "overrode": dream_action != ppo_action and final_action == dream_action,
                "game_id": self._game_id,
            })

    def dream_accuracy(self, last_n: int = 100) -> float:
        """
        Fraction of dreams where dream_action == ppo_action (agreement rate).
        Use as a proxy for world model quality: high agreement = good model.
        """
        log = list(self._dream_log)[-last_n:]
        if not log:
            return 0.0
        agreements = sum(1 for d in log if d["dream_action"] == d["ppo_action"])
        return agreements / len(log)

    def reset(self):
        """Reset between episodes."""
        self._last_dream_values = None
        self._last_dream_step = -1
        self._slow_values = None
        self._slow_values_step = -1
        self._step = 0

    def stats(self) -> dict:
        """Dreamer statistics."""
        return {
            "total_dreams": self._total_dreams,
            "dream_overrides": self._dream_episodes_helped,
            "dream_interval": self.dream_interval,
            "dream_depth": self.dream_depth,
            "horizon_interval": self.horizon_interval,
            "horizon_alpha": self.horizon_alpha,
            "slow_path_fresh": self._slow_values is not None,
            "slow_path_age": self._step - self._slow_values_step if self._slow_values is not None else -1,
            "current_game_id": self._game_id,
            "dream_log_size": len(self._dream_log),
            "dream_agreement_rate": round(self.dream_accuracy(), 3),
        }

    def slow_dream_stats(self) -> dict:
        """Slow-path specific diagnostics."""
        slow_ttl = self.horizon_interval * 3
        age = self._step - self._slow_values_step if self._slow_values_step >= 0 else -1
        return {
            "horizon_interval": self.horizon_interval,
            "horizon_alpha": self.horizon_alpha,
            "slow_values_available": self._slow_values is not None,
            "slow_values_age_steps": age,
            "slow_values_fresh": age >= 0 and age <= slow_ttl,
            "slow_values_ttl": slow_ttl,
        }

    @staticmethod
    def guided_training_action(
        policy_action: int,
        features: np.ndarray,
        world_model,
        game_id: int,
        n_actions: int,
        episode_step: int,
        *,
        dream_eps_start: float = 0.5,
        dream_eps_end: float = 0.05,
        dream_eps_decay: float = 0.003,
        depth: int = 1,
    ) -> int:
        """
        Cross-game training policy: epsilon-decay override with WM argmax dream.

        Used by examples/cross_game_training.py so dream logic lives in one place.
        """
        if world_model is None or not is_ready_for_game(world_model, game_id):
            return policy_action

        dream_eps = max(dream_eps_end, dream_eps_start - episode_step * dream_eps_decay)
        if np.random.random() <= dream_eps:
            return policy_action

        dream_vals = dream_all_actions_for_game(
            world_model, features, depth=depth, game_id=game_id
        )
        return int(np.argmax(dream_vals[:n_actions]))

