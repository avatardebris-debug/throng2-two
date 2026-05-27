"""
exploration_controller.py — Hypothesis-directed Playground/Test mode control.

Core insight:
    Greedy surprise maximization is inefficient — it wastes budget revisiting
    familiar-but-noisy states. The right objective is to pick actions that
    *maximally disambiguate* between competing world model hypotheses, i.e.,
    choose the action that teaches the most per step spent.

    Objective: minimize steps_to_global_surprise_zero
    Not:       minimize total_surprise (passive accuracy)

Two modes:

    PLAYGROUND — active learning phase:
        Goal: reduce WM uncertainty as efficiently as possible.
        Action selection: argmax(information_gain(state, action))
        Sub-modes:
          TARGETED_EXPLORE  — worst-understood entity is known → go interact with it
          BROAD_EXPLORE     — no specific target → pick highest info-gain action

    TEST — exploitation + commitment phase:
        Goal: complete the task using best available knowledge.
        Action selection: policy's best action (PPO / Q-values)
        Consequence awareness: irreversible actions flagged before commit
        Post-interrupt recalibration: if Type 2 surprise occurs in TEST,
          brief RECALIBRATE sub-mode before resuming task.

Mode transitions:

    PLAYGROUND → TEST when:
        - All known entity types have confidence > entity_mastery_threshold
        - step_budget_remaining < critical_budget_fraction * max_budget
        (forced: running out of steps)

    TEST → PLAYGROUND when:
        - Structural surprise detected (new unknown entity / physics rule)
        - step_budget_remaining is healthy AND entity confidence dropped

    Either → RECALIBRATE when:
        - Pattern interrupt received (Type 2, is_pattern_interrupt=True)
        → brief recovery: dedicate N steps to targeted exploration near the
          entity that caused the interrupt

Budget awareness:
    In ARC-AGI3 (and similar games), a step counter in the HUD represents
    a hard constraint. The controller uses `step_budget_remaining` to modulate
    how aggressively it explores vs. exploits:

        budget_fraction = step_budget_remaining / initial_budget
        playground_intensity = sigmoid(budget_fraction - critical_threshold)

    Near 0 budget: forced TEST regardless of WM confidence.
    Full budget: free to explore.

Usage:
    ctrl = ExplorationController(world_model)

    # Each step:
    mode = ctrl.mode(step_budget_remaining=budget)
    if mode == 'playground':
        action = ctrl.select_action_playground(state)
    else:
        action = policy.select_action(state)          # PPO / Q-learning

    # After observing result:
    ctrl.observe(state, action, next_state, reward, surprise_result)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.cell.world_model import CellWorldModel
from src.cell.surprise_classifier import SurpriseResult


# ══════════════════════════════════════════════════════════════
#  Mode constants
# ══════════════════════════════════════════════════════════════

PLAYGROUND    = "playground"
TEST          = "test"
RECALIBRATE   = "recalibrate"   # Brief recovery after pattern interrupt


# ══════════════════════════════════════════════════════════════
#  ExplorationController
# ══════════════════════════════════════════════════════════════

@dataclass
class ExplorationStats:
    n_playground:   int   = 0
    n_test:         int   = 0
    n_recalibrate:  int   = 0
    n_forced_test:  int   = 0    # forced by budget pressure
    n_mode_switches: int  = 0


class ExplorationController:
    """
    Controls Playground / Test mode switching for the Imagination Engine.

    Args:
        world_model:              Trained CellWorldModel — used for WM confidence,
                                  information_gain, and per_entity_confidence.
        n_actions:                Size of the action space.
        entity_mastery_threshold: Per-entity confidence required to leave playground.
                                  Default 0.80 (80% prediction accuracy per entity type).
        overall_confidence_min:   Global WM confidence to enter test mode.
        critical_budget_fraction: Budget fraction below which mode is forced TEST.
                                  e.g. 0.15 = switch to TEST at 15% budget remaining.
        recalibrate_steps:        How many steps to spend in RECALIBRATE after interrupt.
        info_gain_samples:        MC samples for information_gain() approximation.
        explore_epsilon:          ε-greedy component in playground (avoids pure exploitation
                                  even of information gain — keeps action diversity).
    """

    def __init__(
        self,
        world_model:                CellWorldModel,
        n_actions:                  int,
        entity_mastery_threshold:   float = 0.80,
        overall_confidence_min:     float = 0.70,
        critical_budget_fraction:   float = 0.15,
        recalibrate_steps:          int   = 20,
        info_gain_samples:          int   = 3,
        explore_epsilon:            float = 0.10,
    ):
        self.wm                       = world_model
        self.n_actions                = n_actions
        self.entity_mastery_threshold = entity_mastery_threshold
        self.overall_confidence_min   = overall_confidence_min
        self.critical_budget_fraction = critical_budget_fraction
        self.recalibrate_steps        = recalibrate_steps
        self.info_gain_samples        = info_gain_samples
        self.explore_epsilon          = explore_epsilon

        # Current mode
        self._mode            = PLAYGROUND
        self._prev_mode       = PLAYGROUND

        # Recalibrate countdown
        self._recalibrate_remaining = 0
        # Entity to target during recalibration
        self._recalibrate_target: Optional[str] = None

        # Budget tracking
        self._initial_budget:   Optional[int]  = None
        self._current_budget:   Optional[int]  = None

        # Recent information-gain estimates (cache, updated lazily)
        self._ig_cache:         Optional[np.ndarray] = None
        self._ig_cache_state:   Optional[np.ndarray] = None

        # History for diagnostics
        self._mode_history:     deque = deque(maxlen=200)
        self._ig_history:       deque = deque(maxlen=100)   # (step, action, ig)

        # Stats
        self._stats = ExplorationStats()

    # ── Primary interface ────────────────────────────────────

    def mode(
        self,
        step_budget_remaining: Optional[int] = None,
        structural_surprise:   bool          = False,
        interrupt_entity:      Optional[str] = None,
    ) -> str:
        """
        Determine and return the current exploration mode.

        Args:
            step_budget_remaining: Remaining step budget from HUD (None = unlimited).
            structural_surprise:   True if a structural surprise was just detected.
            interrupt_entity:      Entity tag that caused the interrupt (if any).

        Returns:
            One of: 'playground', 'test', 'recalibrate'
        """
        # 1. Update budget
        if step_budget_remaining is not None:
            if self._initial_budget is None:
                self._initial_budget = step_budget_remaining
            self._current_budget = step_budget_remaining

        # 2. Handle pattern interrupt (highest priority)
        if structural_surprise:
            self._enter_recalibrate(interrupt_entity)

        # 3. Recalibrate countdown
        if self._mode == RECALIBRATE:
            self._recalibrate_remaining -= 1
            if self._recalibrate_remaining <= 0:
                self._mode = self._choose_mode_from_wm()
                self._recalibrate_target = None

        # 4. Budget forced TEST
        elif self._is_budget_critical():
            if self._mode != TEST:
                self._switch_mode(TEST, reason="budget_critical")
                self._stats.n_forced_test += 1

        # 5. Normal mode selection from WM state
        else:
            desired = self._choose_mode_from_wm()
            if desired != self._mode:
                self._switch_mode(desired, reason="wm_state")

        # 6. Track
        self._mode_history.append(self._mode)
        if self._mode == PLAYGROUND:
            self._stats.n_playground += 1
        elif self._mode == TEST:
            self._stats.n_test += 1
        else:
            self._stats.n_recalibrate += 1

        return self._mode

    def select_action_playground(self, state: np.ndarray) -> int:
        """
        Select action for PLAYGROUND mode: maximize information gain.

        Strategy:
          1. If in RECALIBRATE and there's a target entity, bias toward
             actions that are likely to interact with that entity type.
             (Currently: random action — entity-specific routing TBD with
              EntityDetector integration).
          2. Otherwise: argmax(information_gain) with ε-greedy diversity.

        Args:
            state: Current observation / feature vector.

        Returns:
            Selected action index.
        """
        # ε-greedy: occasionally random to maintain action diversity
        if np.random.random() < self.explore_epsilon:
            return int(np.random.randint(self.n_actions))

        # Use cached IG if state hasn't changed much
        if self._ig_cache is not None and self._ig_cache_state is not None:
            state_delta = float(np.mean(np.abs(state - self._ig_cache_state)))
            if state_delta < 0.05:
                best = int(np.argmax(self._ig_cache))
                self._ig_history.append(self._ig_cache[best])
                return best

        # Compute information gain for all actions
        ig = self.wm.information_gains_all_actions(state, n_samples=self.info_gain_samples)

        # Cache result
        self._ig_cache       = ig
        self._ig_cache_state = state.copy()

        best_action = int(np.argmax(ig))
        self._ig_history.append(float(ig[best_action]))
        return best_action

    def observe(
        self,
        state:       np.ndarray,
        action:      int,
        next_state:  np.ndarray,
        reward:      float,
        surprise:    Optional[SurpriseResult] = None,
    ):
        """
        Update controller with the result of a step.

        Call this after every env.step() to keep internal state consistent.
        Triggers mode transitions based on observed surprise.

        Args:
            state, action, next_state, reward: Standard transition.
            surprise: SurpriseResult from wm.measure_surprise() (optional).
        """
        # Invalidate IG cache on state change
        self._ig_cache = None

        if surprise is not None and surprise.is_pattern_interrupt:
            # Will be handled on next mode() call
            self._pending_interrupt       = True
            self._pending_interrupt_entity = surprise.entity_tag

    def set_budget(self, initial_budget: int):
        """
        Set the initial step budget (call once at episode start).
        Derived from HUD step counter.
        """
        self._initial_budget = initial_budget
        self._current_budget = initial_budget

    def update_budget(self, remaining: int):
        """Update current remaining budget from HUD."""
        self._current_budget = remaining

    # ── Properties ───────────────────────────────────────────

    @property
    def current_mode(self) -> str:
        return self._mode

    @property
    def budget_fraction(self) -> float:
        """Remaining budget as fraction of initial (1.0=full, 0.0=empty)."""
        if self._initial_budget is None or self._initial_budget == 0:
            return 1.0
        return max(0.0, (self._current_budget or 0) / self._initial_budget)

    @property
    def playground_intensity(self) -> float:
        """
        0-1 measure of how freely the agent can explore.
        Approaches 0 as budget runs low.
        """
        bf = self.budget_fraction
        critical = self.critical_budget_fraction
        if bf <= critical:
            return 0.0
        return (bf - critical) / (1.0 - critical)

    # ── Internal helpers ─────────────────────────────────────

    def _choose_mode_from_wm(self) -> str:
        """Choose mode based purely on WM state (ignoring budget)."""
        conf = self.wm.confidence

        # Need minimum WM confidence before TEST is allowed
        if conf < self.overall_confidence_min:
            return PLAYGROUND

        # Check per-entity mastery
        per_entity = self.wm.per_entity_confidence
        if per_entity:
            worst_conf = min(per_entity.values())
            if worst_conf < self.entity_mastery_threshold:
                return PLAYGROUND      # Still entities we don't understand well

        # WM is calibrated and all entities are well-understood
        return TEST

    def _is_budget_critical(self) -> bool:
        """True if remaining budget is below the critical fraction."""
        if self._initial_budget is None or self._current_budget is None:
            return False
        return self.budget_fraction <= self.critical_budget_fraction

    def _enter_recalibrate(self, entity: Optional[str] = None):
        """Enter RECALIBRATE mode after a pattern interrupt."""
        if self._mode != RECALIBRATE:
            self._switch_mode(RECALIBRATE, reason="pattern_interrupt")
        self._recalibrate_remaining = self.recalibrate_steps
        self._recalibrate_target    = entity

    def _switch_mode(self, new_mode: str, reason: str = ""):
        self._prev_mode = self._mode
        self._mode      = new_mode
        self._stats.n_mode_switches += 1

    def stats(self) -> dict:
        total = self._stats.n_playground + self._stats.n_test + self._stats.n_recalibrate
        mode_dist = {
            "playground":  self._stats.n_playground  / max(1, total),
            "test":        self._stats.n_test         / max(1, total),
            "recalibrate": self._stats.n_recalibrate  / max(1, total),
        }
        return {
            "current_mode":           self._mode,
            "mode_distribution":      {k: round(v, 3) for k, v in mode_dist.items()},
            "n_mode_switches":        self._stats.n_mode_switches,
            "n_forced_test":          self._stats.n_forced_test,
            "budget_fraction":        round(self.budget_fraction, 3),
            "playground_intensity":   round(self.playground_intensity, 3),
            "avg_info_gain":          round(float(np.mean(self._ig_history)), 6)
                                      if self._ig_history else 0.0,
            "wm_confidence":          round(self.wm.confidence, 3),
            "wm_entity_confidence":   {k: round(v, 3) for k, v in self.wm.per_entity_confidence.items()},
            "worst_entity":           self.wm.worst_understood_entity(),
            "recalibrate_remaining":  self._recalibrate_remaining,
            "recalibrate_target":     self._recalibrate_target,
        }
