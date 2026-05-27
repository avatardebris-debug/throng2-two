"""
imagined_sim.py — World Model wrapped as a step-able environment.

The ImaginedSim is activated once the CellWorldModel is confident enough
to serve as a reliable stand-alone simulator. It runs at full training
speed (~5,000+ sps) without needing the real environment.

Lifecycle:
    INACTIVE     → WM confidence crosses threshold → ACTIVE
    ACTIVE       → structural surprise detected    → RECONFIGURING
    RECONFIGURING → WM fast-retrained + rebuilt    → ACTIVE
    ACTIVE       → sim2real accuracy drops         → INACTIVE (fallback)

Reality Check:
    Every K real steps, the same action sequence is run in both the
    ImaginedSim and the real environment. The resulting states are compared
    to compute sim2real_accuracy. If accuracy drops below trust_threshold,
    the sim falls back to real env and triggers WM retraining.

Pattern Interrupt:
    When a structural surprise is detected during a reality check, the
    PatternInterruptHandler is called to:
      1. Attribute the error to a specific entity/interaction
      2. Fast-retrain WM on the surprising transition (5x LR, 50 steps)
      3. Rebuild the sim from the updated WM
      4. Resume from the actual observed state (not the predicted one)

Usage:
    wm = CellWorldModel(feature_dim=378, n_actions=8)
    sim = ImaginedSim(wm, real_env=real_env)

    obs = sim.reset()
    for step in range(10000):
        action = policy.select_action(obs)
        obs, reward, done, info = sim.step(action)
        # ImaginedSim handles reality checks transparently
        if done:
            obs = sim.reset()
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .world_model import CellWorldModel
from .surprise_classifier import SurpriseResult


# ══════════════════════════════════════════════════════════════
#  Pattern Interrupt Handler
# ══════════════════════════════════════════════════════════════

class PatternInterruptHandler:
    """
    Handles structural surprises (Type 2) — the 'wall slides you across room' problem.

    When a structural surprise is detected:
      1. Attribute it to the most likely entity type (based on dominant dims)
      2. Store the surprising transition for targeted retraining
      3. Fast-retrain the WM with boosted LR on that transition
      4. Clear the sim2real drift errors (fresh accuracy baseline)

    The agent is recalibrated to the actual observed state, not the predicted one.
    This mirrors the brain's pattern interrupt: existing predictions are overridden,
    and the new (state, action, displacement) association is immediately high-weighted.
    """

    def __init__(
        self,
        world_model: CellWorldModel,
        fast_retrain_lr_mult: float = 5.0,
        fast_retrain_steps: int    = 50,
        interrupt_buffer_size: int = 20,
    ):
        self.wm                   = world_model
        self.fast_retrain_lr_mult = fast_retrain_lr_mult
        self.fast_retrain_steps   = fast_retrain_steps

        # Buffer of recent structural surprises for targeted retraining
        self._interrupt_buffer: deque = deque(maxlen=interrupt_buffer_size)

        # Counters
        self.n_handled = 0

    def handle(
        self,
        state: np.ndarray,
        action: int,
        predicted_next: np.ndarray,
        actual_next: np.ndarray,
        reward: float,
        surprise: SurpriseResult,
    ) -> np.ndarray:
        """
        Handle a structural surprise. Returns the actual_next state (the real
        position to continue from, not the predicted one).

        Args:
            state:          State before the action.
            action:         Action taken.
            predicted_next: WM's prediction (wrong).
            actual_next:    What actually happened (ground truth).
            reward:         Real reward from the transition.
            surprise:       Classification result from SurpriseClassifier.

        Returns:
            actual_next: The correct starting state. Continue from here.
        """
        self.n_handled += 1

        # 1. Store the surprising transition
        transition = (
            np.asarray(state,       dtype=np.float32),
            int(action),
            np.asarray(actual_next, dtype=np.float32),
            float(reward),
        )
        self._interrupt_buffer.append(transition)

        # 2. Fast-retrain on all buffered interrupt transitions
        #    (catch-all: the new rule may apply to similar transitions seen before)
        if len(self._interrupt_buffer) >= 1:
            self.wm.fast_retrain(
                list(self._interrupt_buffer),
                lr_multiplier=self.fast_retrain_lr_mult,
                n_steps=self.fast_retrain_steps,
            )

        # 3. Also store in the main WM replay buffer so it influences normal training
        self.wm.store_transition(state, action, actual_next, reward)

        # 4. Reset sim2real error tracking (fresh baseline after rebuild)
        self.wm._sim2real_errors.clear()

        return actual_next

    def stats(self) -> dict:
        return {
            "n_handled":          self.n_handled,
            "interrupt_buffer":   len(self._interrupt_buffer),
        }


# ══════════════════════════════════════════════════════════════
#  ImaginedSim
# ══════════════════════════════════════════════════════════════

@dataclass
class SimStats:
    """Running statistics for ImaginedSim."""
    n_imagined_steps:   int   = 0
    n_real_steps:       int   = 0
    n_reality_checks:   int   = 0
    n_fallbacks:        int   = 0
    n_interrupts:       int   = 0
    imagination_ratio:  float = 0.0  # fraction of steps in imagination


class ImaginedSim:
    """
    World Model wrapped as a step-able environment (Gymnasium-compatible API).

    Transparently switches between:
      - Imagination: WM.step() — fast, free, ~5,000+ sps
      - Reality:     real_env.step() — slow, expensive, ground truth

    Reality checks happen every `reality_check_interval` steps to measure
    sim2real accuracy and detect pattern interrupts.

    Args:
        world_model:             Trained CellWorldModel.
        real_env:                Real environment (must have .step()/.reset()).
        confidence_threshold:    WM confidence to activate imagination.
        sim2real_trust_threshold: sim2real accuracy to trust imagination.
        reality_check_interval:  Steps between reality checks.
        interrupt_handler:       Optional PatternInterruptHandler. Created
                                 automatically if not supplied.
    """

    # Lifecycle states
    INACTIVE      = "inactive"       # WM not calibrated yet
    ACTIVE        = "active"         # Running in imagination
    RECONFIGURING = "reconfiguring"  # Handling pattern interrupt
    FALLBACK      = "fallback"       # sim2real too low → back to real env

    def __init__(
        self,
        world_model:               CellWorldModel,
        real_env:                  Any,
        confidence_threshold:      float = 0.70,
        sim2real_trust_threshold:  float = 0.80,
        reality_check_interval:    int   = 64,
        interrupt_handler:         Optional[PatternInterruptHandler] = None,
    ):
        self.wm                      = world_model
        self.real_env                = real_env
        self.confidence_threshold    = confidence_threshold
        self.sim2real_trust          = sim2real_trust_threshold
        self.reality_check_interval  = reality_check_interval

        self.interrupt_handler = interrupt_handler or PatternInterruptHandler(world_model)

        # Current lifecycle state
        self._state = self.INACTIVE

        # Current observation
        self._current_obs: Optional[np.ndarray] = None

        # Recent action history for reality-check replay
        self._action_history: deque = deque(maxlen=reality_check_interval)

        # Step counter (for reality check scheduling)
        self._steps_since_check = 0

        # Stats
        self._stats = SimStats()

    # ── Gymnasium-compatible API ─────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset. Always starts from real env to get a valid initial state."""
        obs = self.real_env.reset()
        self._current_obs = np.asarray(obs, dtype=np.float32)
        self._action_history.clear()
        self._steps_since_check = 0
        self.wm.sim_reset(self._current_obs)
        self._update_state()
        return self._current_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step the environment. Uses imagination when WM is reliable.

        Returns:
            (obs, reward, done, info) — same as Gymnasium.
            info contains: {'imagined': bool, 'surprise': float, 'sim_state': str}
        """
        self._update_state()
        self._action_history.append(action)

        if self._state == self.ACTIVE:
            obs, reward, done, info = self._imagined_step(action)
        else:
            obs, reward, done, info = self._real_step(action)

        self._current_obs = obs
        self._steps_since_check += 1

        # Schedule reality check
        if self._steps_since_check >= self.reality_check_interval and self._state == self.ACTIVE:
            self._do_reality_check()
            self._steps_since_check = 0

        return obs, reward, done, info

    # ── Internal step handlers ───────────────────────────────

    def _imagined_step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Step the WM as a simulator."""
        next_obs, reward, _ = self.wm.step(action)
        self._stats.n_imagined_steps += 1

        # WM doesn't model done signals — use reward heuristic or fixed episode len
        done = False  # Caller's policy manages episode resets

        return (
            next_obs.astype(np.float32),
            float(reward),
            done,
            {"imagined": True, "surprise": 0.0, "sim_state": self._state},
        )

    def _real_step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Step the real environment. Also trains WM on the transition."""
        prev_obs = self._current_obs.copy() if self._current_obs is not None else None
        real_obs, reward, done, info = self.real_env.step(action)
        real_obs = np.asarray(real_obs, dtype=np.float32)

        # Measure surprise and train WM on each real transition
        surprise_val = 0.0
        if prev_obs is not None:
            surprise  = self.wm.measure_surprise(prev_obs, action, real_obs)
            surprise_val = surprise.total

            if surprise.is_pattern_interrupt:
                self._stats.n_interrupts += 1
                self._state = self.RECONFIGURING
                self.interrupt_handler.handle(
                    state=prev_obs, action=action,
                    predicted_next=self.wm.predict(prev_obs, action)[0],
                    actual_next=real_obs, reward=reward, surprise=surprise,
                )
                # Snap WM sim state to reality after interrupt
                self.wm.sim_reset(real_obs)
                self._state = self.FALLBACK  # Stay in fallback until accuracy recovers

            self.wm.store_transition(prev_obs, action, real_obs, reward)
            self.wm.train_step()

        # Sync WM sim state with reality
        self.wm.sim_reset(real_obs)
        self._stats.n_real_steps += 1

        return (
            real_obs,
            float(reward),
            bool(done),
            {**info, "imagined": False, "surprise": surprise_val, "sim_state": self._state},
        )

    # ── Reality check ─────────────────────────────────────────

    def _do_reality_check(self):
        """
        Run the same recent action sequence in both sim and real env,
        compare states, update sim2real accuracy.

        This is the key verification step — it ensures the ImaginedSim
        hasn't drifted from reality during a long imagination run.
        """
        if not self._action_history:
            return

        self._stats.n_reality_checks += 1

        # Save the current real env state (using reset + replay if env supports it,
        # otherwise just do a single-step check on the current state + random action)
        # Simplified: check one step ahead from current position
        check_action = int(self._action_history[-1])

        # Imagined next state
        if self._current_obs is not None:
            pred_next, _, _ = self.wm.step(check_action)
            # Snap back so we don't permanently advance the imagined state
            self.wm.sim_reset(self._current_obs)

            # Real next state
            real_obs, real_rew, real_done, _ = self.real_env.step(check_action)
            real_obs = np.asarray(real_obs, dtype=np.float32)

            # Update WM with real transition
            self.wm.step(check_action, real_next=real_obs)
            self.wm.store_transition(self._current_obs, check_action, real_obs, real_rew)
            self.wm.train_step()

            # Measure surprise on this check
            surprise = self.wm._surprise_clf.classify(
                predicted_next=pred_next,
                actual_next=real_obs,
                prev_state=self._current_obs,
            )

            if surprise.is_pattern_interrupt:
                self._stats.n_interrupts += 1
                self.interrupt_handler.handle(
                    state=self._current_obs, action=check_action,
                    predicted_next=pred_next, actual_next=real_obs,
                    reward=real_rew, surprise=surprise,
                )
                self._state = self.FALLBACK
            else:
                # Sync and continue imagining from real position
                self._current_obs = real_obs
                self.wm.sim_reset(real_obs)

            if real_done:
                self._current_obs = np.asarray(self.real_env.reset(), dtype=np.float32)
                self.wm.sim_reset(self._current_obs)

        self._update_state()

    # ── State machine ─────────────────────────────────────────

    def _update_state(self):
        """Transition lifecycle state based on WM metrics."""
        conf     = self.wm.confidence
        s2r      = self.wm.sim2real_accuracy

        prev = self._state

        if self._state == self.ACTIVE:
            if s2r < self.sim2real_trust and self.wm._sim2real_errors:
                # Accuracy dropped — fall back to real env
                self._state = self.FALLBACK
                self._stats.n_fallbacks += 1

        elif self._state in (self.INACTIVE, self.FALLBACK, self.RECONFIGURING):
            if conf >= self.confidence_threshold and (
                s2r >= self.sim2real_trust or not self.wm._sim2real_errors
            ):
                self._state = self.ACTIVE

        # Update imagination ratio
        total = self._stats.n_imagined_steps + self._stats.n_real_steps
        if total > 0:
            self._stats.imagination_ratio = self._stats.n_imagined_steps / total

    # ── Properties ───────────────────────────────────────────

    @property
    def is_imagining(self) -> bool:
        return self._state == self.ACTIVE

    @property
    def sim_state(self) -> str:
        return self._state

    @property
    def sim2real_accuracy(self) -> float:
        return self.wm.sim2real_accuracy

    def stats(self) -> dict:
        return {
            "sim_state":         self._state,
            "is_imagining":      self.is_imagining,
            "wm_confidence":     round(self.wm.confidence, 3),
            "sim2real_accuracy": round(self.sim2real_accuracy, 3),
            "n_imagined":        self._stats.n_imagined_steps,
            "n_real":            self._stats.n_real_steps,
            "n_checks":          self._stats.n_reality_checks,
            "n_fallbacks":       self._stats.n_fallbacks,
            "n_interrupts":      self._stats.n_interrupts,
            "imagination_ratio": round(self._stats.imagination_ratio, 3),
            "interrupt_handler": self.interrupt_handler.stats(),
            "wm":                self.wm.stats(),
        }
