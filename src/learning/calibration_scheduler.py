"""
calibration_scheduler.py — Async WM calibration against reality.

The Imagination Engine trains the policy at 20k+ sps in z-space.
But the WM can drift from reality silently. The CalibrationScheduler
runs periodic short real-env episodes IN THE BACKGROUND, compares
the transitions to WM predictions, and updates the WM if needed.

Key design: NEVER BLOCKS THE TRAINING LOOP.

    Imagination runs at full speed (main thread / main process).
    Calibration runs in a background thread.
    WM weight updates happen via a thread-safe queue.
    The training loop doesn't wait for calibration — it just
    picks up improved WM weights whenever they're ready.

Architecture:

    ┌─── Main Thread ──────────────────────┐
    │  VectorizedImaginedEnv.step()        │
    │  policy.forward_batch()             │
    │  ppo.update()                       │
    │  wm = check_for_updated_wm()  ← ─ ─│─ ─ ─ ┐
    └──────────────────────────────────────┘      │
                                                   │ (updated weights)
    ┌─── Background Thread ────────────────┐      │
    │  real_env.step() × 64-128 steps     │      │
    │  encode transitions to z            │      │
    │  compare z to WM predictions        │      │
    │  escalate through Tier 1/2/3        │      │
    │  if surprise: fast_retrain WM ──── ─│─ ─ ─ ┘
    │  sleep(calibration_interval)        │
    └──────────────────────────────────────┘

Escalating Fidelity:

    Tier 1  z-space check (always, ~0.1ms):
        WM.predict(z, action) vs real_z
        Error < 0.10? → WM is fine, stop.

    Tier 2  ASCII grid check (5% of checks, ~1ms):
        Reconstruct ASCII grids from imagined vs real obs.
        Structural match? (entities present/absent, positions)
        Error < 0.20? → z-space was off but structure is fine → fast retrain.

    Tier 3  Full obs check (rare, ~5ms):
        Compare raw observation vectors.
        Used when ASCII grid hides sub-cell information.
        Fires only when Tier 2 surprise is high but unexplained.
        → Pattern interrupt handler, full WM retrain.

Statistics:
    After many calibration cycles, the scheduler tracks
    `steps_to_calibration` — how many imagined steps between
    real-env checks that maintain WM accuracy. This number
    should INCREASE over time as the WM gets better at
    modeling the environment's physics.
"""

from __future__ import annotations

import threading
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from queue import Queue, Empty

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.cell.world_model import CellWorldModel
from src.cell.surprise_classifier import SurpriseClassifier, SurpriseResult


# ══════════════════════════════════════════════════════════════
#  Calibration result
# ══════════════════════════════════════════════════════════════

@dataclass
class CalibrationResult:
    """Result of one calibration episode."""
    timestamp:            float
    n_steps:              int
    tier_reached:         int          # 1, 2, or 3
    tier1_surprise:       float        # z-space mean error
    tier2_surprise:       float        # ASCII grid error (0 if not checked)
    tier3_surprise:       float        # full obs error (0 if not checked)
    n_structural:         int          # structural surprises detected
    n_interrupts:         int          # pattern interrupts detected
    retrained:            bool         # did we fast_retrain?
    retrain_loss:         float        # loss after fast_retrain
    wm_confidence_before: float
    wm_confidence_after:  float


# ══════════════════════════════════════════════════════════════
#  CalibrationScheduler
# ══════════════════════════════════════════════════════════════

class CalibrationScheduler:
    """
    Async background WM calibration against reality.

    Does NOT block the training loop. Runs in a background thread.
    Updates WM weights via a thread-safe queue that the main loop
    checks at its own pace.

    Args:
        world_model:          The CellWorldModel used by VectorizedImaginedEnv.
        real_env_fn:          Callable that creates a real environment instance.
                              Called once at init. Must return an env with
                              .reset() → obs and .step(action) → (obs, rew, done, info).
        encode_fn:            Callable(raw_obs) → z_vector (np.ndarray, z_dim).
                              Maps real-env observations to z-space.
        n_actions:            Action space size.
        calibration_steps:    How many real-env steps per calibration episode.
        calibration_interval_s: Seconds between calibration episodes.
                                (Wall-clock, not imagined steps.)
        tier1_threshold:      z-space surprise below this → WM is fine.
        tier2_threshold:      ASCII surprise below this → fast retrain only.
        fast_retrain_steps:   Gradient steps for fast retrain after surprise.
        fast_retrain_lr_mult: Learning rate multiplier for fast retrain.
        ascii_compare_fn:     Optional Callable(raw_obs) → ascii_grid (np.ndarray).
                              Used for Tier 2 checks. If None, Tier 2 is skipped.
    """

    def __init__(
        self,
        world_model:             CellWorldModel,
        real_env_fn:             Callable,
        encode_fn:               Callable,
        n_actions:               int,
        calibration_steps:       int      = 64,
        calibration_interval_s:  float    = 5.0,
        tier1_threshold:         float    = 0.10,
        tier2_threshold:         float    = 0.20,
        fast_retrain_steps:      int      = 30,
        fast_retrain_lr_mult:    float    = 3.0,
        ascii_compare_fn:        Optional[Callable] = None,
    ):
        self.wm                     = world_model
        self.real_env_fn            = real_env_fn
        self.encode_fn              = encode_fn
        self.n_actions              = n_actions
        self.calibration_steps      = calibration_steps
        self.calibration_interval_s = calibration_interval_s
        self.tier1_threshold        = tier1_threshold
        self.tier2_threshold        = tier2_threshold
        self.fast_retrain_steps     = fast_retrain_steps
        self.fast_retrain_lr_mult   = fast_retrain_lr_mult
        self.ascii_compare_fn       = ascii_compare_fn

        # Surprise classifier for calibration checks
        self._clf = SurpriseClassifier(
            structural_abs=0.15,
            structural_spike=3.0,
            interrupt_abs=0.35,
        )

        # Thread-safe queue: background → main thread
        # Contains (wm_state_dict, calibration_result) tuples
        self._update_queue: Queue = Queue(maxsize=4)

        # Background thread state
        self._thread:   Optional[threading.Thread] = None
        self._running   = False
        self._real_env  = None

        # History
        self._results: deque = deque(maxlen=100)

        # Stats
        self._n_calibrations     = 0
        self._n_retrains         = 0
        self._n_tier2_checks     = 0
        self._n_tier3_checks     = 0
        self._total_real_steps   = 0
        self._imagined_steps_between: deque = deque(maxlen=50)

        # Adaptive interval: starts at calibration_interval_s,
        # increases as WM accuracy improves.
        self._current_interval = calibration_interval_s
        self._min_interval     = calibration_interval_s * 0.5
        self._max_interval     = calibration_interval_s * 10.0

    # ── Main thread API ──────────────────────────────────────

    def start(self):
        """Start background calibration thread."""
        if self._running:
            return
        self._running = True
        self._real_env = self.real_env_fn()
        self._thread = threading.Thread(
            target=self._calibration_loop,
            daemon=True,
            name="CalibrationScheduler",
        )
        self._thread.start()

    def stop(self):
        """Stop background calibration thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._real_env is not None and hasattr(self._real_env, 'close'):
            self._real_env.close()
            self._real_env = None

    def check_for_updates(self) -> Optional[CalibrationResult]:
        """
        Non-blocking check for WM updates from background calibration.

        Call this in the training loop (e.g., every N steps or every episode).
        If a calibration completed, the WM weights have already been updated
        (the WM is a shared object). This just returns the result for logging.

        Returns:
            CalibrationResult if a new calibration completed, else None.
        """
        try:
            result = self._update_queue.get_nowait()
            return result
        except Empty:
            return None

    def drain_updates(self) -> List[CalibrationResult]:
        """Non-blocking: get all pending calibration results."""
        results = []
        while True:
            try:
                results.append(self._update_queue.get_nowait())
            except Empty:
                break
        return results

    # ── Background thread ────────────────────────────────────

    def _calibration_loop(self):
        """Main loop of the background calibration thread."""
        while self._running:
            try:
                result = self._run_one_calibration()
                self._results.append(result)
                self._n_calibrations += 1

                # Put result in queue for main thread
                try:
                    self._update_queue.put_nowait(result)
                except Exception:
                    pass  # Queue full — main thread hasn't consumed yet

                # Adapt calibration frequency based on WM accuracy
                self._adapt_interval(result)

            except Exception as e:
                # Calibration failure is non-fatal
                pass

            # Sleep until next calibration
            time.sleep(self._current_interval)

    def _run_one_calibration(self) -> CalibrationResult:
        """
        Run one calibration episode:
          1. Step real env for N steps
          2. Encode each transition to z
          3. Compare WM predictions to actual z-transitions
          4. Escalate through tiers based on surprise
          5. If needed, fast_retrain WM on the surprising transitions
        """
        conf_before = self.wm.confidence

        # ── Collect real transitions ──────────────────────────
        transitions_z = []     # (z_state, action, z_next, reward)
        transitions_raw = []   # (raw_obs, action, raw_next, reward)
        surprises: List[SurpriseResult] = []

        obs = self._real_env.reset()
        z = self.encode_fn(obs)

        for step in range(self.calibration_steps):
            # Random action for exploration-style calibration
            action = int(np.random.randint(0, self.n_actions))

            real_obs, real_rew, done, info = self._real_env.step(action)
            real_z = self.encode_fn(real_obs)

            # Store
            transitions_z.append((
                z.copy(), action, real_z.copy(), float(real_rew)
            ))
            transitions_raw.append((
                np.asarray(obs,      dtype=np.float32),
                action,
                np.asarray(real_obs, dtype=np.float32),
                float(real_rew),
            ))

            # ── Tier 1: z-space surprise check ────────────────
            pred_z, pred_rew = self.wm.predict(z, action)
            surprise = self._clf.classify(
                predicted_next=pred_z,
                actual_next=real_z,
                prev_state=z,
            )
            surprises.append(surprise)

            # Store transition in WM replay buffer (regardless of surprise)
            self.wm.store_transition(z, action, real_z, real_rew)

            # Advance
            z = real_z
            obs = real_obs
            if done:
                obs = self._real_env.reset()
                z = self.encode_fn(obs)

        self._total_real_steps += self.calibration_steps

        # ── Compute aggregate surprise ────────────────────────
        tier1_errors = [s.total for s in surprises]
        tier1_surprise = float(np.mean(tier1_errors))
        n_structural   = sum(1 for s in surprises if s.is_structural)
        n_interrupts   = sum(1 for s in surprises if s.is_pattern_interrupt)

        tier2_surprise = 0.0
        tier3_surprise = 0.0
        tier_reached   = 1
        retrained      = False
        retrain_loss   = 0.0

        # ── Tier 1 pass: z-space surprise low → done ──────────
        if tier1_surprise < self.tier1_threshold and n_interrupts == 0:
            # WM is fine. Train step on collected data, but no urgency.
            self.wm.train_step()
            return CalibrationResult(
                timestamp=time.time(),
                n_steps=self.calibration_steps,
                tier_reached=1,
                tier1_surprise=tier1_surprise,
                tier2_surprise=0.0,
                tier3_surprise=0.0,
                n_structural=n_structural,
                n_interrupts=n_interrupts,
                retrained=False,
                retrain_loss=0.0,
                wm_confidence_before=conf_before,
                wm_confidence_after=self.wm.confidence,
            )

        # ── Tier 2: ASCII grid check (if available) ───────────
        tier_reached = 2
        self._n_tier2_checks += 1

        if self.ascii_compare_fn is not None:
            tier2_errors = []
            for raw_t in transitions_raw:
                obs_ascii  = self.ascii_compare_fn(raw_t[0])
                next_ascii = self.ascii_compare_fn(raw_t[2])
                # Compare entity positions / density patterns
                err = float(np.mean(np.abs(obs_ascii - next_ascii)))
                tier2_errors.append(err)
            tier2_surprise = float(np.mean(tier2_errors))

        # ── Fast retrain on collected transitions ─────────────
        retrained = True
        self._n_retrains += 1

        # Prioritize surprising transitions (higher surprise → more training)
        sorted_idx = np.argsort(tier1_errors)[::-1]
        # Top half most surprising transitions
        top_transitions = [transitions_z[i] for i in sorted_idx[:len(sorted_idx)//2 + 1]]

        ft_result = self.wm.fast_retrain(
            transitions=top_transitions,
            lr_multiplier=self.fast_retrain_lr_mult,
            n_steps=self.fast_retrain_steps,
        )
        retrain_loss = ft_result.get('fast_retrain_loss', 0.0)

        # Also run normal train_step on full buffer
        for _ in range(5):
            self.wm.train_step()

        # ── Tier 3: Full obs check (only on pattern interrupts) ─
        if n_interrupts > 0 or (tier2_surprise > self.tier2_threshold):
            tier_reached = 3
            self._n_tier3_checks += 1

            # Compare full raw observations
            tier3_errors = []
            for raw_t in transitions_raw:
                pred_raw_z, _ = self.wm.predict(
                    self.encode_fn(raw_t[0]), raw_t[1]
                )
                actual_z = self.encode_fn(raw_t[2])
                err = float(np.mean(np.abs(pred_raw_z - actual_z)))
                tier3_errors.append(err)
            tier3_surprise = float(np.mean(tier3_errors))

            # Extra retrain on the most surprising transitions
            self.wm.fast_retrain(
                transitions=top_transitions,
                lr_multiplier=self.fast_retrain_lr_mult * 2,
                n_steps=self.fast_retrain_steps * 2,
            )

        return CalibrationResult(
            timestamp=time.time(),
            n_steps=self.calibration_steps,
            tier_reached=tier_reached,
            tier1_surprise=round(tier1_surprise, 4),
            tier2_surprise=round(tier2_surprise, 4),
            tier3_surprise=round(tier3_surprise, 4),
            n_structural=n_structural,
            n_interrupts=n_interrupts,
            retrained=retrained,
            retrain_loss=round(retrain_loss, 4),
            wm_confidence_before=round(conf_before, 3),
            wm_confidence_after=round(self.wm.confidence, 3),
        )

    # ── Adaptive interval ────────────────────────────────────

    def _adapt_interval(self, result: CalibrationResult):
        """
        Adapt calibration frequency based on WM accuracy.

        If WM is consistently accurate → calibrate less often (save real-env cost).
        If WM had surprise → calibrate more often (catch drift early).
        """
        if result.tier_reached == 1 and result.tier1_surprise < self.tier1_threshold * 0.5:
            # Very accurate — slow down calibration
            self._current_interval = min(
                self._current_interval * 1.2,
                self._max_interval,
            )
        elif result.retrained:
            # Had to retrain — speed up calibration
            self._current_interval = max(
                self._current_interval * 0.6,
                self._min_interval,
            )

    # ── Stats ────────────────────────────────────────────────

    def stats(self) -> dict:
        recent = list(self._results)[-10:] if self._results else []
        avg_tier1 = float(np.mean([r.tier1_surprise for r in recent])) if recent else 0.0
        retrain_pct = sum(1 for r in recent if r.retrained) / max(1, len(recent))

        return {
            "running":              self._running,
            "n_calibrations":       self._n_calibrations,
            "n_retrains":           self._n_retrains,
            "n_tier2_checks":       self._n_tier2_checks,
            "n_tier3_checks":       self._n_tier3_checks,
            "total_real_steps":     self._total_real_steps,
            "current_interval_s":   round(self._current_interval, 1),
            "recent_avg_surprise":  round(avg_tier1, 4),
            "recent_retrain_pct":   round(retrain_pct, 2),
            "wm_confidence":        round(self.wm.confidence, 3),
        }

    def __del__(self):
        self.stop()


# ══════════════════════════════════════════════════════════════
#  Convenience: non-threaded calibrator for testing
# ══════════════════════════════════════════════════════════════

class SyncCalibrator:
    """
    Same logic as CalibrationScheduler but non-threaded.
    Call calibrate() manually from the training loop when convenient.
    Useful for testing and single-threaded environments.
    """

    def __init__(
        self,
        world_model:          CellWorldModel,
        real_env:             Any,
        encode_fn:            Callable,
        n_actions:            int,
        calibration_steps:    int      = 64,
        tier1_threshold:      float    = 0.10,
        fast_retrain_steps:   int      = 30,
        fast_retrain_lr_mult: float    = 3.0,
    ):
        self.wm                  = world_model
        self.real_env            = real_env
        self.encode_fn           = encode_fn
        self.n_actions           = n_actions
        self.calibration_steps   = calibration_steps
        self.tier1_threshold     = tier1_threshold
        self.fast_retrain_steps  = fast_retrain_steps
        self.fast_retrain_lr_mult = fast_retrain_lr_mult
        self._clf = SurpriseClassifier()
        self._results: deque = deque(maxlen=50)

    def calibrate(self) -> CalibrationResult:
        """Run one calibration episode synchronously. Returns result."""
        conf_before = self.wm.confidence
        transitions = []
        surprises = []

        obs = self.real_env.reset()
        z = self.encode_fn(obs)

        for _ in range(self.calibration_steps):
            action = int(np.random.randint(0, self.n_actions))
            real_obs, real_rew, done, _ = self.real_env.step(action)
            real_z = self.encode_fn(real_obs)

            transitions.append((z.copy(), action, real_z.copy(), float(real_rew)))

            pred_z, _ = self.wm.predict(z, action)
            surprise = self._clf.classify(pred_z, real_z, z)
            surprises.append(surprise)

            self.wm.store_transition(z, action, real_z, real_rew)

            z = real_z
            obs = real_obs
            if done:
                obs = self.real_env.reset()
                z = self.encode_fn(obs)

        tier1 = float(np.mean([s.total for s in surprises]))
        n_structural = sum(1 for s in surprises if s.is_structural)
        n_interrupts = sum(1 for s in surprises if s.is_pattern_interrupt)

        retrained = False
        retrain_loss = 0.0

        if tier1 >= self.tier1_threshold or n_interrupts > 0:
            retrained = True
            sorted_idx = np.argsort([s.total for s in surprises])[::-1]
            top = [transitions[i] for i in sorted_idx[:len(sorted_idx)//2 + 1]]
            ft = self.wm.fast_retrain(top, self.fast_retrain_lr_mult, self.fast_retrain_steps)
            retrain_loss = ft.get('fast_retrain_loss', 0.0)
            for _ in range(5):
                self.wm.train_step()

        result = CalibrationResult(
            timestamp=time.time(),
            n_steps=self.calibration_steps,
            tier_reached=1 if tier1 < self.tier1_threshold else 2,
            tier1_surprise=round(tier1, 4),
            tier2_surprise=0.0,
            tier3_surprise=0.0,
            n_structural=n_structural,
            n_interrupts=n_interrupts,
            retrained=retrained,
            retrain_loss=round(retrain_loss, 4),
            wm_confidence_before=round(conf_before, 3),
            wm_confidence_after=round(self.wm.confidence, 3),
        )
        self._results.append(result)
        return result

    def stats(self) -> dict:
        recent = list(self._results)[-10:]
        return {
            "n_calibrations":      len(self._results),
            "recent_avg_surprise": round(
                float(np.mean([r.tier1_surprise for r in recent])), 4
            ) if recent else 0.0,
            "recent_retrain_pct":  round(
                sum(1 for r in recent if r.retrained) / max(1, len(recent)), 2
            ),
        }
