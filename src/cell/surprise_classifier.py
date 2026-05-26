"""
surprise_classifier.py — Type 1 vs Type 2 surprise classification.

Two classes of prediction failure have fundamentally different responses:

  Type 1 — Parametric surprise:
    "I predicted reward=0.3, got 0.2."
    The model is right about *what happens*, just slightly off on magnitude.
    Fix: gradient step. Cost: cheap, continuous.

  Type 2 — Structural surprise (pattern interrupt):
    "I predicted I'd stay at position (3,3). I'm now at (3,8)."
    "I predicted this entity is static. It physically relocated me."
    The model has the wrong *topology* — a physics rule is missing entirely.
    Fix: add rule, fast-retrain, rebuild sim. Cost: expensive, discrete.

The wall-slide mechanic in ARC-AGI3 is a canonical Type 2 example:
  - Player moves left toward a gray-panel wall
  - Expected: player moves 1 cell left
  - Actual: player slides across entire room
  - Error magnitude: ~5-15 cells (>>1), sudden, position-wide
  → Pattern interrupt: add (wall_adjacent, action=left) → slide_to_far_side rule

Detection:
  Type 2 is identified by COMBINATION of:
    1. Large normalized prediction error (||predicted - actual|| > threshold)
    2. Suddenness: error spike vs recent rolling average
    3. Spatial coherence: multiple features change in same direction (displacement)
       vs scattered noise (which is Type 1 even at high magnitude)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional


# ══════════════════════════════════════════════════════════════
#  Result dataclass
# ══════════════════════════════════════════════════════════════

@dataclass
class SurpriseResult:
    """
    Result of classifying a prediction error.

    Attributes:
        total:              Normalized L2 prediction error (0 = perfect).
        type:               "parametric" | "structural"
        is_pattern_interrupt: True if structural AND above interrupt threshold.
        spike_ratio:        How much larger than the rolling average this error is.
        spatial_coherence:  0-1, how spatially correlated the error is (1=displacement).
        dominant_dims:      Indices of the feature dimensions with the largest errors.
        entity_tag:         Optional caller-supplied tag for which entity caused this.
    """
    total: float
    type: str                        # "parametric" | "structural"
    is_pattern_interrupt: bool
    spike_ratio: float               # error / rolling_avg_error
    spatial_coherence: float         # 0-1
    dominant_dims: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    entity_tag: Optional[str] = None

    @property
    def is_structural(self) -> bool:
        return self.type == "structural"

    @property
    def is_parametric(self) -> bool:
        return self.type == "parametric"

    def __repr__(self) -> str:
        interrupt_str = " [PATTERN INTERRUPT]" if self.is_pattern_interrupt else ""
        return (
            f"SurpriseResult(total={self.total:.3f}, type={self.type}"
            f", spike={self.spike_ratio:.1f}x, coherence={self.spatial_coherence:.2f}"
            f"{interrupt_str})"
        )


# ══════════════════════════════════════════════════════════════
#  Classifier
# ══════════════════════════════════════════════════════════════

class SurpriseClassifier:
    """
    Classifies prediction errors as parametric (Type 1) or structural (Type 2).

    Maintains a rolling history of recent errors to detect spikes.

    Usage:
        clf = SurpriseClassifier()
        result = clf.classify(predicted_next_state, actual_next_state, prev_state)
        if result.is_pattern_interrupt:
            pattern_interrupt_handler.handle(...)

    Thresholds (all tunable):
        structural_abs:   Absolute normalized error to call something structural.
        structural_spike: How many × the rolling average to call it a spike.
        interrupt_abs:    Absolute error to call a full pattern interrupt.
        coherence_min:    Minimum spatial coherence to confirm structural (not just noise).
        history_len:      Rolling window for spike detection.
        top_k_dims:       Number of dominant dimensions to return for attribution.
    """

    def __init__(
        self,
        structural_abs: float = 0.15,   # >15% of feature range = structural
        structural_spike: float = 3.0,  # 3× rolling avg = spike
        interrupt_abs: float = 0.35,    # >35% = pattern interrupt
        coherence_min: float = 0.30,    # spatial coherence needed for structural
        history_len: int = 50,
        top_k_dims: int = 10,
    ):
        self.structural_abs   = structural_abs
        self.structural_spike = structural_spike
        self.interrupt_abs    = interrupt_abs
        self.coherence_min    = coherence_min
        self.top_k_dims       = top_k_dims

        # Rolling error history for spike detection
        self._error_history: deque = deque(maxlen=history_len)

        # Per-entity-tag error tracking
        self._entity_errors: Dict[str, deque] = {}

        # Counters
        self.n_parametric    = 0
        self.n_structural    = 0
        self.n_interrupts    = 0

    # ── Public API ──────────────────────────────────────────

    def classify(
        self,
        predicted_next: np.ndarray,
        actual_next: np.ndarray,
        prev_state: Optional[np.ndarray] = None,
        entity_tag: Optional[str] = None,
    ) -> SurpriseResult:
        """
        Classify the prediction error between predicted_next and actual_next.

        Args:
            predicted_next: WM prediction for next state.
            actual_next:    Actually observed next state.
            prev_state:     State before the action (used to measure displacement).
                            If None, displacement heuristic is skipped.
            entity_tag:     Optional label for which entity caused this transition.

        Returns:
            SurpriseResult with type classification and diagnostics.
        """
        predicted_next = np.asarray(predicted_next, dtype=np.float32)
        actual_next    = np.asarray(actual_next,    dtype=np.float32)

        # ── 1. Compute normalized error ────────────────────
        error_vec   = actual_next - predicted_next
        abs_errors  = np.abs(error_vec)
        norm_error  = float(np.mean(abs_errors))        # MAE (more robust than MSE)
        max_error   = float(np.max(abs_errors))

        # ── 2. Spike detection ─────────────────────────────
        rolling_avg = float(np.mean(self._error_history)) if self._error_history else norm_error
        spike_ratio = norm_error / (rolling_avg + 1e-8)

        # ── 3. Spatial coherence ───────────────────────────
        # High coherence = errors are correlated (displacement).
        # Low coherence  = errors are scattered noise.
        coherence = self._spatial_coherence(error_vec, prev_state, actual_next)

        # ── 4. Top-K dominant dimensions ───────────────────
        top_k = min(self.top_k_dims, len(abs_errors))
        dominant_dims = np.argpartition(abs_errors, -top_k)[-top_k:] if top_k > 0 else np.array([], dtype=int)

        # ── 5. Classification ──────────────────────────────
        is_large_error  = norm_error  >= self.structural_abs
        is_spike        = spike_ratio >= self.structural_spike and len(self._error_history) >= 5
        is_coherent     = coherence   >= self.coherence_min
        is_structural   = (is_large_error or is_spike) and is_coherent
        is_interrupt    = is_structural and (
            norm_error >= self.interrupt_abs or max_error >= self.interrupt_abs * 2
        )

        surprise_type = "structural" if is_structural else "parametric"

        # ── 6. Update history ──────────────────────────────
        self._error_history.append(norm_error)

        if entity_tag is not None:
            if entity_tag not in self._entity_errors:
                self._entity_errors[entity_tag] = deque(maxlen=100)
            self._entity_errors[entity_tag].append(norm_error)

        # ── 7. Update counters ─────────────────────────────
        if is_interrupt:
            self.n_interrupts += 1
        if is_structural:
            self.n_structural += 1
        else:
            self.n_parametric += 1

        return SurpriseResult(
            total=norm_error,
            type=surprise_type,
            is_pattern_interrupt=is_interrupt,
            spike_ratio=spike_ratio,
            spatial_coherence=coherence,
            dominant_dims=dominant_dims,
            entity_tag=entity_tag,
        )

    def rolling_avg_error(self) -> float:
        """Current rolling average prediction error."""
        return float(np.mean(self._error_history)) if self._error_history else 0.0

    def per_entity_avg_error(self) -> Dict[str, float]:
        """Average prediction error per entity tag."""
        return {
            tag: float(np.mean(errors))
            for tag, errors in self._entity_errors.items()
        }

    def worst_understood_entity(self) -> Optional[str]:
        """Entity tag with the highest average prediction error."""
        per_entity = self.per_entity_avg_error()
        if not per_entity:
            return None
        return max(per_entity, key=lambda k: per_entity[k])

    def stats(self) -> dict:
        return {
            "n_parametric":    self.n_parametric,
            "n_structural":    self.n_structural,
            "n_interrupts":    self.n_interrupts,
            "rolling_avg_err": round(self.rolling_avg_error(), 4),
            "entity_errors":   {k: round(v, 4) for k, v in self.per_entity_avg_error().items()},
        }

    # ── Internal helpers ─────────────────────────────────────

    def _spatial_coherence(
        self,
        error_vec: np.ndarray,
        prev_state: Optional[np.ndarray],
        actual_next: np.ndarray,
    ) -> float:
        """
        Measure how correlated the error dimensions are.

        High coherence = errors all point in same direction (displacement event).
        Low coherence  = errors scattered (noise, unrelated features).

        Two methods:
          A. Displacement ratio: if prev_state given, compare actual displacement
             to predicted displacement. Large ratio → displacement event.
          B. Error sign alignment: fraction of error dims with the same sign.
             >70% same-sign → coherent.
        """
        if len(error_vec) == 0:
            return 0.0

        # Method A: displacement ratio (more reliable when prev_state available)
        if prev_state is not None:
            prev_state = np.asarray(prev_state, dtype=np.float32)
            actual_displacement = np.linalg.norm(actual_next - prev_state)
            prediction_error    = np.linalg.norm(error_vec)

            if actual_displacement > 1e-6:
                # Displacement events: error is a large fraction of total movement
                ratio = prediction_error / (actual_displacement + 1e-8)
                return float(min(1.0, ratio))

        # Method B: sign alignment
        nonzero = error_vec[np.abs(error_vec) > 1e-6]
        if len(nonzero) < 3:
            return 0.0

        pos_fraction = float(np.sum(nonzero > 0)) / len(nonzero)
        # Coherence = how far from 50/50 split (0=random, 1=all same sign)
        return float(abs(pos_fraction - 0.5) * 2.0)
