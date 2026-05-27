"""Shared world-model diagnostics (no replay buffer, no dynamics networks)."""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

from ..surprise_classifier import SurpriseClassifier


class WorldModelCore:
    """
    Base for learned dynamics models: training counters, surprise classifier,
    and sim2real tracking. Replay buffers live on concrete subclasses only.
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        batch_size: int = 64,
        min_transitions: int = 200,
    ):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.min_transitions = min_transitions

        self._total_updates = 0
        self._losses: deque = deque(maxlen=100)

        self._surprise_clf = SurpriseClassifier(
            structural_abs=0.15,
            structural_spike=3.0,
            interrupt_abs=0.35,
            coherence_min=0.30,
        )

        self._sim2real_errors: deque = deque(maxlen=200)
        self._current_sim_state: Optional[np.ndarray] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def replay_size(self) -> int:
        """Transitions in the primary replay store (subclass-specific)."""
        return 0

    @property
    def is_ready(self) -> bool:
        return self.replay_size() >= self.min_transitions and self._total_updates >= 10

    def is_ready_for(self, game_id: int) -> bool:
        """Default: global readiness (single-game). Multi-game overrides."""
        del game_id
        return self.is_ready

    @property
    def confidence(self) -> float:
        if not self._losses or not self.is_ready:
            return 0.0
        avg_loss = np.mean(list(self._losses)[-20:])
        return float(min(1.0, 1.0 / (1.0 + avg_loss)))

    @property
    def sim2real_accuracy(self) -> float:
        if not self._sim2real_errors:
            return 0.0
        mean_err = float(np.mean(self._sim2real_errors))
        return float(max(0.0, 1.0 - mean_err))

    @property
    def per_entity_confidence(self) -> Dict[str, float]:
        per_entity_err = self._surprise_clf.per_entity_avg_error()
        return {
            tag: float(min(1.0, 1.0 / (1.0 + err)))
            for tag, err in per_entity_err.items()
        }

    def worst_understood_entity(self) -> Optional[str]:
        return self._surprise_clf.worst_understood_entity()

    def stats(self) -> dict:
        clf_stats = self._surprise_clf.stats()
        return {
            "buffer_size": self.replay_size(),
            "total_updates": self._total_updates,
            "is_ready": self.is_ready,
            "confidence": round(self.confidence, 3),
            "sim2real_accuracy": round(self.sim2real_accuracy, 3),
            "avg_loss": round(float(np.mean(list(self._losses)[-20:])), 4)
            if self._losses
            else 0.0,
            "n_parametric": clf_stats["n_parametric"],
            "n_structural": clf_stats["n_structural"],
            "n_interrupts": clf_stats["n_interrupts"],
            "rolling_err": clf_stats["rolling_avg_err"],
            "entity_errors": clf_stats["entity_errors"],
            "per_entity_confidence": {
                k: round(v, 3) for k, v in self.per_entity_confidence.items()
            },
        }
