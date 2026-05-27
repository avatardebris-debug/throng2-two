"""
uncertainty_seeker.py — Navigate toward uncertain entities to reduce global
prediction error through local interaction.

The core idea:
    If the WM can't predict entity X well → go INTERACT with X.
    After interaction, two outcomes:
      1. Prediction improves → learned X's dynamics → move on
      2. Prediction doesn't improve after N tries →
         a. X is stochastic → model as distribution, stop trying
         b. X has PRECURSORS → its behavior depends on something else
            happening first (key→door, switch→wall)

    In case (b), track conditional dependencies:
        "Entity X's prediction error dropped after interacting with Y first"
        → store edge Y→X in the precursor graph.

    This captures chains like:
        grab_key → door_opens
        press_switch → wall_moves
        rotate_shape_A → shape_A_matches_B → collect_both

Entity Lifecycle:
    UNKNOWN       → never interacted → highest priority to visit
    INVESTIGATING → currently trying to predict → N attempts remaining
    UNDERSTOOD    → prediction error < threshold → low priority
    STOCHASTIC    → N attempts failed, no improvement → model as distribution
    PRECURSOR_DEP → prediction depends on another entity → seek precursor first

Precursor Graph:
    Directed graph. Edge (A → B) means:
        "After interacting with A, entity B's prediction confidence increased."

    This encodes conditional dependencies without explicit logic.
    Built purely from observed confidence changes.

    When seeking to understand entity B:
      1. Try direct interaction N times
      2. If no improvement, check precursor graph: any edges →B?
      3. If yes, go interact with the precursor first, then retry B
      4. If no precursors known, mark B as stochastic (or undiscovered precursor)

Usage:
    seeker = UncertaintySeeker(world_model, n_actions=8)

    # Each step: get exploration target
    target = seeker.next_target()            # → entity_tag or None
    action = seeker.suggest_action(state)    # → action toward target

    # After observing result:
    seeker.observe(state, action, next_state, entity_tag, surprise)

    # Periodically:
    seeker.update_precursor_graph()
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.cell.world_model import CellWorldModel
from src.cell.surprise_classifier import SurpriseResult


# ══════════════════════════════════════════════════════════════
#  Entity lifecycle
# ══════════════════════════════════════════════════════════════

class EntityStatus(Enum):
    UNKNOWN        = "unknown"          # Never interacted → highest priority
    INVESTIGATING  = "investigating"    # Currently trying to predict
    UNDERSTOOD     = "understood"       # Prediction error < threshold
    STOCHASTIC     = "stochastic"       # N attempts, no improvement → distribution
    PRECURSOR_DEP  = "precursor_dep"    # Depends on another entity


@dataclass
class EntityRecord:
    """Tracked state for one entity/variable in the environment."""
    tag:                 str
    status:              EntityStatus = EntityStatus.UNKNOWN
    n_interactions:      int   = 0           # total times interacted
    n_investigation_attempts: int = 0        # attempts during current investigation
    max_investigation:   int   = 10          # max attempts before giving up
    prediction_errors:   list  = field(default_factory=list)  # recent errors
    confidence:          float = 0.0         # WM confidence for this entity
    confidence_history:  list  = field(default_factory=list)
    last_interaction_step: int = 0
    precursors:          set   = field(default_factory=set)  # entity tags that affect this one
    is_consequential:    bool  = False       # True if this entity was near a consequence

    @property
    def recent_avg_error(self) -> float:
        if not self.prediction_errors:
            return 1.0  # Max uncertainty if never observed
        return float(np.mean(self.prediction_errors[-10:]))

    @property
    def error_trend(self) -> float:
        """Negative = improving, positive = getting worse."""
        if len(self.prediction_errors) < 4:
            return 0.0
        recent = np.mean(self.prediction_errors[-5:])
        older  = np.mean(self.prediction_errors[-10:-5]) if len(self.prediction_errors) >= 10 else np.mean(self.prediction_errors[:5])
        return float(recent - older)

    @property
    def is_improving(self) -> bool:
        return self.error_trend < -0.01


# ══════════════════════════════════════════════════════════════
#  Precursor Graph
# ══════════════════════════════════════════════════════════════

class PrecursorGraph:
    """
    Directed graph encoding conditional dependencies between entities.

    Edge (A → B, strength=0.7) means:
        "After interacting with entity A, entity B's prediction confidence
         increased by 0.7 on average."

    Built from observed confidence changes, not hardcoded logic.
    """

    def __init__(self, min_edge_strength: float = 0.05):
        self.min_edge_strength = min_edge_strength

        # edges[B] = {A: strength} — "A is a precursor of B"
        self._edges: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Observation buffer: (interaction_entity, affected_entity, confidence_delta)
        self._observations: deque = deque(maxlen=500)

    def observe_confidence_change(
        self,
        interacted_entity: str,
        affected_entity:   str,
        confidence_delta:  float,
    ):
        """
        Record that interacting with `interacted_entity` changed
        `affected_entity`'s prediction confidence by `confidence_delta`.

        Positive delta = confidence increased (potential precursor).
        Negative delta = confidence decreased (interference).
        """
        if interacted_entity == affected_entity:
            return  # Self-interaction not a precursor

        self._observations.append((interacted_entity, affected_entity, confidence_delta))

    def rebuild(self):
        """
        Rebuild edges from observations.
        Called periodically (not every step — observations accumulate).
        """
        # Aggregate: for each (A, B) pair, average the confidence deltas
        pair_deltas: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for src, dst, delta in self._observations:
            pair_deltas[(src, dst)].append(delta)

        self._edges.clear()
        for (src, dst), deltas in pair_deltas.items():
            avg_delta = float(np.mean(deltas))
            if avg_delta > self.min_edge_strength:
                self._edges[dst][src] = avg_delta

    def get_precursors(self, entity_tag: str) -> Dict[str, float]:
        """Get precursors for an entity: {precursor_tag: strength}."""
        return dict(self._edges.get(entity_tag, {}))

    def strongest_precursor(self, entity_tag: str) -> Optional[str]:
        """Get the single strongest precursor for an entity."""
        precs = self.get_precursors(entity_tag)
        if not precs:
            return None
        return max(precs, key=precs.get)

    def has_precursors(self, entity_tag: str) -> bool:
        return bool(self._edges.get(entity_tag))

    def all_edges(self) -> List[Tuple[str, str, float]]:
        """List all edges as (precursor, dependent, strength)."""
        edges = []
        for dst, srcs in self._edges.items():
            for src, strength in srcs.items():
                edges.append((src, dst, strength))
        return sorted(edges, key=lambda e: -e[2])

    def stats(self) -> dict:
        all_e = self.all_edges()
        return {
            "n_edges":        len(all_e),
            "n_observations": len(self._observations),
            "top_edges":      [(s, d, round(w, 3)) for s, d, w in all_e[:5]],
        }


# ══════════════════════════════════════════════════════════════
#  UncertaintySeeker
# ══════════════════════════════════════════════════════════════

class UncertaintySeeker:
    """
    Directed exploration toward uncertain entities.

    Strategy:
      1. Identify entities with highest prediction uncertainty
      2. Navigate toward them (suggest actions that lead to interaction)
      3. After interaction, check if prediction improved
      4. If not improved after N tries → mark stochastic or seek precursors
      5. Build precursor graph from observed conditional dependencies

    Args:
        world_model:            CellWorldModel with per-entity confidence.
        n_actions:              Action space size.
        understood_threshold:   Prediction error below this → entity understood.
        stochastic_threshold:   After max_investigation with no improvement → stochastic.
        max_investigation:      Max interaction attempts before giving up.
        precursor_check_interval: Steps between precursor graph rebuilds.
    """

    def __init__(
        self,
        world_model:              CellWorldModel,
        n_actions:                int,
        understood_threshold:     float = 0.08,
        stochastic_threshold:     float = 0.25,
        max_investigation:        int   = 15,
        precursor_check_interval: int   = 50,
    ):
        self.wm                       = world_model
        self.n_actions                = n_actions
        self.understood_threshold     = understood_threshold
        self.stochastic_threshold     = stochastic_threshold
        self.max_investigation        = max_investigation
        self.precursor_check_interval = precursor_check_interval

        # Entity registry
        self._entities: Dict[str, EntityRecord] = {}

        # Precursor graph
        self.precursor_graph = PrecursorGraph()

        # Current exploration target
        self._current_target: Optional[str] = None

        # Step counter (for precursor rebuild scheduling)
        self._step_count = 0

        # Snapshot of all entity confidences before each interaction
        # Used to detect "interacting with A changed B's confidence"
        self._confidence_snapshot: Dict[str, float] = {}

        # Stats
        self._n_investigations_started  = 0
        self._n_understood              = 0
        self._n_stochastic              = 0
        self._n_precursor_discoveries   = 0

    # ── Entity registration ──────────────────────────────────

    def register_entity(self, tag: str, is_consequential: bool = False):
        """Register a new entity type. Called when a new entity is detected."""
        if tag not in self._entities:
            self._entities[tag] = EntityRecord(
                tag=tag,
                max_investigation=self.max_investigation,
                is_consequential=is_consequential,
            )

    def _ensure_entity(self, tag: str):
        if tag not in self._entities:
            self.register_entity(tag)

    # ── Main interface ───────────────────────────────────────

    def next_target(self) -> Optional[str]:
        """
        Get the entity that should be explored next.

        Priority:
          1. UNKNOWN entities (never interacted)
          2. PRECURSOR_DEP whose precursors are UNDERSTOOD (ready to retry)
          3. INVESTIGATING entities (ongoing investigation)
          4. CONSEQUENTIAL entities with declining confidence
          5. Highest recent_avg_error among non-stochastic entities

        Returns entity tag or None if no exploration target.
        """
        if not self._entities:
            return None

        # Update confidences from WM
        self._sync_confidences()

        best_tag = None
        best_priority = -1.0

        for tag, rec in self._entities.items():
            if rec.status == EntityStatus.STOCHASTIC:
                continue
            if rec.status == EntityStatus.UNDERSTOOD and rec.recent_avg_error < self.understood_threshold:
                continue

            # Priority scoring
            priority = 0.0

            if rec.status == EntityStatus.UNKNOWN:
                priority = 10.0  # Highest: never seen

            elif rec.status == EntityStatus.PRECURSOR_DEP:
                # Check if precursors are now understood
                prec = self.precursor_graph.strongest_precursor(tag)
                if prec and prec in self._entities:
                    prec_rec = self._entities[prec]
                    if prec_rec.status == EntityStatus.UNDERSTOOD:
                        priority = 8.0  # Precursor done → retry dependent

            elif rec.status == EntityStatus.INVESTIGATING:
                priority = 5.0 + rec.recent_avg_error

            else:
                priority = rec.recent_avg_error

            # Boost consequential entities
            if rec.is_consequential:
                priority *= 1.5

            if priority > best_priority:
                best_priority = priority
                best_tag = tag

        self._current_target = best_tag
        return best_tag

    def suggest_action(self, state: np.ndarray) -> int:
        """
        Suggest an action that moves toward the current exploration target.

        Currently: uses information_gain as a proxy (action with highest
        WM uncertainty = most likely to encounter uncertain entity).

        Future: entity-specific navigation using learned approach policies.
        """
        if not self.wm.is_ready:
            return int(np.random.randint(self.n_actions))

        # Use information gain — actions with highest uncertainty
        # are most likely to lead to interaction with uncertain entities
        ig = self.wm.information_gains_all_actions(state, n_samples=3)
        return int(np.argmax(ig))

    def observe(
        self,
        state:       np.ndarray,
        action:      int,
        next_state:  np.ndarray,
        entity_tag:  Optional[str],
        surprise:    Optional[SurpriseResult] = None,
        reward:      float = 0.0,
    ):
        """
        Observe the result of an interaction.

        Updates entity records, checks for prediction improvement,
        detects precursor relationships.
        """
        self._step_count += 1

        if entity_tag is None:
            return

        self._ensure_entity(entity_tag)
        rec = self._entities[entity_tag]

        # Record prediction error
        if surprise is not None:
            rec.prediction_errors.append(surprise.total)
        else:
            # Compute from WM directly
            sr = self.wm.measure_surprise(state, action, next_state, entity_tag)
            rec.prediction_errors.append(sr.total)

        rec.n_interactions += 1
        rec.last_interaction_step = self._step_count

        # Mark consequential if near a large reward
        if abs(reward) > 1.0:
            rec.is_consequential = True

        # ── State machine transitions ─────────────────────────

        old_status = rec.status

        if rec.status == EntityStatus.UNKNOWN:
            rec.status = EntityStatus.INVESTIGATING
            rec.n_investigation_attempts = 1
            self._n_investigations_started += 1
            self._snapshot_confidences()  # Baseline for precursor detection

        elif rec.status == EntityStatus.INVESTIGATING:
            rec.n_investigation_attempts += 1

            if rec.recent_avg_error < self.understood_threshold:
                # Prediction is good → understood
                rec.status = EntityStatus.UNDERSTOOD
                rec.confidence = 1.0 - rec.recent_avg_error
                self._n_understood += 1

            elif rec.n_investigation_attempts >= rec.max_investigation:
                # Max attempts reached
                if rec.is_improving:
                    # Still improving → give more attempts
                    rec.max_investigation += 5
                else:
                    # Not improving → check for precursors
                    precs = self.precursor_graph.get_precursors(entity_tag)
                    if precs:
                        rec.status = EntityStatus.PRECURSOR_DEP
                        rec.precursors = set(precs.keys())
                        self._n_precursor_discoveries += 1
                    else:
                        rec.status = EntityStatus.STOCHASTIC
                        self._n_stochastic += 1

        elif rec.status == EntityStatus.PRECURSOR_DEP:
            # Retry after precursor was handled
            if rec.recent_avg_error < self.understood_threshold:
                rec.status = EntityStatus.UNDERSTOOD
                self._n_understood += 1
            # else: still depends on precursor, keep trying

        elif rec.status == EntityStatus.UNDERSTOOD:
            # Check for regression
            if rec.recent_avg_error > self.understood_threshold * 2:
                rec.status = EntityStatus.INVESTIGATING
                rec.n_investigation_attempts = 0

        # ── Precursor detection ───────────────────────────────
        # After interacting with entity_tag, check if OTHER entities'
        # confidences changed → potential precursor relationship
        self._detect_precursors(entity_tag)

        # Periodic precursor graph rebuild
        if self._step_count % self.precursor_check_interval == 0:
            self.precursor_graph.rebuild()

    def _snapshot_confidences(self):
        """Save current per-entity confidences for precursor detection."""
        self._confidence_snapshot = dict(self.wm.per_entity_confidence)

    def _detect_precursors(self, interacted_entity: str):
        """
        After interacting with `interacted_entity`, check if any OTHER
        entity's confidence changed. If so, record as potential precursor.
        """
        current = self.wm.per_entity_confidence
        for tag, conf in current.items():
            if tag == interacted_entity:
                continue
            prev = self._confidence_snapshot.get(tag, 0.0)
            delta = conf - prev
            if abs(delta) > 0.02:  # Significant change
                self.precursor_graph.observe_confidence_change(
                    interacted_entity=interacted_entity,
                    affected_entity=tag,
                    confidence_delta=delta,
                )
        # Update snapshot
        self._confidence_snapshot = current

    def _sync_confidences(self):
        """Update entity records with current WM per-entity confidence."""
        wm_conf = self.wm.per_entity_confidence
        for tag, conf in wm_conf.items():
            if tag in self._entities:
                self._entities[tag].confidence = conf
                self._entities[tag].confidence_history.append(conf)
                # Trim history
                if len(self._entities[tag].confidence_history) > 100:
                    self._entities[tag].confidence_history = \
                        self._entities[tag].confidence_history[-50:]

    # ── Query methods ────────────────────────────────────────

    def get_exploration_priorities(self) -> Dict[str, float]:
        """Get priority score for every entity. Higher = more worth exploring."""
        priorities = {}
        for tag, rec in self._entities.items():
            if rec.status == EntityStatus.STOCHASTIC:
                priorities[tag] = 0.0
            elif rec.status == EntityStatus.UNDERSTOOD:
                priorities[tag] = 0.1
            elif rec.status == EntityStatus.UNKNOWN:
                priorities[tag] = 10.0
            elif rec.status == EntityStatus.PRECURSOR_DEP:
                priorities[tag] = 5.0
            else:
                priorities[tag] = rec.recent_avg_error * (1.5 if rec.is_consequential else 1.0)
        return priorities

    def get_stochastic_entities(self) -> List[str]:
        """Entities confirmed as stochastic (model as distribution)."""
        return [tag for tag, rec in self._entities.items()
                if rec.status == EntityStatus.STOCHASTIC]

    def get_precursor_chain(self, entity_tag: str) -> List[str]:
        """
        Get the full precursor chain for an entity.
        E.g., for 'door': ['key', 'switch'] if key→switch→door.
        """
        chain = []
        visited = set()
        current = entity_tag
        while current:
            prec = self.precursor_graph.strongest_precursor(current)
            if prec is None or prec in visited:
                break
            chain.append(prec)
            visited.add(prec)
            current = prec
        return list(reversed(chain))

    # ── Stats ────────────────────────────────────────────────

    def stats(self) -> dict:
        status_counts = defaultdict(int)
        for rec in self._entities.values():
            status_counts[rec.status.value] += 1

        return {
            "n_entities":           len(self._entities),
            "status_distribution":  dict(status_counts),
            "current_target":       self._current_target,
            "n_investigations":     self._n_investigations_started,
            "n_understood":         self._n_understood,
            "n_stochastic":         self._n_stochastic,
            "n_precursor_found":    self._n_precursor_discoveries,
            "precursor_graph":      self.precursor_graph.stats(),
            "exploration_priorities": {
                tag: round(p, 3)
                for tag, p in sorted(
                    self.get_exploration_priorities().items(),
                    key=lambda x: -x[1],
                )[:10]
            },
        }
