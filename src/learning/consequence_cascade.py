"""
consequence_cascade.py — Weighted replay for rare consequential transitions.

When a rare but consequential event is observed (win, loss, large reward,
structural state change), the ENTIRE trajectory that led to it gets stored
with exponentially decaying weights backward from the event. This ensures
the WM trains heavily on the causal chain, not just the terminal event.

Standard PER uses TD error as priority. TD error is near-zero for 4999 of
5000 steps leading to a rare outcome. Consequence weighting fixes this by
explicitly propagating the consequence signal backward through the trajectory.

Weight formula:
    w(step_i) = magnitude × decay^(consequence_step - i)

Where:
    magnitude = abs(reward) or custom consequence score
    decay = 0.99 (per-step exponential falloff)
    consequence_step = the step where the consequence occurred

This means step N-1 gets weight ≈ magnitude×0.99 and step N-100 gets
weight ≈ magnitude×0.37. The chain is long enough to capture the full
approach path but fades for the irrelevant early-episode wandering.

Usage:
    cascade = ConsequenceCascadeBuffer(capacity=50_000)

    # During episode: store transitions
    cascade.begin_episode()
    for step in range(episode_len):
        cascade.store_step(state, action, next_state, reward)

    # After episode: mark consequential events
    cascade.mark_consequential(step=4999, magnitude=10.0)
    cascade.end_episode()

    # During WM training: sample proportional to consequence weight
    batch = cascade.sample(batch_size=64)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class WeightedTransition:
    """Transition with consequence weight and metadata."""
    state:       np.ndarray
    action:      int
    next_state:  np.ndarray
    reward:      float
    weight:      float           # consequence weight (higher = more important)
    episode_id:  int             # which episode this came from
    step_idx:    int             # step within episode
    entity_tag:  Optional[str] = None  # entity involved (if known)


class ConsequenceCascadeBuffer:
    """
    Replay buffer that weights transitions by their proximity to
    rare consequential events.

    Args:
        capacity:       Max transitions stored. Evicts lowest-weight first.
        decay:          Per-step weight decay backward from consequence.
        base_weight:    Default weight for transitions with no consequence.
        min_magnitude:  Minimum consequence magnitude to trigger cascade.
    """

    def __init__(
        self,
        capacity:      int   = 50_000,
        decay:         float = 0.995,
        base_weight:   float = 0.01,
        min_magnitude: float = 0.5,
    ):
        self.capacity      = capacity
        self.decay         = decay
        self.base_weight   = base_weight
        self.min_magnitude = min_magnitude

        # Storage
        self._buffer:  List[WeightedTransition] = []
        self._weights: np.ndarray = np.array([], dtype=np.float64)

        # Current episode accumulator
        self._current_episode: List[Tuple] = []  # (state, action, next_state, reward, entity_tag)
        self._episode_counter = 0

        # Stats
        self._n_consequences     = 0
        self._total_stored       = 0
        self._n_cascade_chains   = 0
        self._max_chain_length   = 0

    # ── Episode lifecycle ────────────────────────────────────

    def begin_episode(self):
        """Start accumulating a new episode."""
        self._current_episode = []

    def store_step(
        self,
        state:      np.ndarray,
        action:     int,
        next_state: np.ndarray,
        reward:     float,
        entity_tag: Optional[str] = None,
    ):
        """Accumulate one step in the current episode."""
        self._current_episode.append((
            np.asarray(state,      dtype=np.float32).copy(),
            int(action),
            np.asarray(next_state, dtype=np.float32).copy(),
            float(reward),
            entity_tag,
        ))

    def mark_consequential(
        self,
        step:      int,
        magnitude: float,
        tag:       str = "consequence",
    ):
        """
        Mark a step in the current episode as consequential.
        Backward-propagates weights through the trajectory.

        Args:
            step:      Index of the consequential step.
            magnitude: Importance (e.g., abs(reward), win=10, loss=5).
            tag:       Label for this type of consequence.
        """
        if magnitude < self.min_magnitude:
            return
        if not self._current_episode:
            return

        step = min(step, len(self._current_episode) - 1)
        self._n_consequences += 1
        self._n_cascade_chains += 1

        # Backward cascade: assign weights
        chain_length = 0
        for i in range(step, -1, -1):
            steps_from_consequence = step - i
            w = magnitude * (self.decay ** steps_from_consequence)
            if w < self.base_weight * 0.1:
                break  # Weight too small, stop cascading

            s, a, ns, r, etag = self._current_episode[i]
            self._add_transition(WeightedTransition(
                state=s, action=a, next_state=ns, reward=r,
                weight=w,
                episode_id=self._episode_counter,
                step_idx=i,
                entity_tag=etag,
            ))
            chain_length += 1

        self._max_chain_length = max(self._max_chain_length, chain_length)

    def end_episode(self, auto_detect_consequences: bool = True):
        """
        Finalize the episode. Optionally auto-detect consequences
        from large reward spikes.

        Args:
            auto_detect_consequences: If True, automatically mark steps
                with |reward| > min_magnitude as consequential.
        """
        if auto_detect_consequences and self._current_episode:
            for i, (s, a, ns, r, etag) in enumerate(self._current_episode):
                if abs(r) >= self.min_magnitude:
                    self.mark_consequential(i, abs(r), tag="auto_reward")

        # Store remaining transitions with base weight
        for i, (s, a, ns, r, etag) in enumerate(self._current_episode):
            self._add_transition(WeightedTransition(
                state=s, action=a, next_state=ns, reward=r,
                weight=self.base_weight,
                episode_id=self._episode_counter,
                step_idx=i,
                entity_tag=etag,
            ))

        self._episode_counter += 1
        self._current_episode = []

    # ── Sampling ─────────────────────────────────────────────

    def sample(self, batch_size: int) -> List[WeightedTransition]:
        """
        Sample transitions proportional to consequence weight.

        Higher-weight transitions (near consequences) get sampled
        much more frequently than baseline wandering steps.
        """
        if not self._buffer:
            return []

        n = min(batch_size, len(self._buffer))

        # Normalize weights to probabilities
        probs = self._weights[:len(self._buffer)]
        total = probs.sum()
        if total < 1e-10:
            probs = np.ones(len(self._buffer)) / len(self._buffer)
        else:
            probs = probs / total

        indices = np.random.choice(len(self._buffer), size=n, replace=False, p=probs)
        return [self._buffer[i] for i in indices]

    def sample_as_tuples(self, batch_size: int) -> List[Tuple]:
        """Sample as (state, action, next_state, reward) tuples for WM training."""
        transitions = self.sample(batch_size)
        return [(t.state, t.action, t.next_state, t.reward) for t in transitions]

    # ── Internal ─────────────────────────────────────────────

    def _add_transition(self, t: WeightedTransition):
        """Add transition, evicting lowest-weight if at capacity."""
        if len(self._buffer) >= self.capacity:
            # Evict lowest weight
            min_idx = int(np.argmin(self._weights[:len(self._buffer)]))

            # Only evict if new transition has higher weight
            if t.weight <= self._weights[min_idx]:
                return  # Not important enough to store

            self._buffer[min_idx] = t
            self._weights[min_idx] = t.weight
        else:
            self._buffer.append(t)
            self._weights = np.append(self._weights, t.weight)

        self._total_stored += 1

    def __len__(self) -> int:
        return len(self._buffer)

    # ── Stats ────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._buffer:
            return {"size": 0, "n_consequences": 0}

        w = self._weights[:len(self._buffer)]
        high_weight = int(np.sum(w > self.base_weight * 10))

        return {
            "size":              len(self._buffer),
            "n_consequences":    self._n_consequences,
            "n_cascade_chains":  self._n_cascade_chains,
            "max_chain_length":  self._max_chain_length,
            "total_stored":      self._total_stored,
            "high_weight_count": high_weight,
            "mean_weight":       round(float(np.mean(w)), 4),
            "max_weight":        round(float(np.max(w)), 4),
            "weight_ratio":      round(float(np.max(w) / max(np.min(w), 1e-10)), 1),
        }
