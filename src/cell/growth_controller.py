"""
GrowthController — Neurogenesis and pruning for the ThrongletCell.

Adapted from throng2's AdaptiveDensityController + BrainLifecycleManager.
Simplified to work with the SNN feature extractor:

- GROW when: performance plateauing + room to grow + WM prediction error high
- PRUNE when: redundant neurons (low variance) + above min size
- Budget limiter: max neurons, max memory

No explicit meta-learner — uses simple performance metrics to auto-decide.
Avoids the throng5 bloat trap (7 poisonous subsystems).
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
from scipy.sparse import csr_matrix


class GrowthController:
    """
    Controls neurogenesis and pruning for the SNN feature extractor.

    Decisions are based on:
    - Reward plateau detection (running avg vs best avg)
    - SNN neuron activity variance (low = redundant)
    - Computational budget (max neurons)
    """

    def __init__(
        self,
        min_neurons: int = 32,
        max_neurons: int = 512,
        growth_batch: int = 16,
        prune_batch: int = 8,
        plateau_window: int = 50,
        plateau_threshold: float = 0.05,
        check_interval: int = 10,
        activity_history_len: int = 100,
    ):
        """
        Args:
            min_neurons: Minimum SNN neurons (never prune below this).
            max_neurons: Maximum SNN neurons (computational budget).
            growth_batch: Neurons to add per growth event.
            prune_batch: Neurons to remove per prune event.
            plateau_window: Episodes to check for performance plateau.
            plateau_threshold: Improvement needed to avoid "plateau" label.
            check_interval: Check grow/prune every N episodes.
            activity_history_len: Steps of activity to track per neuron.
        """
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.growth_batch = growth_batch
        self.prune_batch = prune_batch
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.check_interval = check_interval

        # Tracking
        self._reward_history = deque(maxlen=plateau_window * 2)
        self._best_avg = -float("inf")
        self._episode_count = 0
        self._grow_events = 0
        self._prune_events = 0
        self._neuron_activity_variance = None

    def on_episode_end(self, episode_reward: float):
        """Record episode reward for plateau detection."""
        self._reward_history.append(episode_reward)
        self._episode_count += 1

    def should_check(self) -> bool:
        """Check if we should evaluate grow/prune this episode."""
        return (
            self._episode_count > 0
            and self._episode_count % self.check_interval == 0
            and len(self._reward_history) >= self.plateau_window
        )

    def _is_plateauing(self) -> bool:
        """Detect if performance has plateaued."""
        if len(self._reward_history) < self.plateau_window:
            return False

        recent = list(self._reward_history)
        recent_avg = np.mean(recent[-self.plateau_window:])

        # Update best
        if recent_avg > self._best_avg + self.plateau_threshold:
            self._best_avg = recent_avg
            return False

        # Compare recent to older window
        if len(recent) >= self.plateau_window * 2:
            older_avg = np.mean(recent[:self.plateau_window])
            improvement = (recent_avg - older_avg) / max(abs(older_avg), 1.0)
            return improvement < self.plateau_threshold

        return False

    def decide(
        self,
        current_neurons: int,
        snn_activity: Optional[np.ndarray] = None,
        wm_loss: float = 0.0,
    ) -> str:
        """
        Decide whether to grow, prune, or hold.

        Args:
            current_neurons: Current SNN neuron count.
            snn_activity: Recent SNN activity vector (for redundancy detection).
            wm_loss: WorldModel loss (high = environment is complex).

        Returns:
            "grow", "prune", or "hold"
        """
        if not self.should_check():
            return "hold"

        plateauing = self._is_plateauing()

        # Compute neuron redundancy if activity provided
        redundant_ratio = 0.0
        if snn_activity is not None and len(snn_activity) > 0:
            # Neurons with very low variance are redundant
            variance = np.var(snn_activity, axis=0) if snn_activity.ndim > 1 else np.zeros(1)
            self._neuron_activity_variance = variance
            if len(variance) > 0:
                redundant_ratio = float(np.mean(variance < 1e-4))

        # Decision logic
        if plateauing and current_neurons < self.max_neurons:
            # Plateauing + room to grow → add neurons
            return "grow"
        elif redundant_ratio > 0.5 and current_neurons > self.min_neurons:
            # More than half neurons are redundant → prune
            return "prune"
        elif current_neurons > self.max_neurons:
            # Over budget → prune
            return "prune"

        return "hold"

    def grow_snn(self, snn) -> int:
        """
        Add neurons to the SNN feature extractor.

        Args:
            snn: SNNFeatureExtractor instance.

        Returns:
            Number of neurons added.
        """
        n_add = min(self.growth_batch, self.max_neurons - snn.n_neurons)
        if n_add <= 0:
            return 0

        old_n = snn.n_neurons
        new_n = old_n + n_add

        # Extend SNN arrays
        # 1. Positions (Fibonacci spiral extension)
        golden = (1 + np.sqrt(5)) / 2
        new_indices = np.arange(old_n, new_n)
        theta = 2 * np.pi * new_indices / golden
        r = np.sqrt(new_indices / new_n)
        new_positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        snn.positions = np.vstack([snn.positions, new_positions])

        # 2. Input weights (random, same scale as existing)
        scale = np.std(snn.input_weights) if snn.input_weights.size > 0 else 0.02
        new_input_weights = np.random.randn(n_add, snn.input_dim) * scale
        snn.input_weights = np.vstack([snn.input_weights, new_input_weights])

        # 3. Connectivity (sparse — connect new neurons to random existing)
        avg_conn = max(1, snn.weights.nnz // old_n)
        n_new_conn = n_add * avg_conn
        rows = np.concatenate([
            np.repeat(np.arange(old_n, new_n), avg_conn),
            np.random.randint(0, old_n, n_new_conn),
        ])
        cols = np.concatenate([
            np.random.randint(0, old_n, n_new_conn),
            np.repeat(np.arange(old_n, new_n), avg_conn),
        ])
        data = np.random.randn(len(rows)) * 0.02
        # Merge with existing
        from scipy.sparse import vstack, hstack, csr_matrix as csr

        old_w = snn.weights
        # Pad old weights to new size
        padded = csr((old_w.data, old_w.indices, old_w.indptr), shape=(old_n, old_n))
        full = csr((data, (rows, cols)), shape=(new_n, new_n))
        # Combine: place old weights in top-left, add new connections
        new_weights = csr((new_n, new_n))
        # Copy old block
        for i in range(old_n):
            start, end = padded.indptr[i], padded.indptr[i + 1]
            for j_idx in range(start, end):
                full[i, padded.indices[j_idx]] = padded.data[j_idx]
        snn.weights = full.tocsr()

        # 4. Activity arrays
        snn.activity = np.concatenate([snn.activity, np.zeros(n_add)])
        snn.prev_activity = np.concatenate([snn.prev_activity, np.zeros(n_add)])

        # 5. Frequencies
        snn.neuron_frequencies = np.concatenate([
            snn.neuron_frequencies,
            np.random.uniform(1.0, 100.0, size=n_add),
        ])

        # 6. Region IDs
        snn.region_ids = np.arange(new_n) % snn.n_regions

        # Update count
        snn.n_neurons = new_n

        self._grow_events += 1
        return n_add

    def prune_snn(self, snn) -> int:
        """
        Remove redundant neurons from the SNN feature extractor.

        Removes neurons with lowest activity variance (most redundant).

        Args:
            snn: SNNFeatureExtractor instance.

        Returns:
            Number of neurons removed.
        """
        n_remove = min(self.prune_batch, snn.n_neurons - self.min_neurons)
        if n_remove <= 0:
            return 0

        # Find most redundant neurons (lowest activity magnitude)
        activity_score = np.abs(snn.activity)
        keep_indices = np.argsort(activity_score)[n_remove:]  # Keep highest activity
        keep_indices = np.sort(keep_indices)

        new_n = len(keep_indices)

        # Subset all arrays
        snn.positions = snn.positions[keep_indices]
        snn.input_weights = snn.input_weights[keep_indices]
        snn.activity = snn.activity[keep_indices]
        snn.prev_activity = snn.prev_activity[keep_indices]
        snn.neuron_frequencies = snn.neuron_frequencies[keep_indices]

        # Rebuild connectivity (subset rows/cols)
        old_w = snn.weights.toarray()
        new_w = old_w[np.ix_(keep_indices, keep_indices)]
        snn.weights = csr_matrix(new_w)

        # Region IDs
        snn.region_ids = np.arange(new_n) % snn.n_regions

        # Update count
        snn.n_neurons = new_n

        self._prune_events += 1
        return n_remove

    def stats(self) -> dict:
        """Growth controller statistics."""
        recent_avg = (
            float(np.mean(list(self._reward_history)[-self.plateau_window:]))
            if self._reward_history else 0.0
        )
        return {
            "episode_count": self._episode_count,
            "grow_events": self._grow_events,
            "prune_events": self._prune_events,
            "best_avg": round(float(self._best_avg), 2)
            if self._best_avg > -float("inf") else 0.0,
            "recent_avg": round(recent_avg, 2),
            "plateauing": self._is_plateauing() if len(self._reward_history) >= self.plateau_window else False,
        }
