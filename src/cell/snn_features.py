"""
SNN Feature Extractor — Wraps throng2's PredictiveThrongletBrain as a feature provider.

The SNN does NOT act, does NOT receive reward. It observes compressed state,
detects temporal patterns via STDP-like predictive coding, and outputs a
fixed-size feature vector that augments the PPO's input.

Think of it as a receiver/antenna — it resonates with patterns in the data.
"""

import numpy as np
import io
import sys
from contextlib import redirect_stdout
from scipy.sparse import csr_matrix


class SNNFeatureExtractor:
    """
    SNN-based feature extractor using PredictiveThrongletBrain.

    Takes compressed state from encoder, runs one propagation step,
    and returns a fixed-size feature vector from SNN activity.

    Each neuron stores a frequency parameter (unused now, for v2 oscillatory).
    """

    def __init__(self, n_neurons: int = 64, input_dim: int = 16,
                 n_regions: int = 8, avg_connections: int = 6):
        """
        Args:
            n_neurons: Number of SNN neurons.
            input_dim: Dimension of compressed state from encoder.
            n_regions: Number of spatial regions for feature aggregation.
            avg_connections: Average connections per neuron.
        """
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.n_regions = n_regions

        # Build SNN — suppress the verbose init prints
        f = io.StringIO()
        with redirect_stdout(f):
            self._build_snn(n_neurons, avg_connections)

        # Input projection: compressed_dim → n_neurons
        # Small magnitude to avoid overwhelming SNN dynamics
        self.input_weights = np.random.randn(n_neurons, input_dim) * 0.02

        # Activity state
        self.activity = np.zeros(n_neurons)
        self.prev_activity = np.zeros(n_neurons)
        self.prediction_error = 0.0

        # Frequency parameter per neuron (stored, not wired — for future oscillatory)
        self.neuron_frequencies = np.random.uniform(1.0, 100.0, size=n_neurons)

        # Region assignments (divide neurons into spatial regions)
        self.region_ids = np.arange(n_neurons) % n_regions

        # Stats
        self._step_count = 0
        self._total_error = 0.0
        self._total_spikes = 0

    def _build_snn(self, n_neurons: int, avg_connections: int):
        """Build the SNN using throng2's geometry + connectivity."""
        try:
            from src.core.predictive_thronglet import (
                fibonacci_spiral_2d,
                create_small_world_connections_fast,
            )
            self.positions = fibonacci_spiral_2d(n_neurons)
            self.weights = create_small_world_connections_fast(
                self.positions,
                avg_connections=avg_connections,
                local_ratio=0.8,
            )
        except ImportError:
            # Fallback: build minimal SNN without throng2 imports
            self.positions = self._fibonacci_spiral(n_neurons)
            self.weights = self._random_sparse_connections(n_neurons, avg_connections)

    def _fibonacci_spiral(self, n: int) -> np.ndarray:
        """Minimal Fibonacci spiral placement."""
        golden = (1 + np.sqrt(5)) / 2
        indices = np.arange(n)
        theta = 2 * np.pi * indices / golden
        r = np.sqrt(indices / n)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    def _random_sparse_connections(self, n: int, avg_conn: int) -> csr_matrix:
        """Fallback sparse connectivity."""
        total = n * avg_conn
        rows = np.random.randint(0, n, total)
        cols = np.random.randint(0, n, total)
        data = np.random.randn(total) * 0.1
        mask = rows != cols
        return csr_matrix((data[mask], (rows[mask], cols[mask])), shape=(n, n))

    @property
    def feature_dim(self) -> int:
        """Dimension of the output feature vector."""
        # n_regions (mean rates) + 1 (prediction error) + n_regions (spike counts)
        return self.n_regions * 2 + 1

    def step(self, compressed_state: np.ndarray) -> np.ndarray:
        """
        Run one SNN step and extract features.

        Args:
            compressed_state: Output from encoder (compressed_dim,)

        Returns:
            feature_vector: Fixed-size SNN features
        """
        # 1. Project input into SNN space
        input_current = self.input_weights @ compressed_state

        # 2. Propagate through SNN (sparse matrix multiply)
        recurrent = self.weights @ self.activity
        total_input = input_current + recurrent * 0.3

        # 3. Simple LIF activation (threshold at 0, clamped)
        self.prev_activity = self.activity.copy()
        self.activity = np.tanh(total_input)  # Bounded [-1, 1]
        spikes = (self.activity > 0.3).astype(float)

        # 4. Leaky decay
        self.activity = self.activity * 0.8 + spikes * 0.1

        # 5. Prediction error (how surprising was this input?)
        if self._step_count > 0:
            self.prediction_error = float(np.mean(np.abs(
                self.activity - self.prev_activity
            )))
        else:
            self.prediction_error = 0.0

        # 6. Extract features by region
        region_rates = np.zeros(self.n_regions)
        region_spikes = np.zeros(self.n_regions)
        for r in range(self.n_regions):
            mask = self.region_ids == r
            region_rates[r] = np.mean(self.activity[mask])
            region_spikes[r] = np.sum(spikes[mask])

        # 7. Compose feature vector — NORMALIZED to small magnitude
        #    so SNN features augment but don't overwhelm raw obs
        total_spikes = max(1.0, np.sum(spikes))
        features = np.concatenate([
            np.tanh(region_rates) * 0.1,            # Bounded, scaled down
            [np.tanh(self.prediction_error) * 0.1], # Bounded scalar
            region_spikes / total_spikes * 0.1,     # Normalized distribution
        ])

        # Stats
        self._step_count += 1
        self._total_error += self.prediction_error
        self._total_spikes += int(np.sum(spikes))

        return features.astype(np.float32)

    def reset(self):
        """Reset SNN state (between episodes)."""
        self.activity = np.zeros(self.n_neurons)
        self.prev_activity = np.zeros(self.n_neurons)
        self.prediction_error = 0.0

    def stats(self) -> dict:
        """SNN statistics."""
        return {
            "n_neurons": self.n_neurons,
            "n_connections": self.weights.nnz,
            "step_count": self._step_count,
            "avg_prediction_error": round(
                self._total_error / max(1, self._step_count), 4
            ),
            "avg_spikes_per_step": round(
                self._total_spikes / max(1, self._step_count), 2
            ),
            "feature_dim": self.feature_dim,
        }
