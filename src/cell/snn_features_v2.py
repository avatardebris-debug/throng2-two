"""
SNN Feature Extractor v2 — Resonant Frequency Bands

Key changes from v1:
  - NO Fibonacci spiral (geometry not proven necessary)  
  - NO CSR sparse matrix (dense numpy, faster at <2K neurons)
  - NO STDP (rate coding + STDP shown to fail in 87 Tetris experiments)
  - YES frequency bands: delta/theta/alpha/gamma
  - YES prediction error learning (unsupervised, self-supervising)
  - YES local update: each band updates at its own rate

Frequency bands (inspired by neural oscillations):
  delta  (1-4 Hz):  slow, 1 update per 8 steps  → deep memory / context
  theta  (4-8 Hz):  medium, 1 per 4 steps        → episodic / planning
  alpha  (8-13 Hz): fast, 1 per 2 steps           → attention gating
  gamma  (30+ Hz):  fastest, every step           → fast binding / signal

Each band is a small dense neuron group. Bands interact via cross-coupling
weights. The network learns to predict its own next state — prediction error
is both the learning signal (unsupervised) and a PPO feature (useful info).

No reward signal needed: the SNN gets better by itself at predicting the
compressed state stream. High error = surprising input = useful feature.
"""

import numpy as np


# Band update periods (in SNN steps)
BAND_PERIODS = {
    "delta": 8,
    "theta": 4,
    "alpha": 2,
    "gamma": 1,
}


class ResonantSNN:
    """
    Resonant frequency-band SNN.

    Instead of one monolithic synchronous network, neurons are partitioned
    into 4 bands. Each band:
      - Has its own dense weight matrix (fast numpy matmul)
      - Updates at its own rate (gamma=every step, delta=every 8)
      - Predicts its own next activity, learns from prediction error
      - Receives input from all bands (cross-coupling, small magnitude)

    The prediction error per band becomes an explicit feature for PPO.
    """

    def __init__(
        self,
        n_neurons: int = 64,
        input_dim: int = 16,
        band_split: tuple = (0.15, 0.20, 0.25, 0.40),  # delta/theta/alpha/gamma fractions
        learning_rate: float = 0.01,
        leak: float = 0.85,
        threshold: float = 0.3,
    ):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.leak = leak
        self.threshold = threshold
        self._step_count = 0

        # Partition neurons into bands
        assert abs(sum(band_split) - 1.0) < 1e-6, "band_split must sum to 1"
        sizes = []
        for i, frac in enumerate(band_split):
            if i < len(band_split) - 1:
                sizes.append(max(1, int(n_neurons * frac)))
            else:
                sizes.append(n_neurons - sum(sizes))

        self.bands = {}
        offset = 0
        for name, size in zip(["delta", "theta", "alpha", "gamma"], sizes):
            self.bands[name] = {
                "size": size,
                "offset": offset,
                "period": BAND_PERIODS[name],
                # Recurrent weights within band (dense, small init)
                "W_rec": np.random.randn(size, size) * 0.05,
                # Prediction weights: predict next activity from current
                "W_pred": np.random.randn(size, size) * 0.05,
                # Activity state
                "activity": np.zeros(size),
                "prediction": np.zeros(size),  # last prediction
                "pred_error": 0.0,
            }
            offset += size

        # Input projection: input_dim → n_neurons
        self.W_input = np.random.randn(n_neurons, input_dim) * 0.02

        # Cross-band coupling (all→all, small magnitude)
        # Shape: (n_neurons, n_neurons), but only off-diagonal blocks used
        self.W_cross = np.random.randn(n_neurons, n_neurons) * 0.01
        np.fill_diagonal(self.W_cross, 0.0)  # no self-coupling

        # Full activity vector (concatenation of all bands)
        self.activity = np.zeros(n_neurons)

        # Stats
        self._total_pred_error = 0.0
        self._total_spikes = 0
        self._n_pred_updates = 0

    @property
    def feature_dim(self) -> int:
        # Per band: mean_rate + pred_error + spike_fraction = 3 per band
        return len(self.bands) * 3

    def step(self, compressed_state: np.ndarray) -> np.ndarray:
        """
        One SNN step. Returns feature vector.

        Process:
          1. Compute input currents
          2. For each band scheduled this step: update activity + learn
          3. Extract features from all bands
        """
        # 1. Global input current (all neurons)
        input_current = self.W_input @ compressed_state  # (n_neurons,)

        # 2. Cross-band coupling (uses full activity from last step)
        cross_current = self.W_cross @ self.activity  # (n_neurons,)

        features = []
        offset = 0

        for name, band in self.bands.items():
            size = band["size"]
            sl = slice(offset, offset + size)

            # Only update this band on its scheduled steps
            if self._step_count % band["period"] == 0:
                # Prediction error from last prediction
                if self._step_count > 0:
                    error = band["activity"] - band["prediction"]
                    band["pred_error"] = float(np.mean(np.abs(error)))

                    # Self-supervised learning: adjust W_pred to reduce error
                    # dW = lr * error_outer(activity, activity)
                    dW = self.learning_rate * np.outer(error, band["activity"])
                    band["W_pred"] += dW
                    # Clip to prevent explosion
                    np.clip(band["W_pred"], -1.0, 1.0, out=band["W_pred"])
                    self._n_pred_updates += 1

                # Total input to this band
                total = (
                    input_current[sl]                    # sensory
                    + (band["W_rec"] @ band["activity"]) * 0.3   # recurrent
                    + cross_current[sl] * 0.1            # cross-band
                )

                # LIF update with leak
                new_activity = band["activity"] * self.leak + np.tanh(total)

                # Spike threshold
                spikes = (new_activity > self.threshold).astype(np.float32)
                new_activity = new_activity * (1 - spikes) + spikes * 0.1  # reset

                # Make prediction for next step
                band["prediction"] = np.tanh(band["W_pred"] @ new_activity)

                band["activity"] = new_activity
                self.activity[sl] = new_activity

            # Feature extraction (always, regardless of update)
            act = band["activity"]
            spikes = (act > self.threshold).astype(np.float32)

            mean_rate = float(np.mean(act))
            pred_err = float(band["pred_error"])
            spike_frac = float(np.mean(spikes))

            # Scale to small magnitude so features augment but don't dominate
            features.extend([
                np.tanh(mean_rate) * 0.1,
                np.tanh(pred_err) * 0.1,
                spike_frac * 0.1,
            ])

            self._total_spikes += int(np.sum(spikes))
            offset += size

        self._total_pred_error += sum(b["pred_error"] for b in self.bands.values())
        self._step_count += 1
        return np.array(features, dtype=np.float32)

    def reset(self):
        """Reset activity between episodes (keep learned weights)."""
        for band in self.bands.values():
            band["activity"] = np.zeros(band["size"])
            band["prediction"] = np.zeros(band["size"])
            # Keep pred_error for continuity
        self.activity = np.zeros(self.n_neurons)

    def stats(self) -> dict:
        return {
            "n_neurons": self.n_neurons,
            "bands": {
                name: {
                    "size": b["size"],
                    "period": b["period"],
                    "pred_error": round(b["pred_error"], 4),
                }
                for name, b in self.bands.items()
            },
            "step_count": self._step_count,
            "pred_updates": self._n_pred_updates,
            "avg_pred_error": round(
                self._total_pred_error / max(1, self._step_count), 4
            ),
            "avg_spikes_per_step": round(
                self._total_spikes / max(1, self._step_count), 2
            ),
            "feature_dim": self.feature_dim,
        }
