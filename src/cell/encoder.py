"""
Trainable State Encoder — Converts raw observations to compressed representations.

End-to-end trainable with PPO (gradients flow through).
The encoder learns what compression is useful for policy performance,
not reconstruction accuracy (Hoffman: compress for fitness, not fidelity).
"""

import torch
import torch.nn as nn
import numpy as np


class StateEncoder(nn.Module):
    """
    Trainable MLP encoder: obs_dim → compressed_dim.

    Learns a compressed representation of raw observations that
    maximizes downstream policy performance (not reconstruction).
    """

    def __init__(self, obs_dim: int, compressed_dim: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.compressed_dim = compressed_dim

        # Small MLP — intentionally shallow so it doesn't overshadow the SNN
        hidden = max(32, compressed_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, compressed_dim),
            nn.Tanh(),  # Bounded output for SNN compatibility
        )

        # Stats tracking
        self._encode_count = 0
        self._total_magnitude = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (differentiable, for end-to-end training with PPO)."""
        return self.net(x)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode a single observation (numpy in, numpy out, no grad)."""
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            features = self.net(x).squeeze(0).cpu().numpy()

        # Track stats
        self._encode_count += 1
        self._total_magnitude += np.linalg.norm(features)
        return features

    def stats(self) -> dict:
        """Encoding statistics."""
        avg_mag = (self._total_magnitude / max(1, self._encode_count))
        return {
            "encode_count": self._encode_count,
            "avg_feature_magnitude": round(avg_mag, 4),
            "compression_ratio": round(self.obs_dim / self.compressed_dim, 2),
        }
