"""
grid_state.py — ASCII grid → learner-compatible state representations

Converts ASCII density grids into formats usable by:
  - Q-learners (hashed state keys or small feature vectors)
  - PPO/neural nets (flat float vectors)
  - Spatial analysis tools

The key idea: don't try to hash the full grid (10^300 states).
Instead, extract compact region features that generalize.
"""

import numpy as np
import hashlib
from typing import Optional


class GridState:
    """
    Converts ASCII density grids to compact state representations.

    Args:
        rows:         Grid height (must match AsciiEncoder)
        cols:         Grid width (must match AsciiEncoder)
        n_regions_r:  Number of row-wise regions for feature extraction
        n_regions_c:  Number of col-wise regions
        track_delta:  Whether to track frame-to-frame change features
    """

    def __init__(
        self,
        rows: int = 20,
        cols: int = 15,
        n_regions_r: int = 4,
        n_regions_c: int = 3,
        track_delta: bool = True,
    ):
        self.rows = rows
        self.cols = cols
        self.n_regions_r = n_regions_r
        self.n_regions_c = n_regions_c
        self.n_regions = n_regions_r * n_regions_c
        self.track_delta = track_delta

        self._prev_grid: Optional[np.ndarray] = None

        # Feature dims: mean+std per region × (current + optional delta)
        region_features = self.n_regions * 2  # mean + std
        delta_features = self.n_regions if track_delta else 0
        self.feature_dim = region_features + delta_features

    def region_features(self, grid: np.ndarray) -> np.ndarray:
        """
        Extract region-level statistics from a density grid.

        Divides grid into n_regions_r × n_regions_c blocks.
        Returns mean + std density per region.

        Returns: float32 vector of shape (n_regions * 2,), values in [0,1]
        """
        if grid.ndim == 3:
            grid = grid[0]

        H, W = grid.shape
        rh = H // self.n_regions_r
        rw = W // self.n_regions_c
        features = []

        for i in range(self.n_regions_r):
            for j in range(self.n_regions_c):
                block = grid[
                    i*rh:(i+1)*rh,
                    j*rw:(j+1)*rw
                ].astype(np.float32) / 9.0  # normalise to [0,1]
                features.append(float(block.mean()))
                features.append(float(block.std()))

        return np.array(features, dtype=np.float32)

    def features(self, grid: np.ndarray) -> np.ndarray:
        """
        Full feature vector: region stats + optional delta features.
        Call this every step.
        """
        reg = self.region_features(grid)

        if self.track_delta and self._prev_grid is not None:
            # Absolute change per region
            delta_grid = np.abs(
                grid.astype(np.float32) - self._prev_grid.astype(np.float32)
            )
            prev_gs = GridState(
                self.rows, self.cols,
                self.n_regions_r, self.n_regions_c,
                track_delta=False
            )
            delta_feats = prev_gs.region_features(delta_grid.astype(np.int8))
            result = np.concatenate([reg, delta_feats])
        else:
            if self.track_delta:
                result = np.concatenate([reg, np.zeros(self.n_regions, dtype=np.float32)])
            else:
                result = reg

        g = grid if grid.ndim == 2 else grid[0]
        self._prev_grid = g.copy()
        return result

    def flat(self, grid: np.ndarray) -> np.ndarray:
        """Full flattened grid as float32 in [0,1]. For neural nets."""
        if grid.ndim == 3:
            grid = grid[0]
        return grid.flatten().astype(np.float32) / 9.0

    def hash(self, grid: np.ndarray) -> str:
        """
        Stable string hash of the full grid for exact Q-table lookup.
        Only practical for small grids (<= 11×11).
        """
        if grid.ndim == 3:
            grid = grid[0]
        return hashlib.md5(grid.tobytes()).hexdigest()[:16]

    def quantized_key(self, grid: np.ndarray, levels: int = 3) -> tuple:
        """
        Reduce density levels for smaller Q-table (10 → levels buckets).
        levels=3 maps to: dark / mid / bright → manageable state space.
        """
        if grid.ndim == 3:
            grid = grid[0]
        quantized = (grid * levels // 10).clip(0, levels - 1)
        return tuple(quantized.flatten().tolist())

    def reset(self):
        """Call at episode start to clear delta tracking."""
        self._prev_grid = None

    def stats(self) -> dict:
        return {
            "grid_shape": (self.rows, self.cols),
            "n_regions": (self.n_regions_r, self.n_regions_c),
            "feature_dim": self.feature_dim,
            "flat_dim": self.rows * self.cols,
        }
