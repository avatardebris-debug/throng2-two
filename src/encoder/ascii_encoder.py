"""
ascii_encoder.py — Pixel frame → ASCII grid converter

Converts Atari RGB frames (or any image) into a compact ASCII grid.
The grid is the "compressed interface" to the environment — small enough
for tabular methods, structured enough to preserve spatial meaning.

Design:
  - 10 density levels: " .:-=+*#%@" (space=dark, @=bright)
  - Configurable grid size (default 20×15 for Montezuma)
  - Optional per-channel grids (R/G/B) for entity detection by color
  - Optional delta mode (changes from previous frame)
  - Target: <0.5ms per frame on CPU

Usage:
    enc = AsciiEncoder(rows=20, cols=15)
    grid = enc.encode(rgb_frame)        # (20, 15) int array, values 0-9
    text = enc.to_text(grid)            # human-readable string
    flat = grid.flatten().astype(float) # (300,) for MLP input
"""

import numpy as np


# ASCII density levels: space (dark) → @ (bright)
DENSITY_CHARS = " .:-=+*#%@"
N_LEVELS = len(DENSITY_CHARS)  # 10


class AsciiEncoder:
    """
    Converts pixel frames to ASCII density grids.

    Args:
        rows: Grid height (default 20)
        cols: Grid width (default 15)
        color_channels: If True, produce 3 separate grids (R, G, B)
        delta_mode: If True, return change from previous frame
        grayscale_weights: RGB → grayscale weighting (luminance standard)
    """

    def __init__(
        self,
        rows: int = 20,
        cols: int = 15,
        color_channels: bool = False,
        delta_mode: bool = False,
        grayscale_weights: tuple = (0.299, 0.587, 0.114),
    ):
        self.rows = rows
        self.cols = cols
        self.color_channels = color_channels
        self.delta_mode = delta_mode
        self.grayscale_weights = np.array(grayscale_weights, dtype=np.float32)

        self._prev_grid = None

        # Feature dimension exposed publicly
        n_grids = 3 if color_channels else 1
        if delta_mode:
            n_grids *= 2  # current + delta
        self.feature_dim = rows * cols * n_grids

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode an RGB frame into an ASCII density grid.

        Args:
            frame: numpy array of shape (H, W, 3), dtype uint8

        Returns:
            grid: numpy array of shape (rows, cols) or (3, rows, cols),
                  dtype int8, values 0-9
        """
        if self.color_channels:
            grids = []
            for c in range(3):
                channel = frame[:, :, c].astype(np.float32)
                grids.append(self._pool_to_grid(channel))
            grid = np.stack(grids, axis=0)  # (3, rows, cols)
        else:
            # Grayscale conversion
            gray = (frame.astype(np.float32) @ self.grayscale_weights)
            grid = self._pool_to_grid(gray)  # (rows, cols)

        if self.delta_mode:
            if self._prev_grid is not None:
                delta = (grid.astype(np.int16) - self._prev_grid.astype(np.int16))
                delta = np.clip(delta + 5, 0, 9).astype(np.int8)
                grid = np.concatenate([
                    grid[np.newaxis] if grid.ndim == 2 else grid,
                    delta[np.newaxis] if delta.ndim == 2 else delta,
                ], axis=0)
            self._prev_grid = (grid[0] if grid.ndim == 3 else grid).copy()
        else:
            self._prev_grid = grid.copy()

        return grid

    def _pool_to_grid(self, channel: np.ndarray) -> np.ndarray:
        """Pool a 2D channel down to (rows, cols) density grid."""
        H, W = channel.shape
        # Use reshaping for fast average pooling
        # Crop to exact multiple of grid size
        h_crop = (H // self.rows) * self.rows
        w_crop = (W // self.cols) * self.cols
        cropped = channel[:h_crop, :w_crop]

        # Average pool: (rows, block_h, cols, block_w) → (rows, cols)
        pooled = cropped.reshape(
            self.rows, h_crop // self.rows,
            self.cols, w_crop // self.cols
        ).mean(axis=(1, 3))  # (rows, cols)

        # Map 0-255 → 0-9 density levels
        density = (pooled / 255.0 * (N_LEVELS - 1)).astype(np.int8)
        return density

    def to_text(self, grid: np.ndarray) -> str:
        """Convert density grid to human-readable ASCII string."""
        if grid.ndim == 3:
            grid = grid[0]  # use first channel for display
        lines = []
        for row in grid:
            lines.append("".join(DENSITY_CHARS[v] for v in row))
        return "\n".join(lines)

    def flat_features(self, grid: np.ndarray) -> np.ndarray:
        """Flatten grid to float32 vector in [0,1] for neural net input."""
        return (grid.flatten().astype(np.float32)) / (N_LEVELS - 1)

    def reset(self):
        """Reset delta state (call at episode start)."""
        self._prev_grid = None

    def stats(self) -> dict:
        return {
            "grid_shape": (self.rows, self.cols),
            "color_channels": self.color_channels,
            "delta_mode": self.delta_mode,
            "feature_dim": self.feature_dim,
            "chars": DENSITY_CHARS,
        }
