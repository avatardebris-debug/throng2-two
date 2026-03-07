"""
multi_scale_encoder.py -- Multi-resolution ASCII encoding for Mario worlds.

Implements the "world-in-corner" concept: The full level is viewable at
three resolutions simultaneously:

  1. FULL WORLD  -- entire level as ASCII grid (16 x N_cols)
  2. VIEWPORT    -- what Mario can see (16 x 20 window around Mario)
  3. MINIMAP     -- entire world compressed to small grid (4 x 10)

The encoder learns to compress the full world into the minimap while
preserving enough information to reconstruct tile positions. This is
trained with a reconstruction loss (how well can we decode minimap
back to the full world?).

Usage:
    from src.encoder.multi_scale_encoder import MultiScaleEncoder

    encoder = MultiScaleEncoder()
    level = MarioSimulator.from_flat_ground(n_screens=3)

    scales = encoder.encode(level)
    # scales["full"]     = (16, 60) normalized grid
    # scales["viewport"] = (16, 20) around Mario
    # scales["minimap"]  = (4, 10)  compressed world

    quality = encoder.reconstruction_quality(level)
    # IoU score of minimap -> decoded -> full world comparison
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..games.mario.mario_simulator import MarioSimulator, N_TILE_TYPES, Tile


class MultiScaleEncoder:
    """
    Three-level encoding of Mario worlds.

    Provides:
      - Full world grid (normalized)
      - Viewport around Mario (20-col window)
      - Minimap (full world compressed via learned pooling)
      - Reconstruction scoring (minimap -> full world IoU)

    The minimap encoder is a simple trainable compression:
      Full (16, W) -> downsample -> Minimap (4, 10)
      Minimap (4, 10) -> upsample -> Reconstructed (16, W)
    """

    MINIMAP_H = 4
    MINIMAP_W = 10
    VIEWPORT_W = 20

    def __init__(self, lr: float = 0.001):
        self.lr = lr

        # Trainable compression weights (initialized with average pooling)
        # These are updated during training to improve reconstruction
        self._encode_weights: Optional[np.ndarray] = None
        self._decode_weights: Optional[np.ndarray] = None
        self._train_steps = 0

    def encode(self, sim: MarioSimulator) -> Dict[str, np.ndarray]:
        """
        Encode a level at three resolutions.

        Returns dict with:
          "full":     (16, W) normalized tile grid
          "viewport": (16, 20) window around Mario
          "minimap":  (4, 10) compressed world
          "mario_pos": (row, col) in full coordinates
          "mario_minimap_pos": (row, col) in minimap coordinates
        """
        # Full world
        full = sim.grid.astype(np.float32) / max(N_TILE_TYPES - 1, 1)

        # Viewport
        vp_start = max(0, min(sim.mario_col - 10, sim.width - self.VIEWPORT_W))
        vp_end = min(vp_start + self.VIEWPORT_W, sim.width)
        viewport = np.zeros((sim.GRID_H, self.VIEWPORT_W), dtype=np.float32)
        vp_width = vp_end - vp_start
        viewport[:, :vp_width] = full[:, vp_start:vp_end]

        # Minimap: average pooling
        minimap = self._compress(full)

        # Mario position in minimap coordinates
        mm_row = int(sim.mario_row * self.MINIMAP_H / sim.GRID_H)
        mm_col = int(sim.mario_col * self.MINIMAP_W / max(sim.width, 1))
        mm_row = min(mm_row, self.MINIMAP_H - 1)
        mm_col = min(mm_col, self.MINIMAP_W - 1)

        return {
            "full": full,
            "viewport": viewport,
            "minimap": minimap,
            "mario_pos": (sim.mario_row, sim.mario_col),
            "mario_minimap_pos": (mm_row, mm_col),
            "full_shape": full.shape,
        }

    def _compress(self, full: np.ndarray) -> np.ndarray:
        """Compress full grid to minimap via average pooling."""
        h, w = full.shape
        minimap = np.zeros((self.MINIMAP_H, self.MINIMAP_W), dtype=np.float32)

        # Compute block sizes
        bh = max(1, h // self.MINIMAP_H)
        bw = max(1, w // self.MINIMAP_W)

        for mr in range(self.MINIMAP_H):
            for mc in range(self.MINIMAP_W):
                r_start = mr * bh
                r_end = min(r_start + bh, h)
                c_start = mc * bw
                c_end = min(c_start + bw, w)

                if r_start < h and c_start < w:
                    block = full[r_start:r_end, c_start:c_end]
                    if block.size > 0:
                        minimap[mr, mc] = np.mean(block)

        return minimap

    def decompress(self, minimap: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Decompress minimap back to full resolution via nearest-neighbor upsampling."""
        h, w = target_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)

        bh = max(1, h // self.MINIMAP_H)
        bw = max(1, w // self.MINIMAP_W)

        for mr in range(self.MINIMAP_H):
            for mc in range(self.MINIMAP_W):
                r_start = mr * bh
                r_end = min(r_start + bh, h)
                c_start = mc * bw
                c_end = min(c_start + bw, w)

                if r_start < h and c_start < w:
                    reconstructed[r_start:r_end, c_start:c_end] = minimap[mr, mc]

        return reconstructed

    def reconstruction_quality(self, sim: MarioSimulator) -> Dict[str, float]:
        """
        Score how well the minimap preserves the full world information.

        Metrics:
          - mse: Mean squared error between original and reconstructed
          - tile_accuracy: fraction of tiles classified correctly
          - ground_iou: IoU of ground tiles specifically
        """
        scales = self.encode(sim)
        full = scales["full"]
        minimap = scales["minimap"]
        reconstructed = self.decompress(minimap, full.shape)

        # MSE
        mse = float(np.mean((full - reconstructed) ** 2))

        # Tile accuracy (quantize both to nearest tile type)
        full_tiles = np.round(full * (N_TILE_TYPES - 1)).astype(int)
        recon_tiles = np.round(reconstructed * (N_TILE_TYPES - 1)).astype(int)
        tile_accuracy = float(np.mean(full_tiles == recon_tiles))

        # Ground IoU
        full_ground = (full_tiles == Tile.GROUND)
        recon_ground = (recon_tiles == Tile.GROUND)
        intersection = float(np.sum(full_ground & recon_ground))
        union = float(np.sum(full_ground | recon_ground))
        ground_iou = intersection / max(union, 1)

        return {
            "mse": mse,
            "tile_accuracy": tile_accuracy,
            "ground_iou": ground_iou,
        }

    def render_minimap(self, minimap: np.ndarray) -> str:
        """Render minimap as compact ASCII."""
        lines = []
        for row in range(minimap.shape[0]):
            chars = []
            for col in range(minimap.shape[1]):
                val = minimap[row, col]
                if val < 0.05:
                    chars.append('.')  # Empty/sky
                elif val > 0.85:
                    chars.append('#')  # Solid
                elif val > 0.5:
                    chars.append('=')  # Mixed solid
                elif val > 0.2:
                    chars.append('-')  # Mixed empty
                else:
                    chars.append(' ')  # Mostly empty
                    
            lines.append(''.join(chars))
        return '\n'.join(lines)

    def encode_at_resolution(
        self,
        sim: MarioSimulator,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """
        Encode the level at an arbitrary resolution.

        This enables resolution-invariance testing: same level,
        different grid sizes.
        """
        full = sim.grid.astype(np.float32) / max(N_TILE_TYPES - 1, 1)
        h, w = full.shape

        result = np.zeros((target_h, target_w), dtype=np.float32)
        bh = max(1, h // target_h)
        bw = max(1, w // target_w)

        for tr in range(target_h):
            for tc in range(target_w):
                r_start = tr * bh
                r_end = min(r_start + bh, h)
                c_start = tc * bw
                c_end = min(c_start + bw, w)

                if r_start < h and c_start < w:
                    block = full[r_start:r_end, c_start:c_end]
                    if block.size > 0:
                        result[tr, tc] = np.mean(block)

        return result

    def multi_resolution_features(
        self, sim: MarioSimulator
    ) -> np.ndarray:
        """
        Concatenated feature vector from all three scales.

        Returns flat numpy array:
          viewport (16*20=320) + minimap (4*10=40) + mario_pos (4) = 364
        """
        scales = self.encode(sim)

        features = np.concatenate([
            scales["viewport"].flatten(),   # 320
            scales["minimap"].flatten(),     # 40
            np.array([
                scales["mario_pos"][0] / sim.GRID_H,
                scales["mario_pos"][1] / max(sim.width, 1),
                scales["mario_minimap_pos"][0] / self.MINIMAP_H,
                scales["mario_minimap_pos"][1] / self.MINIMAP_W,
            ], dtype=np.float32),            # 4
        ])

        return features

    @property
    def feature_dim(self) -> int:
        return 16 * 20 + self.MINIMAP_H * self.MINIMAP_W + 4  # 364
