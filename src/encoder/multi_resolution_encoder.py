"""
multi_resolution_encoder.py — Global + Focal + Entity ASCII encoding.

The Problem:
    A single 15×20 ASCII grid (300 cells) merges information at cell
    boundaries. A Goomba at the left edge of a cell is represented
    identically to one at the right edge. Sub-cell position, entity
    proximity, and interaction geometry are lost.

The Solution:
    Three simultaneous encodings at different spatial resolutions:

    GLOBAL (15×20 = 300)
        Full scene overview. Navigation, room structure, entity positions.
        Same as current MarioSimulator obs[0:300].
        → z_global via projection (16-dim)
        → WM operates here (fast, coarse)

    FOCAL (N×N centered on player, default 7×7 = 49)
        Sub-region of the global grid centered on the player's position.
        Higher effective resolution because it covers less area.
        Captures: entity proximity, interaction geometry, local obstacles.
        → z_focal via projection (8-dim)
        → SNN operates here (temporal pattern in local context)

    ENTITY (variable, max ~20 entities × 4 features = 80)
        Per-entity feature vectors: (entity_type, rel_x, rel_y, distance).
        Direct encoding of all detected entities relative to player.
        No grid discretization — raw relative positions.
        Captures: exact entity relationships, interaction candidates.
        → z_entity via projection (8-dim)
        → UncertaintySeeker uses this for target navigation

    Combined state:
        z = concat(z_global, z_focal, z_entity)  → (32-dim default)
        This is what the WM and policy see.

    Resolution hierarchy:
        Global   → "there's something at tile (3,7)"
        Focal    → "the entity at (3,7) is at the top-left of its cell"
        Entity   → "entity is type=goomba, 2.3 tiles right, 0.5 tiles up"

Universality:
    This encoder works with:
      - Structured obs (MarioSimulator, ARC-AGI3): extract grid + player pos
      - Pixel frames (raw RGB): use AsciiEncoder to create grids first
      - RAM-only (Montezuma): skip focal/entity, use global only

Usage:
    enc = MultiResolutionEncoder(
        global_shape=(15, 20),
        focal_size=7,
        z_global_dim=16,
        z_focal_dim=8,
        z_entity_dim=8,
    )

    # From structured obs (like MarioSimulator):
    z = enc.encode_structured(
        global_grid=obs[:300].reshape(15, 20),
        player_pos=(mario_row, mario_col),
        entities=[(type_id, row, col), ...],
    )

    # From pixel frame:
    z = enc.encode_pixels(rgb_frame, player_pos_pixels=(px, py))
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .ascii_encoder import AsciiEncoder, N_LEVELS


# ══════════════════════════════════════════════════════════════
#  Numpy linear layer (no PyTorch dependency)
# ══════════════════════════════════════════════════════════════

class NumpyLinear:
    """Simple dense layer for projection. Xavier init."""

    def __init__(self, in_dim: int, out_dim: int):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = np.random.randn(out_dim, in_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (in_dim,) or (N, in_dim) → (out_dim,) or (N, out_dim)"""
        return np.tanh(x @ self.W.T + self.b)

    def forward_batch(self, x: np.ndarray) -> np.ndarray:
        """x: (N, in_dim) → (N, out_dim)"""
        return np.tanh(x @ self.W.T + self.b)


# ══════════════════════════════════════════════════════════════
#  Entity feature extraction
# ══════════════════════════════════════════════════════════════

# Standard entity types (extendable)
ENTITY_TYPES = {
    'empty':    0,
    'ground':   1,
    'block':    2,
    'enemy':    3,
    'coin':     4,
    'pipe':     5,
    'player':   6,
    'door':     7,
    'key':      8,
    'switch':   9,
    'wall':     10,
    'object':   11,
    'unknown':  12,
}
N_ENTITY_TYPES = len(ENTITY_TYPES)


@dataclass
class DetectedEntity:
    """An entity detected in the environment."""
    type_id:   int          # ENTITY_TYPES value
    type_name: str          # human-readable name
    grid_row:  float        # row in global grid (can be fractional)
    grid_col:  float        # col in global grid
    rel_row:   float = 0.0  # relative to player (positive = below)
    rel_col:   float = 0.0  # relative to player (positive = right)
    distance:  float = 0.0  # Euclidean distance to player
    density:   int   = 0    # density level in the grid cell (0-9)

    @property
    def feature_vector(self) -> np.ndarray:
        """(type_normalized, rel_row, rel_col, distance) → 4-float feature."""
        return np.array([
            self.type_id / N_ENTITY_TYPES,
            self.rel_row / 15.0,   # normalize to ~[-1, 1]
            self.rel_col / 20.0,
            min(self.distance / 25.0, 1.0),  # cap at 1.0
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  MultiResolutionEncoder
# ══════════════════════════════════════════════════════════════

class MultiResolutionEncoder:
    """
    Three-tier encoding: Global + Focal + Entity → combined z-vector.

    Args:
        global_shape:    (rows, cols) of the global grid.
        focal_size:      Side length of the focal grid (square, centered on player).
        z_global_dim:    Dimensionality of global z-vector.
        z_focal_dim:     Dimensionality of focal z-vector.
        z_entity_dim:    Dimensionality of entity z-vector.
        max_entities:    Maximum entities in the entity feature vector.
        pixel_encoder:   Optional AsciiEncoder for pixel-based input.
    """

    def __init__(
        self,
        global_shape:    Tuple[int, int] = (15, 20),
        focal_size:      int             = 7,
        z_global_dim:    int             = 16,
        z_focal_dim:     int             = 8,
        z_entity_dim:    int             = 8,
        max_entities:    int             = 16,
        pixel_encoder:   Optional[AsciiEncoder] = None,
    ):
        self.global_rows, self.global_cols = global_shape
        self.focal_size    = focal_size
        self.z_global_dim  = z_global_dim
        self.z_focal_dim   = z_focal_dim
        self.z_entity_dim  = z_entity_dim
        self.max_entities  = max_entities

        # Input dimensions
        self.global_flat_dim = self.global_rows * self.global_cols
        self.focal_flat_dim  = focal_size * focal_size
        self.entity_flat_dim = max_entities * 4  # 4 features per entity

        # Projection layers (numpy, no PyTorch)
        self._proj_global = NumpyLinear(self.global_flat_dim, z_global_dim)
        self._proj_focal  = NumpyLinear(self.focal_flat_dim,  z_focal_dim)
        self._proj_entity = NumpyLinear(self.entity_flat_dim, z_entity_dim)

        # Optional pixel encoder for raw frame input
        self._pixel_enc = pixel_encoder or AsciiEncoder(
            rows=self.global_rows, cols=self.global_cols
        )

        # Total z-dim
        self.z_dim = z_global_dim + z_focal_dim + z_entity_dim

        # Stats
        self._encode_count = 0

    # ── Primary encoding methods ─────────────────────────────

    def encode_structured(
        self,
        global_grid:   np.ndarray,
        player_pos:    Tuple[float, float],
        entities:      Optional[List[Tuple]] = None,
    ) -> np.ndarray:
        """
        Encode from structured observation (like MarioSimulator obs).

        Args:
            global_grid: (rows, cols) int/float grid (values 0-9 or 0-1).
            player_pos:  (row, col) of the player in grid coordinates.
            entities:    Optional list of (type_name_or_id, row, col) tuples.

        Returns:
            z: (z_dim,) float32 combined representation.
        """
        # 1. Global encoding
        z_global = self._encode_global(global_grid)

        # 2. Focal encoding (centered on player)
        focal_grid = self._extract_focal(global_grid, player_pos)
        z_focal = self._encode_focal(focal_grid)

        # 3. Entity encoding
        z_entity = self._encode_entities(entities, player_pos)

        # Combine
        z = np.concatenate([z_global, z_focal, z_entity])
        self._encode_count += 1
        return z

    def encode_pixels(
        self,
        frame:             np.ndarray,
        player_pos_pixels: Optional[Tuple[int, int]] = None,
        entities:          Optional[List[Tuple]]      = None,
    ) -> np.ndarray:
        """
        Encode from raw pixel frame.

        Args:
            frame:             (H, W, 3) RGB image, uint8.
            player_pos_pixels: (y, x) pixel position of player.
                               If None, uses center of frame.
            entities:          Optional detected entities.

        Returns:
            z: (z_dim,) float32 combined representation.
        """
        # Convert pixels to ASCII grid
        global_grid = self._pixel_enc.encode(frame)

        # Convert pixel position to grid position
        if player_pos_pixels is not None:
            H, W = frame.shape[:2]
            player_row = player_pos_pixels[0] / H * self.global_rows
            player_col = player_pos_pixels[1] / W * self.global_cols
        else:
            player_row = self.global_rows / 2
            player_col = self.global_cols / 2

        return self.encode_structured(
            global_grid=global_grid,
            player_pos=(player_row, player_col),
            entities=entities,
        )

    def encode_flat_obs(
        self,
        obs:         np.ndarray,
        grid_slice:  slice      = slice(0, 300),
        player_row_idx: int     = 320,
        player_col_idx: int     = 321,
        grid_shape:  Tuple[int, int] = (15, 20),
    ) -> np.ndarray:
        """
        Encode from a flat observation vector (like MarioSimulator's 378-dim obs).

        This is the fast-path for the training loop — directly extracts
        the grid and player position from the flat obs without any
        pixel conversion.

        Args:
            obs:            Flat observation vector.
            grid_slice:     Slice indices for the grid portion of obs.
            player_row_idx: Index of player row in obs (normalized 0-1).
            player_col_idx: Index of player col in obs (normalized 0-1).
            grid_shape:     (rows, cols) of the encoded grid.

        Returns:
            z: (z_dim,) float32 combined representation.
        """
        rows, cols = grid_shape
        grid = obs[grid_slice].reshape(rows, cols)

        # Player position (denormalize from 0-1 range)
        p_row = obs[player_row_idx] * rows if player_row_idx < len(obs) else rows / 2
        p_col = obs[player_col_idx] * cols if player_col_idx < len(obs) else cols / 2

        return self.encode_structured(
            global_grid=grid,
            player_pos=(p_row, p_col),
            entities=None,  # Entity detection from flat obs TBD
        )

    def encode_flat_obs_batch(
        self,
        obs_batch:      np.ndarray,
        grid_slice:     slice      = slice(0, 300),
        player_row_idx: int        = 320,
        player_col_idx: int        = 321,
        grid_shape:     Tuple[int, int] = (15, 20),
    ) -> np.ndarray:
        """
        Batch encode N flat observations → (N, z_dim).

        This is the vectorized path for VectorizedImaginedEnv compatibility.
        """
        N = obs_batch.shape[0]
        rows, cols = grid_shape

        # Extract grids: (N, rows*cols) → normalize
        grids_flat = obs_batch[:, grid_slice]  # (N, 300)

        # Global projection (batched)
        z_globals = self._proj_global.forward_batch(grids_flat)  # (N, z_global_dim)

        # Player positions
        p_rows = obs_batch[:, player_row_idx] * rows  # (N,)
        p_cols = obs_batch[:, player_col_idx] * cols  # (N,)

        # Focal grids (per-env, harder to batch but fast in numpy)
        focal_flat = np.zeros((N, self.focal_flat_dim), dtype=np.float32)
        for i in range(N):
            grid_2d = grids_flat[i].reshape(rows, cols)
            focal = self._extract_focal(grid_2d, (p_rows[i], p_cols[i]))
            focal_flat[i] = focal.flatten()

        z_focals = self._proj_focal.forward_batch(focal_flat)    # (N, z_focal_dim)

        # Entity placeholder (no entities from flat obs currently)
        z_entities = np.zeros((N, self.z_entity_dim), dtype=np.float32)

        return np.concatenate([z_globals, z_focals, z_entities], axis=1)  # (N, z_dim)

    # ── Internal encoding helpers ────────────────────────────

    def _encode_global(self, grid: np.ndarray) -> np.ndarray:
        """Global grid → z_global."""
        flat = grid.flatten().astype(np.float32)
        # Normalize to [0, 1] if integer grid
        if flat.max() > 1.0:
            flat = flat / (N_LEVELS - 1)
        return self._proj_global.forward(flat)

    def _encode_focal(self, focal_grid: np.ndarray) -> np.ndarray:
        """Focal grid → z_focal."""
        flat = focal_grid.flatten().astype(np.float32)
        if flat.max() > 1.0:
            flat = flat / (N_LEVELS - 1)
        return self._proj_focal.forward(flat)

    def _encode_entities(
        self,
        entities:   Optional[List[Tuple]],
        player_pos: Tuple[float, float],
    ) -> np.ndarray:
        """Entity list → z_entity."""
        feat_vec = np.zeros(self.entity_flat_dim, dtype=np.float32)

        if entities is None or len(entities) == 0:
            return self._proj_entity.forward(feat_vec)

        p_row, p_col = player_pos
        for i, ent in enumerate(entities[:self.max_entities]):
            if len(ent) == 3:
                type_key, e_row, e_col = ent
            else:
                continue

            # Resolve type
            if isinstance(type_key, str):
                type_id = ENTITY_TYPES.get(type_key, ENTITY_TYPES['unknown'])
            else:
                type_id = int(type_key)

            rel_row = e_row - p_row
            rel_col = e_col - p_col
            dist    = np.sqrt(rel_row**2 + rel_col**2)

            de = DetectedEntity(
                type_id=type_id, type_name=str(type_key),
                grid_row=e_row, grid_col=e_col,
                rel_row=rel_row, rel_col=rel_col,
                distance=dist,
            )
            feat_vec[i*4:(i+1)*4] = de.feature_vector

        return self._proj_entity.forward(feat_vec)

    def _extract_focal(
        self,
        global_grid: np.ndarray,
        player_pos:  Tuple[float, float],
    ) -> np.ndarray:
        """
        Extract focal grid centered on player position.

        Handles boundary padding: if the player is near an edge,
        out-of-bounds cells are filled with 0 (empty).
        """
        rows, cols = global_grid.shape
        p_row, p_col = int(round(player_pos[0])), int(round(player_pos[1]))
        half = self.focal_size // 2

        focal = np.zeros((self.focal_size, self.focal_size), dtype=global_grid.dtype)

        for fr in range(self.focal_size):
            for fc in range(self.focal_size):
                gr = p_row - half + fr
                gc = p_col - half + fc
                if 0 <= gr < rows and 0 <= gc < cols:
                    focal[fr, fc] = global_grid[gr, gc]

        return focal

    # ── Properties & stats ───────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Total z-vector dimensionality."""
        return self.z_dim

    def describe_resolution(self) -> dict:
        """Describe what each tier sees."""
        global_cell_area = 1.0  # normalized
        focal_cell_area  = (self.focal_size / self.global_rows) * (self.focal_size / self.global_cols)

        return {
            "global": {
                "shape": (self.global_rows, self.global_cols),
                "cells": self.global_flat_dim,
                "z_dim": self.z_global_dim,
                "coverage": "100% of scene",
            },
            "focal": {
                "shape": (self.focal_size, self.focal_size),
                "cells": self.focal_flat_dim,
                "z_dim": self.z_focal_dim,
                "coverage": f"{focal_cell_area*100:.1f}% of scene (centered on player)",
                "effective_resolution": f"{1/focal_cell_area:.1f}× relative to global",
            },
            "entity": {
                "max_entities": self.max_entities,
                "features_per": 4,
                "z_dim": self.z_entity_dim,
                "resolution": "exact relative positions (no grid discretization)",
            },
            "total_z_dim": self.z_dim,
        }

    def stats(self) -> dict:
        return {
            "z_dim":         self.z_dim,
            "encode_count":  self._encode_count,
            "resolutions":   self.describe_resolution(),
        }
