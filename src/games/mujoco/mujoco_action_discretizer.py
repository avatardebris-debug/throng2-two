"""
mujoco_action_discretizer.py — Discretize MuJoCo's continuous action space.

MuJoCo environments have continuous actions (joint torques), but our
agent architecture uses discrete actions. This module bridges that gap
with three strategies:

  Strategy 1: Ternary per-joint  {-1, 0, +1} × N_joints
    → 3^N_joints actions. Simple and effective for low-DOF tasks.
    Reacher (2 joints) → 9 actions = manageable.
    HalfCheetah (6 joints) → 729 actions = too many; use primitives instead.

  Strategy 2: Action primitives  (pre-defined macro-actions)
    A hand-crafted set of meaningful movements, e.g.:
    "all joints forward", "reach left", "reach right", "freeze"
    → 8-16 actions regardless of joint count.

  Strategy 3: K-means from demos  (cluster expert trajectories)
    Given a set of continuous action trajectories (from a reference policy
    or random rollouts), cluster into K centroids.
    → K configurable actions; adapts to any task.

Usage:
    disc = MuJoCoActionDiscretizer(n_joints=2, strategy="ternary")
    action_idx = disc.encode(np.array([0.5, -0.3]))   # continuous → idx
    continuous  = disc.decode(action_idx)              # idx → continuous
    n = disc.n_actions                                 # 9 for 2 joints

    # K-means (fit from random actions first)
    disc_km = MuJoCoActionDiscretizer(n_joints=6, strategy="kmeans", k=32)
    disc_km.fit(random_actions_array)   # (N, 6) array of continuous actions
    idx = disc_km.encode(action)
    cont = disc_km.decode(idx)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np


# ═══════════════════════════════════════════════════════════════
# STRATEGY: TERNARY PER-JOINT
# ═══════════════════════════════════════════════════════════════

def _build_ternary_table(n_joints: int, scale: float = 1.0) -> np.ndarray:
    """
    Build a ternary action table.

    Each action is a combination of {-scale, 0, +scale} per joint.
    Action 0 = all joints at -scale, Action 3^N-1 = all joints at +scale.

    Returns:
        table: (3^n_joints, n_joints) float32 array
    """
    n_actions = 3 ** n_joints
    table = np.zeros((n_actions, n_joints), dtype=np.float32)
    for i in range(n_actions):
        code = i
        for j in range(n_joints):
            table[i, j] = float(code % 3 - 1) * scale   # -1, 0, or +1 (scaled)
            code //= 3
    return table


# ═══════════════════════════════════════════════════════════════
# STRATEGY: ACTION PRIMITIVES
# ═══════════════════════════════════════════════════════════════

def _build_primitive_table(n_joints: int, action_dim: int, scale: float = 1.0) -> np.ndarray:
    """
    Build a hand-crafted primitive action table.

    Primitives:
        0: Zero (freeze / hold position)
        1: All joints positive max
        2: All joints negative max
        3..N+2: One joint positive, others zero  (N actions)
        N+3..2N+2: One joint negative, others zero (N actions)
        Total: 3 + 2*N_joints actions

    Args:
        n_joints: Number of controllable joints.
        action_dim: Actual action vector dimension (may differ from n_joints).
        scale: Torque magnitude for non-zero actions.
    """
    primitives = []

    # 0: Freeze
    primitives.append(np.zeros(action_dim, dtype=np.float32))

    # 1: All forward
    a = np.zeros(action_dim, dtype=np.float32)
    a[:n_joints] = scale
    primitives.append(a)

    # 2: All backward
    a = np.zeros(action_dim, dtype=np.float32)
    a[:n_joints] = -scale
    primitives.append(a)

    # Per-joint positive
    for j in range(min(n_joints, action_dim)):
        a = np.zeros(action_dim, dtype=np.float32)
        a[j] = scale
        primitives.append(a)

    # Per-joint negative
    for j in range(min(n_joints, action_dim)):
        a = np.zeros(action_dim, dtype=np.float32)
        a[j] = -scale
        primitives.append(a)

    # Pairs (first joint + second joint): diagonal moves
    if n_joints >= 2 and action_dim >= 2:
        for sign_a in [1.0, -1.0]:
            for sign_b in [1.0, -1.0]:
                a = np.zeros(action_dim, dtype=np.float32)
                a[0] = sign_a * scale
                a[1] = sign_b * scale
                primitives.append(a)

    return np.stack(primitives, axis=0)


# ═══════════════════════════════════════════════════════════════
# MAIN DISCRETIZER
# ═══════════════════════════════════════════════════════════════

class MuJoCoActionDiscretizer:
    """
    Maps discrete integer actions ↔ continuous MuJoCo torque vectors.

    Interface (same for all strategies):
        disc.decode(action_idx) → np.ndarray (continuous torques)
        disc.encode(torque_vec) → int (nearest discrete action)
        disc.n_actions           → int

    Strategies:
        "ternary"   — per-joint {-1, 0, +1} (good for N_joints ≤ 4)
        "primitive" — hand-crafted macro-actions (good for any N_joints)
        "kmeans"    — data-driven clustering (requires fit() call)
    """

    def __init__(
        self,
        n_joints: int = 2,
        action_dim: Optional[int] = None,
        strategy: str = "ternary",
        k: int = 16,                   # K for kmeans
        scale: float = 1.0,            # Torque magnitude for ternary/primitive
        seed: int = 42,
    ):
        """
        Args:
            n_joints: Number of joint degrees of freedom.
            action_dim: Total continuous action dimension (defaults to n_joints).
            strategy: One of "ternary", "primitive", "kmeans".
            k: Number of clusters for kmeans strategy.
            scale: Torque magnitude for ternary/primitive strategies (0-1 range).
            seed: RNG seed.
        """
        self.n_joints = n_joints
        self.action_dim = action_dim if action_dim is not None else n_joints
        self.strategy = strategy
        self.k = k
        self.scale = scale
        self._rng = np.random.RandomState(seed)

        # Build lookup table
        self._table: Optional[np.ndarray] = None
        self._is_fitted = False
        self._build()

    def _build(self):
        """Build the discrete action table based on strategy."""
        if self.strategy == "ternary":
            # Only use ternary if N_joints ≤ 5 (3^6 = 729 is too many actions)
            if self.n_joints <= 5:
                self._table = _build_ternary_table(self.n_joints, self.scale)
            else:
                import warnings
                warnings.warn(
                    f"MuJoCoActionDiscretizer: ternary strategy requested but n_joints={self.n_joints} > 5 "
                    f"({3**self.n_joints} actions). Automatically falling back to 'primitive' strategy. "
                    "Set strategy='primitive' explicitly to suppress this warning.",
                    UserWarning,
                    stacklevel=3,
                )
                self._table = _build_primitive_table(
                    self.n_joints, self.action_dim, self.scale
                )

        elif self.strategy == "primitive":
            self._table = _build_primitive_table(
                self.n_joints, self.action_dim, self.scale
            )

        elif self.strategy == "kmeans":
            # Initialize with random unit vectors; will be replaced by fit()
            self._table = self._rng.uniform(
                -self.scale, self.scale, (self.k, self.action_dim)
            ).astype(np.float32)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}. "
                             f"Valid: 'ternary', 'primitive', 'kmeans'")

        self._is_fitted = (self.strategy != "kmeans")

    def fit(self, actions: np.ndarray, n_iters: int = 50):
        """
        Fit k-means clustering to a set of continuous actions.

        Args:
            actions: (N, action_dim) array of continuous action samples.
            n_iters: Number of k-means iterations.

        Raises:
            ValueError: if fewer samples than clusters requested.
        """
        if self.strategy != "kmeans":
            raise RuntimeError("fit() is only for kmeans strategy")

        actions = np.asarray(actions, dtype=np.float32)
        if len(actions) < self.k:
            raise ValueError(
                f"k-means fit requires at least k={self.k} samples, got {len(actions)}. "
                "Either provide more action samples or reduce k."
            )

        # Simple k-means (Lloyd's algorithm, numpy-only)
        k = min(self.k, len(actions))
        indices = self._rng.choice(len(actions), k, replace=False)
        centroids = actions[indices].copy()

        for _ in range(n_iters):
            # Assign each action to nearest centroid
            diffs = actions[:, None, :] - centroids[None, :, :]   # (N, k, D)
            dists = (diffs ** 2).sum(axis=2)                       # (N, k)
            assignments = dists.argmin(axis=1)                     # (N,)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = assignments == c
                if mask.sum() > 0:
                    new_centroids[c] = actions[mask].mean(axis=0)
                else:
                    # Empty cluster — re-seed from a random data point
                    new_centroids[c] = actions[self._rng.randint(len(actions))]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self._table = centroids
        self._is_fitted = True

    def decode(self, action_idx: int) -> np.ndarray:
        """
        Convert a discrete action index to a continuous torque vector.

        Args:
            action_idx: Integer action index.

        Returns:
            Continuous torque vector (action_dim,) float32.
        """
        idx = int(action_idx) % len(self._table)
        return self._table[idx].copy()

    def encode(self, continuous_action: np.ndarray) -> int:
        """
        Find the nearest discrete action for a continuous torque vector.

        Args:
            continuous_action: (action_dim,) float32 array.

        Returns:
            Nearest discrete action index.
        """
        a = np.asarray(continuous_action, dtype=np.float32).flatten()
        # Pad or truncate to match action_dim
        if len(a) < self.action_dim:
            a = np.pad(a, (0, self.action_dim - len(a)))
        elif len(a) > self.action_dim:
            a = a[:self.action_dim]

        diffs = self._table - a[None, :]          # (n_actions, action_dim)
        dists = (diffs ** 2).sum(axis=1)          # (n_actions,)
        return int(np.argmin(dists))

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(self._table)

    @property
    def action_table(self) -> np.ndarray:
        """Full action table (n_actions, action_dim)."""
        return self._table.copy()

    def describe(self) -> Dict:
        """Human-readable description of the discretization."""
        return {
            "strategy": self.strategy,
            "n_joints": self.n_joints,
            "action_dim": self.action_dim,
            "n_actions": self.n_actions,
            "scale": self.scale,
            "is_fitted": self._is_fitted,
            "action_magnitudes": {
                "mean": round(float(np.abs(self._table).mean()), 4),
                "max": round(float(np.abs(self._table).max()), 4),
            }
        }

    def __repr__(self) -> str:
        return (
            f"MuJoCoActionDiscretizer(strategy={self.strategy!r}, "
            f"n_joints={self.n_joints}, n_actions={self.n_actions})"
        )
