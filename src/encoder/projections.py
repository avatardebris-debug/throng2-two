"""Numpy projection layers shared by universal and registry encoders."""
from __future__ import annotations

from typing import List

import numpy as np


class NumpyLinear:
    """
    Simple numpy linear projection: in_dim → out_dim.

    Default: Xavier random weights (fast, no data needed).
    Optional: fit_pca(observations) replaces weights with PCA components,
    giving a meaningful low-dim projection after seeing real obs.

    PCA advantages over random:
      - Maximises variance explained in out_dim dims
      - Similar observations project to similar z-vectors (clustering)
      - Faster convergence for downstream world model
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._is_pca_fitted = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Project input to output space."""
        return x @ self.W + self.b

    def fit_pca(
        self,
        observations: np.ndarray,
        center: bool = True,
    ) -> "NumpyLinear":
        """
        Fit PCA on a set of observations and replace W with principal components.

        After calling this, __call__ produces the PCA projection.  The output
        is not L2-normalised here (UniversalEncoder handles that).

        Args:
            observations: (N, in_dim) float32 array of raw preprocessed obs.
            center: If True, subtract mean before computing components.
                    The mean is stored in self.b as the bias offset.

        Returns:
            self, for chaining.
        """
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected (N, {self.in_dim}) obs array, got {obs.shape}"
            )
        if len(obs) < self.out_dim:
            import warnings
            warnings.warn(
                f"fit_pca: only {len(obs)} samples for {self.out_dim} components. "
                "Fewer components than samples — results may be poor."
            )

        if center:
            mean = obs.mean(axis=0)   # (in_dim,)
            obs = obs - mean
            self.b = -mean.astype(np.float32) @ self.W  # centre in output space
        else:
            mean = np.zeros(obs.shape[1])

        # SVD-based PCA (more numerically stable than eig on correlation mat)
        k = min(self.out_dim, obs.shape[0] - 1, obs.shape[1])
        _, _, Vt = np.linalg.svd(obs, full_matrices=False)
        # Vt rows are principal components; take top-k
        components = Vt[:k].T.astype(np.float32)   # (in_dim, k)

        # Pad with random vectors if we have fewer SVD components than out_dim
        if k < self.out_dim:
            rng = np.random.RandomState(0)
            pad = rng.randn(self.in_dim, self.out_dim - k).astype(np.float32)
            # Orthogonalise padding against existing components
            for i in range(pad.shape[1]):
                v = pad[:, i]
                for c in components.T:
                    v -= np.dot(v, c) * c
                norm = np.linalg.norm(v)
                if norm > 1e-8:
                    pad[:, i] = v / norm
            components = np.concatenate([components, pad], axis=1)

        self.W = components.astype(np.float32)
        self._is_pca_fitted = True
        return self

    def fit_from_observations(
        self,
        obs_list: List[np.ndarray],
    ) -> "NumpyLinear":
        """
        Convenience: fit PCA from a list of raw obs vectors.

        Args:
            obs_list: List of (in_dim,) float32 arrays.

        Returns:
            self
        """
        matrix = np.stack([np.asarray(o, dtype=np.float32).flatten()[:self.in_dim]
                           for o in obs_list], axis=0)
        return self.fit_pca(matrix)

    @property
    def is_pca_fitted(self) -> bool:
        return self._is_pca_fitted


# ═══════════════════════════════════════════════════════════════
# CONTRASTIVE PROJECTION (SimCLR-style)
# ═══════════════════════════════════════════════════════════════

class ContrastiveProjection:
    """
    2-layer MLP encoder trained with NT-Xent (SimCLR) contrastive loss.

    Architecture:
        Backbone:      in_dim → hidden_dim (ReLU) → z_dim   ← used at inference
        Proj head:     z_dim  → proj_dim   (ReLU) → proj_dim ← only for training

    The projection head is discarded after training — __call__ returns the
    backbone output only.  This is the standard SimCLR pattern: the projection
    head absorbs augmentation-invariant details that would otherwise corrupt the
    z-space used by the downstream world model.

    Same __call__ interface as NumpyLinear — drop-in replacement.

    Augmentations (applied on-the-fly, no extra data needed):
        1. Gaussian noise:   x + ε,  ε ~ N(0, aug_noise)
        2. Feature dropout:  randomly zero aug_dropout fraction of dims
        3. Scale jitter:     x * Uniform(1-aug_scale, 1+aug_scale) per sample

    NT-Xent loss (for a batch of N obs, 2N augmented vectors):
        sim(u,v) = dot(u,v) (both unit-normed)
        loss_i = -log( exp(sim(zi,zi')/τ) / Σⱼ≠ᵢ exp(sim(zi,zj)/τ) )
        L = mean over all 2N anchors
    """

    def __init__(
        self,
        in_dim: int,
        z_dim: int,
        hidden_dim: int = 64,
        proj_dim: int = 16,
        seed: int = 0,
    ):
        rng = np.random.RandomState(seed)
        self.in_dim  = in_dim
        self.z_dim   = z_dim
        self.out_dim = z_dim        # alias — matches NumpyLinear interface
        self._is_contrastive_fitted = False
        self._is_pca_fitted = False  # compat

        # ── Backbone: in → hidden → z ──────────────────────────
        lim1 = np.sqrt(6.0 / (in_dim + hidden_dim))
        self.W1 = rng.uniform(-lim1, lim1, (in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        lim2 = np.sqrt(6.0 / (hidden_dim + z_dim))
        self.W2 = rng.uniform(-lim2, lim2, (hidden_dim, z_dim)).astype(np.float32)
        self.b2 = np.zeros(z_dim, dtype=np.float32)

        # ── Projection head: z → proj_dim (for training only) ──
        lim3 = np.sqrt(6.0 / (z_dim + proj_dim))
        self.Wp1 = rng.uniform(-lim3, lim3, (z_dim, proj_dim)).astype(np.float32)
        self.bp1 = np.zeros(proj_dim, dtype=np.float32)

        # Adam moments for all weight matrices
        self._adam = {
            k: {"m": np.zeros_like(v), "v": np.zeros_like(v)}
            for k, v in [
                ("W1", self.W1), ("b1", self.b1),
                ("W2", self.W2), ("b2", self.b2),
                ("Wp1", self.Wp1), ("bp1", self.bp1),
            ]
        }
        self._adam_t = 0  # step count for bias correction

    # ── INFERENCE ──────────────────────────────────────────────

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward through backbone only → z (NOT unit-normed here; encoder normalises)."""
        h = np.maximum(0.0, x @ self.W1 + self.b1)   # ReLU
        return h @ self.W2 + self.b2

    def _backbone_batch(self, X: np.ndarray):
        """Batched backbone: (N, in_dim) → z: (N, z_dim), H: (N, hidden_dim)."""
        H = np.maximum(0.0, X @ self.W1 + self.b1)
        Z = H @ self.W2 + self.b2
        return Z, H

    def _proj_head_batch(self, Z: np.ndarray):
        """Project z → proj space + L2-normalise: (N, z_dim) → (N, proj_dim)."""
        P = np.maximum(0.0, Z @ self.Wp1 + self.bp1)
        norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
        return P / norms, P  # (normed, pre-norm)

    # ── AUGMENTATIONS ──────────────────────────────────────────

    def _augment(
        self,
        X: np.ndarray,
        rng: np.random.RandomState,
        aug_noise: float,
        aug_dropout: float,
        aug_scale: float,
    ) -> np.ndarray:
        """Apply one random augmentation per sample independently."""
        out = X.copy()
        N = len(out)
        # Gaussian noise
        if aug_noise > 0:
            out += rng.randn(*out.shape).astype(np.float32) * aug_noise
        # Feature dropout
        if aug_dropout > 0:
            mask = (rng.rand(*out.shape) > aug_dropout).astype(np.float32)
            out *= mask
        # Scale jitter (per-sample scalar)
        if aug_scale > 0:
            scales = rng.uniform(1 - aug_scale, 1 + aug_scale, (N, 1)).astype(np.float32)
            out *= scales
        return out

    # ── TRAINING ───────────────────────────────────────────────

    def _adam_step(self, key: str, param: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
        """Single Adam update; returns updated param."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._adam[key]["m"] = beta1 * self._adam[key]["m"] + (1 - beta1) * grad
        self._adam[key]["v"] = beta2 * self._adam[key]["v"] + (1 - beta2) * grad ** 2
        m_hat = self._adam[key]["m"] / (1 - beta1 ** self._adam_t)
        v_hat = self._adam[key]["v"] / (1 - beta2 ** self._adam_t)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        observations: np.ndarray,
        n_epochs: int = 30,
        lr: float = 3e-3,
        temperature: float = 0.1,
        batch_size: int = 64,
        aug_noise: float = 0.05,
        aug_dropout: float = 0.2,
        aug_scale: float = 0.1,
        seed: int = 42,
        verbose: bool = False,
    ) -> List[float]:
        """
        Train with NT-Xent contrastive loss.

        Args:
            observations: (N, in_dim) float32 array of preprocessed obs.
            n_epochs: Training epochs.
            lr: Adam learning rate.
            temperature: NT-Xent softmax temperature τ (lower = sharper clusters).
            batch_size: Pairs per mini-batch (2*batch_size vectors total).
            aug_noise: Gaussian noise σ for augmentation.
            aug_dropout: Feature dropout rate [0-1].
            aug_scale: Scale jitter range ±scale.
            seed: RNG seed for augmentation.
            verbose: Print per-epoch loss.

        Returns:
            List of per-epoch mean NT-Xent losses.
        """
        import warnings
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.in_dim:
            raise ValueError(f"Expected (N, {self.in_dim}) obs, got {obs.shape}")
        N = len(obs)
        if N < batch_size:
            warnings.warn(
                f"ContrastiveProjection.fit: only {N} samples < batch_size={batch_size}. "
                "Reducing batch_size to N."
            )
            batch_size = N

        rng = np.random.RandomState(seed)
        epoch_losses: List[float] = []

        for epoch in range(n_epochs):
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start: start + batch_size]
                if len(idx) < 2:
                    continue
                xb = obs[idx]                                         # (B, in_dim)
                B = len(xb)

                # Two augmented views
                xa = self._augment(xb, rng, aug_noise, aug_dropout, aug_scale)
                xb2 = self._augment(xb, rng, aug_noise, aug_dropout, aug_scale)

                # Stack into 2B for joint forward pass
                x2 = np.concatenate([xa, xb2], axis=0)               # (2B, in_dim)

                # Forward: backbone → proj head
                self._adam_t += 1
                Z2, H2 = self._backbone_batch(x2)                     # (2B, z_dim), (2B, hidden)
                P2, P2_prenorm = self._proj_head_batch(Z2)            # (2B, proj_dim)

                # NT-Xent loss ─────────────────────────────────────
                # Similarity matrix (all vs all, P2 already unit-normed)
                Sim = P2 @ P2.T / temperature                         # (2B, 2B)

                # Mask out self-similarities
                diag_mask = np.eye(2 * B, dtype=bool)
                Sim[diag_mask] = -1e9

                # Positive pairs: (i, i+B) and (i+B, i)
                pos_idx = np.concatenate([
                    np.arange(B, 2 * B),
                    np.arange(0, B),
                ], axis=0)                                             # (2B,)

                # Log-softmax: loss_i = -Sim[i, pos_i] + logsumexp(Sim[i,:])
                Sim_max = Sim.max(axis=1, keepdims=True)
                log_sum_exp = Sim_max.squeeze() + np.log(
                    np.exp(Sim - Sim_max).sum(axis=1) + 1e-9
                )
                pos_sim = Sim[np.arange(2 * B), pos_idx]
                losses = -pos_sim + log_sum_exp
                loss = losses.mean()
                epoch_loss += loss
                n_batches += 1

                # ── Backprop ─────────────────────────────────────
                # dL/dSim[i,j] = (1/2B) * (softmax[i,j] - 1{j==pos_i})
                softmax = np.exp(Sim - Sim_max) / (
                    np.exp(Sim - Sim_max).sum(axis=1, keepdims=True) + 1e-9
                )
                dSim = softmax / (2 * B)
                dSim[np.arange(2 * B), pos_idx] -= 1.0 / (2 * B)

                # dL/dP2 (pre-normalisation): gradient through cosine sim
                # Simplified: dL/dP_normed = (dSim + dSimᵀ) @ P_normed / τ
                dP_normed = (dSim + dSim.T) @ P2 / temperature        # (2B, proj_dim)

                # Gradient through L2 normalisation
                dot = (P2 * dP_normed).sum(axis=1, keepdims=True)
                norms = np.linalg.norm(P2_prenorm, axis=1, keepdims=True) + 1e-8
                dP_pre = (dP_normed - P2 * dot) / norms               # (2B, proj_dim)

                # Proj head: ReLU backward
                relu_mask = (P2_prenorm > 0).astype(np.float32)
                dP_relu = dP_pre * relu_mask                           # (2B, proj_dim)

                dWp1 = Z2.T @ dP_relu                                 # (z_dim, proj_dim)
                dbp1 = dP_relu.sum(axis=0)
                dZ2  = dP_relu @ self.Wp1.T                           # (2B, z_dim)

                # Backbone layer 2: linear, no activation
                dW2 = H2.T @ dZ2                                      # (hidden, z_dim)
                db2 = dZ2.sum(axis=0)
                dH2 = dZ2 @ self.W2.T                                 # (2B, hidden)

                # Backbone layer 1: ReLU backward
                relu1_mask = (H2 > 0).astype(np.float32)
                dH1 = dH2 * relu1_mask
                dW1 = x2.T @ dH1                                      # (in_dim, hidden)
                db1 = dH1.sum(axis=0)

                # Adam updates
                self.Wp1 = self._adam_step("Wp1", self.Wp1, dWp1, lr)
                self.bp1 = self._adam_step("bp1", self.bp1, dbp1, lr)
                self.W2  = self._adam_step("W2",  self.W2,  dW2,  lr)
                self.b2  = self._adam_step("b2",  self.b2,  db2,  lr)
                self.W1  = self._adam_step("W1",  self.W1,  dW1,  lr)
                self.b1  = self._adam_step("b1",  self.b1,  db1,  lr)

            mean_loss = epoch_loss / max(1, n_batches)
            epoch_losses.append(mean_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    ContrastiveProjection epoch {epoch+1}/{n_epochs}: loss={mean_loss:.4f}")

        self._is_contrastive_fitted = True
        return epoch_losses

    def fit_pca(self, observations: np.ndarray, center: bool = True) -> "ContrastiveProjection":
        """
        Bootstrap backbone weights with PCA before contrastive fine-tuning.

        PCA gives a better starting point than random init — the backbone
        axes are already aligned with real observation variance, so contrastive
        training starts from a meaningful geometry rather than random noise.
        """
        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != self.in_dim:
            raise ValueError(f"Expected (N, {self.in_dim}) obs, got {obs.shape}")
        if center:
            mean = obs.mean(axis=0)
            obs = obs - mean
            self.b1 = (-mean @ np.ones((self.in_dim, 64), dtype=np.float32)).astype(np.float32)
        k = min(64, obs.shape[0] - 1, obs.shape[1])   # hidden_dim
        _, _, Vt = np.linalg.svd(obs, full_matrices=False)
        components = Vt[:k].T.astype(np.float32)        # (in_dim, k)
        if k < 64:
            rng = np.random.RandomState(0)
            pad = rng.randn(self.in_dim, 64 - k).astype(np.float32)
            components = np.concatenate([components, pad], axis=1)
        self.W1 = components.astype(np.float32)
        self._is_pca_fitted = True
        return self

    def fit_from_observations(self, obs_list: List[np.ndarray]) -> "ContrastiveProjection":
        """Convenience: fit PCA from a list of raw obs vectors (same API as NumpyLinear)."""
        matrix = np.stack([np.asarray(o, dtype=np.float32).flatten()[:self.in_dim]
                           for o in obs_list], axis=0)
        return self.fit_pca(matrix)

    @property
    def is_contrastive_fitted(self) -> bool:
        return self._is_contrastive_fitted

    @property
    def is_pca_fitted(self) -> bool:
        return self._is_pca_fitted


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL ENCODER
# ═══════════════════════════════════════════════════════════════

