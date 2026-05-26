"""
universal_encoder.py — Unified cross-game observation encoder.

Produces comparable z-vectors from any environment's observations by
routing through a two-stage pipeline:

  Stage 1 — AsciiEncoder (per-game config):
    Any obs (raw float vector OR RGB frame) → density grid
    - Mario/Atari: flattened MarioSimulator obs (378-dim) used directly
    - CartPole/MuJoCo: structured RAM vector used directly
    - Pixel-based: RGB (H,W,3) → (rows×cols,) density grid

  Stage 2 — Projection to shared z-space:
    density/raw obs → z-vector (z_dim, normalized to unit sphere)
    Implemented as a simple numpy linear layer (no PyTorch dependency)
    so it can be used in the pure-numpy training loop.
    Optionally replaced by a trained StateEncoder (PyTorch) for gradient flow.

  Game ID:
    A small one-hot vector is appended to z: z_full = [z, game_id_onehot]
    so the downstream world model can condition on the game.

Registry:
    GAME_CONFIGS maps game_name → EncoderConfig (obs_type, grid_shape, z_dim)
    New games are added by registering a config, no code changes required.

Usage:
    enc = UniversalEncoder(game_name="mario", z_dim=32)
    z = enc.encode(obs)          # (z_dim + n_games,) float32 np.ndarray

    # Pixel games (CartPole with render):
    enc_cp = UniversalEncoder(game_name="cartpole", z_dim=32)
    z = enc_cp.encode(rgb_frame)  # rgb_frame: (H, W, 3) uint8
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import numpy as np

from .ascii_encoder import AsciiEncoder
from .projections import ContrastiveProjection, NumpyLinear


# ═══════════════════════════════════════════════════════════════
# GAME REGISTRY
# ═══════════════════════════════════════════════════════════════

@dataclass
class EncoderConfig:
    """Configuration for a single game's encoder."""
    game_name: str
    game_id: int                   # Unique integer ID (consistent across runs)
    obs_type: str                  # "flat" | "pixel" | "mario_sim"
    obs_dim: int                   # Dimensionality of raw obs (for flat/mario_sim)
    grid_rows: int = 15            # ASCII grid rows (for pixel obs)
    grid_cols: int = 20            # ASCII grid cols (for pixel obs)
    aux_dim: int = 0               # Dimension of extra features appended to gridl
    description: str = ""


# Built-in game registry — extend by calling register_game()
_GAME_REGISTRY: Dict[str, EncoderConfig] = {}

def _register_defaults():
    """Populate the registry with standard game configs."""
    games = [
        EncoderConfig(
            game_name="mario",
            game_id=0,
            obs_type="mario_sim",      # MarioSimulator 378-dim obs
            obs_dim=378,
            description="Mario W1-1 ASCII simulator",
        ),
        EncoderConfig(
            game_name="cartpole",
            game_id=1,
            obs_type="flat",           # 4-dim RAM observation
            obs_dim=4,
            description="CartPole-v1 (4 RAM features)",
        ),
        EncoderConfig(
            game_name="mountaincar",
            game_id=2,
            obs_type="flat",           # 2-dim RAM observation
            obs_dim=2,
            description="MountainCar-v0 (2 RAM features)",
        ),
        EncoderConfig(
            game_name="lunarlander",
            game_id=3,
            obs_type="flat",           # 8-dim RAM observation
            obs_dim=8,
            description="LunarLander-v2 (8 RAM features)",
        ),
        EncoderConfig(
            game_name="montezuma",
            game_id=4,
            obs_type="pixel",
            obs_dim=300,                # 20×15 grid
            grid_rows=20,
            grid_cols=15,
            description="Montezuma's Revenge (pixel → ASCII)",
        ),
        EncoderConfig(
            game_name="gridworld",
            game_id=5,
            obs_type="flat",
            obs_dim=16,
            description="4×4 GridWorld (flat one-hot)",
        ),
        EncoderConfig(
            game_name="pong",
            game_id=6,
            obs_type="flat",
            obs_dim=300,
            description="Atari Pong (flat grid from adapter)",
        ),
        EncoderConfig(
            game_name="breakout",
            game_id=7,
            obs_type="flat",
            obs_dim=300,
            description="Atari Breakout (flat grid from adapter)",
        ),
        EncoderConfig(
            game_name="spaceinvaders",
            game_id=8,
            obs_type="flat",
            obs_dim=300,
            description="Atari Space Invaders (flat grid from adapter)",
        ),
    ]
    for g in games:
        _GAME_REGISTRY[g.game_name] = g

_register_defaults()


def register_game(config: EncoderConfig):
    """Register a new game config (or override an existing one)."""
    _GAME_REGISTRY[config.game_name] = config


def list_games() -> List[str]:
    """Return list of registered game names."""
    return sorted(_GAME_REGISTRY.keys())



def get_n_games() -> int:
    """
    Return the number of registered games (max game_id + 1).

    Always reads from the live registry, so newly registered games
    are reflected immediately. Prefer this over the N_GAMES alias below.
    """
    if not _GAME_REGISTRY:
        return 0
    return max(c.game_id for c in _GAME_REGISTRY.values()) + 1


# Backward-compatible alias.  NOTE: this is computed ONCE at import time.
# Use get_n_games() instead if registering games dynamically at runtime.
N_GAMES = get_n_games()


class UniversalEncoder:
    """
    Unified cross-game observation encoder.

    Produces a fixed-size z-vector plus game-ID one-hot for any registered game.

    Pipeline:
        obs (any type)
          │
          ▼ _preprocess()  → raw feature vector (float32)
          │
          ▼ _project()     → z-vector (z_dim,)
          │
          ▼ normalize()    → unit-sphere z
          │
          ▼ append game_id → z_full (z_dim + N_GAMES,)

    The z-vector is suitable for:
      - CellWorldModel.store_transition() as state/next_state
      - CellDreamer.dream() as features
      - Cross-game replay buffer (all entries same shape)
    """

    def __init__(
        self,
        game_name: str,
        z_dim: int = 32,
        normalize_z: bool = True,
        include_game_id: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            game_name: One of the registered game names (see list_games()).
            z_dim: Dimension of the z-vector before game_id appended.
            normalize_z: If True, normalize z to unit sphere before game_id.
            include_game_id: If True, append N_GAMES-dim one-hot to z.
            seed: RNG seed for projection weights (same seed = consistent init).
        """
        if game_name not in _GAME_REGISTRY:
            raise ValueError(
                f"Unknown game '{game_name}'. Registered: {list_games()}"
            )

        self.config = _GAME_REGISTRY[game_name]
        self.z_dim = z_dim
        self.normalize_z = normalize_z
        self.include_game_id = include_game_id

        # ASCII encoder (only used for pixel obs)
        self._ascii_enc: Optional[AsciiEncoder] = None
        if self.config.obs_type == "pixel":
            self._ascii_enc = AsciiEncoder(
                rows=self.config.grid_rows,
                cols=self.config.grid_cols,
                color_channels=False,
            )

        # Projection: raw obs → z_dim
        raw_dim = self._raw_dim()
        self._project = NumpyLinear(raw_dim, z_dim, seed=seed)

        # Game ID one-hot — sized to fit the actual game_id, not just N_GAMES
        # This allows custom games with any ID to work correctly
        vec_size = max(N_GAMES, self.config.game_id + 1)
        self._game_id_vec = np.zeros(vec_size, dtype=np.float32)
        self._game_id_vec[self.config.game_id] = 1.0

        # Output dimensionality
        self.out_dim = z_dim + (len(self._game_id_vec) if include_game_id else 0)

        # Stats
        self._encode_count = 0
        self._z_sum = np.zeros(z_dim, dtype=np.float64)
        self._z_sq_sum = np.zeros(z_dim, dtype=np.float64)

    def save_projection(self, path: str):
        """Save projection weights (W and b) to a .npz file."""
        np.savez(path, W=self._project.W, b=self._project.b)

    def load_projection(self, path: str):
        """Load projection weights from a .npz file."""
        if not os.path.exists(path):
            return
        data = np.load(path)
        self._project.W = data["W"]
        self._project.b = data["b"]
        self._project._is_pca_fitted = True

    def switch_game(self, new_game_name: str, seed: Optional[int] = None):
        """
        Switch the active game config without resetting the projection weights.
        
        This allows the encoder to be used in a transfer learning gauntlet where
        the latent projection (PCA/Contrastive) remains the same, but the 
        input preprocessing (pixel grids) and game-ID conditioning change.
        
        Args:
            new_game_name: Registered game name to switch to.
            seed: Optional new RNG seed for game-id vector consistency.
        """
        if new_game_name not in _GAME_REGISTRY:
            raise ValueError(f"Unknown game '{new_game_name}'")
            
        self.config = _GAME_REGISTRY[new_game_name]
        
        # Update ASCII encoder if needed
        if self.config.obs_type == "pixel":
            # Only recreate if dimensions changed
            if (self._ascii_enc is None or 
                self._ascii_enc.rows != self.config.grid_rows or 
                self._ascii_enc.cols != self.config.grid_cols):
                self._ascii_enc = AsciiEncoder(
                    rows=self.config.grid_rows,
                    cols=self.config.grid_cols,
                    color_channels=False,
                )
        else:
            self._ascii_enc = None
            
        # Verify projection dimensions (must be compatible)
        raw_dim = self._raw_dim()
        if raw_dim != self._project.in_dim:
            import warnings
            warnings.warn(
                f"Switching from {self._project.in_dim}-dim to {raw_dim}-dim obs. "
                "The current projection weights will be reset to accommodate new dimensions."
            )
            self._project = NumpyLinear(raw_dim, self.z_dim, seed=(seed or 42))

        # Update Game ID one-hot
        vec_size = max(get_n_games(), self.config.game_id + 1)
        self._game_id_vec = np.zeros(vec_size, dtype=np.float32)
        self._game_id_vec[self.config.game_id] = 1.0

        # Update output dimensionality
        self.out_dim = self.z_dim + (len(self._game_id_vec) if self.include_game_id else 0)

    def _raw_dim(self) -> int:
        """Raw feature dimension after preprocessing."""
        cfg = self.config
        if cfg.obs_type == "pixel":
            return cfg.grid_rows * cfg.grid_cols + cfg.aux_dim
        else:
            return cfg.obs_dim + cfg.aux_dim

    def _preprocess(
        self,
        obs: np.ndarray,
        aux: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert raw obs to a flat float32 feature vector.

        Args:
            obs: Raw observation (flat array or RGB frame).
            aux: Optional auxiliary features to append (e.g., proprioception).

        Returns:
            flat float32 feature vector ready for projection.
        """
        cfg = self.config

        if cfg.obs_type == "pixel":
            # RGB frame → ASCII density grid
            if obs.ndim != 3 or obs.shape[2] != 3:
                raise ValueError(
                    f"Pixel obs must be (H, W, 3), got {obs.shape}"
                )
            grid = self._ascii_enc.encode(obs)           # (rows, cols) int
            raw = grid.flatten().astype(np.float32) / 9.0  # Normalize to [0,1]

        elif cfg.obs_type in ("flat", "mario_sim"):
            # Already a flat vector — just normalize
            raw = np.asarray(obs, dtype=np.float32).flatten()
            if len(raw) != cfg.obs_dim:
                # Truncate or pad gracefully
                if len(raw) > cfg.obs_dim:
                    raw = raw[:cfg.obs_dim]
                else:
                    raw = np.pad(raw, (0, cfg.obs_dim - len(raw)))
            # Soft normalization: clip to [-10, 10], then divide by 10
            raw = np.clip(raw, -10.0, 10.0) / 10.0

        else:
            raise ValueError(f"Unknown obs_type: {cfg.obs_type!r}")

        # Append auxiliary features if provided
        if aux is not None and len(aux) > 0:
            aux_arr = np.asarray(aux, dtype=np.float32).flatten()
            # Pad/truncate aux to expected dim
            if cfg.aux_dim > 0:
                if len(aux_arr) > cfg.aux_dim:
                    aux_arr = aux_arr[:cfg.aux_dim]
                elif len(aux_arr) < cfg.aux_dim:
                    aux_arr = np.pad(aux_arr, (0, cfg.aux_dim - len(aux_arr)))
            raw = np.concatenate([raw, aux_arr])

        return raw

    def encode(
        self,
        obs: np.ndarray,
        aux: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode an observation to a z-vector.

        Args:
            obs: Raw observation from the environment.
            aux: Optional auxiliary features (proprioception, RAM stats, etc.)

        Returns:
            z_full: (out_dim,) float32 array = [z_normalized, game_id_onehot]
        """
        raw = self._preprocess(obs, aux)

        # Project to z-space
        z = self._project(raw)

        # Unit sphere normalization
        if self.normalize_z:
            norm = np.linalg.norm(z)
            if norm > 1e-8:
                z = z / norm

        # Track running stats
        self._encode_count += 1
        self._z_sum += z.astype(np.float64)
        self._z_sq_sum += (z ** 2).astype(np.float64)

        # Append game ID one-hot
        if self.include_game_id:
            return np.concatenate([z, self._game_id_vec], axis=0)
        return z

    @property
    def game_id(self) -> int:
        """Integer game ID."""
        return self.config.game_id

    @property
    def game_id_vec(self) -> np.ndarray:
        """One-hot game ID vector (N_GAMES,)."""
        return self._game_id_vec.copy()

    @property
    def is_pca_fitted(self) -> bool:
        """True if the projection has been fitted via fit_projection()."""
        return self._project.is_pca_fitted

    def fit_projection(
        self,
        obs_list: List[np.ndarray],
        aux_list: Optional[List[Optional[np.ndarray]]] = None,
    ) -> "UniversalEncoder":
        """
        Fit PCA projection on real observations from this game.

        Replaces the random Xavier weights with principal components computed
        from the actual observation distribution, giving the world model a
        meaningful low-dimensional space aligned with real state variance.

        Call this once after collecting a warm-up batch of observations
        (e.g. 200-500 random-policy obs) before joint training begins.

        Args:
            obs_list: List of raw environment observations (pre-encode).
            aux_list: Optional list of aux arrays; must match obs_list length.

        Returns:
            self, for chaining.
        """
        import warnings
        if len(obs_list) < self.z_dim:
            warnings.warn(
                f"fit_projection({self.config.game_name!r}): only {len(obs_list)} "
                f"samples for z_dim={self.z_dim}. PCA needs >= z_dim samples for "
                "full components. Proceeding with fewer components + random padding."
            )

        # Preprocess each obs through the same pipeline as encode() so PCA
        # sees the actual distribution the projection operates on.
        raw_list = []
        for i, obs in enumerate(obs_list):
            aux = aux_list[i] if aux_list is not None else None
            raw_list.append(self._preprocess(obs, aux))

        self._project.fit_from_observations(raw_list)
        return self

    def fit_contrastive(
        self,
        obs_list: List[np.ndarray],
        aux_list: Optional[List[Optional[np.ndarray]]] = None,
        n_epochs: int = 30,
        lr: float = 3e-3,
        temperature: float = 0.1,
        batch_size: int = 64,
        pca_first: bool = True,
        verbose: bool = False,
    ) -> "UniversalEncoder":
        """
        Fit contrastive projection on real observations from this game.

        Two phases:
          1. PCA warm-start (if pca_first=True): aligns backbone axes to real variance.
          2. NT-Xent contrastive training: makes same-state augmentations cluster together.

        Requires the underlying projection to be a ContrastiveProjection.
        Falls back to PCA-only (fit_projection) if projection is NumpyLinear.

        Args:
            obs_list: List of raw environment observations.
            aux_list: Optional auxiliary arrays.
            n_epochs: Contrastive training epochs.
            lr: Adam learning rate.
            temperature: NT-Xent temperature τ.
            batch_size: Mini-batch size.
            pca_first: Bootstrap backbone with PCA before contrastive training.
            verbose: Print training progress.

        Returns:
            self, for chaining.
        """
        # Preprocess obs through the same pipeline as encode()
        raw_list = []
        for i, obs in enumerate(obs_list):
            aux = aux_list[i] if aux_list is not None else None
            raw_list.append(self._preprocess(obs, aux))
        raw_matrix = np.stack(raw_list, axis=0)  # (N, raw_dim)

        proj = self._project

        # Check if projection supports contrastive training
        if not hasattr(proj, "fit"):
            # NumpyLinear fallback: just do PCA
            if verbose:
                print(f"  fit_contrastive({self.config.game_name!r}): "
                      "projection is NumpyLinear, falling back to PCA.")
            proj.fit_from_observations(raw_list)
            return self

        # Phase 1: PCA warm-start
        if pca_first and len(raw_list) >= proj.in_dim:
            proj.fit_pca(raw_matrix)
            if verbose:
                print(f"  fit_contrastive({self.config.game_name!r}): PCA done")

        # Phase 2: Contrastive fine-tuning
        losses = proj.fit(
            raw_matrix,
            n_epochs=n_epochs,
            lr=lr,
            temperature=temperature,
            batch_size=batch_size,
            verbose=verbose,
        )
        if verbose:
            print(f"  fit_contrastive({self.config.game_name!r}): "
                  f"final loss={losses[-1]:.4f}")
        return self

    @property
    def is_contrastive_fitted(self) -> bool:
        """True if the projection has been contrastively trained."""
        return getattr(self._project, "is_contrastive_fitted", False)

    def stats(self) -> Dict[str, Any]:
        """Encoding statistics."""
        n = max(1, self._encode_count)
        z_mean = self._z_sum / n
        z_var = self._z_sq_sum / n - z_mean ** 2
        return {
            "game": self.config.game_name,
            "game_id": self.config.game_id,
            "obs_type": self.config.obs_type,
            "z_dim": self.z_dim,
            "out_dim": self.out_dim,
            "encode_count": self._encode_count,
            "z_mean_norm": round(float(np.linalg.norm(z_mean)), 4),
            "z_std_mean": round(float(np.sqrt(z_var.mean())), 4),
        }

    def __repr__(self) -> str:
        return (
            f"UniversalEncoder(game={self.config.game_name!r}, "
            f"z_dim={self.z_dim}, out_dim={self.out_dim})"
        )


# ═══════════════════════════════════════════════════════════════
# ENCODER REGISTRY — manage one encoder per game
# ═══════════════════════════════════════════════════════════════

class EncoderRegistry:
    """
    Manages one UniversalEncoder per game, sharing the same z_dim
    so transitions from all games are directly comparable.

    Usage:
        registry = EncoderRegistry(z_dim=32, games=["mario","cartpole"])
        z_mario = registry.encode("mario", mario_obs)
        z_cp    = registry.encode("cartpole", cartpole_obs)
        # z_mario.shape == z_cp.shape == (32 + N_GAMES,)
    """

    def __init__(
        self,
        z_dim: int = 32,
        games: Optional[List[str]] = None,
        normalize_z: bool = True,
        seed: int = 42,
    ):
        if games is None:
            games = ["mario", "cartpole", "mountaincar"]

        self.z_dim = z_dim
        self.games = games
        self._encoders: Dict[str, UniversalEncoder] = {}

        for g in games:
            self._encoders[g] = UniversalEncoder(
                game_name=g,
                z_dim=z_dim,
                normalize_z=normalize_z,
                include_game_id=True,
                seed=seed,
            )

        self.out_dim = self._encoders[games[0]].out_dim

    def encode(
        self,
        game_name: str,
        obs: np.ndarray,
        aux: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode an observation from the named game."""
        if game_name not in self._encoders:
            raise KeyError(f"Game '{game_name}' not in registry. Registered: {self.games}")
        return self._encoders[game_name].encode(obs, aux)

    def game_id(self, game_name: str) -> int:
        """Integer game ID for the named game."""
        return self._encoders[game_name].game_id

    def fit_all(
        self,
        obs_by_game: Dict[str, List[np.ndarray]],
        aux_by_game: Optional[Dict[str, List[Optional[np.ndarray]]]] = None,
    ) -> "EncoderRegistry":
        """
        Fit PCA projections for all registered games from collected observations.

        Call this once after a warm-up data collection phase. Each game's
        encoder is fitted independently on its own obs distribution.

        Args:
            obs_by_game: {game_name: [obs, ...]} raw env observations.
            aux_by_game: Optional {game_name: [aux, ...]} auxiliary arrays.

        Returns:
            self, for chaining.
        """
        for game, obs_list in obs_by_game.items():
            if game not in self._encoders:
                continue
            aux_list = aux_by_game.get(game) if aux_by_game else None
            self._encoders[game].fit_projection(obs_list, aux_list)
        return self

    def fit_contrastive_all(
        self,
        obs_by_game: Dict[str, List[np.ndarray]],
        aux_by_game: Optional[Dict[str, List[Optional[np.ndarray]]]] = None,
        n_epochs: int = 30,
        lr: float = 3e-3,
        temperature: float = 0.1,
        batch_size: int = 64,
        pca_first: bool = True,
        verbose: bool = False,
    ) -> "EncoderRegistry":
        """
        Run contrastive pre-training for all registered games.

        Two-phase per game: (1) PCA warm-start, (2) NT-Xent fine-tuning.
        Call once after warm-up data collection, before main joint training.
        """
        for game, obs_list in obs_by_game.items():
            if game not in self._encoders:
                continue
            aux_list = aux_by_game.get(game) if aux_by_game else None
            self._encoders[game].fit_contrastive(
                obs_list, aux_list,
                n_epochs=n_epochs, lr=lr, temperature=temperature,
                batch_size=batch_size, pca_first=pca_first, verbose=verbose,
            )
        return self

    @property
    def is_contrastive_fitted(self) -> Dict[str, bool]:
        """Per-game dict showing whether contrastive training has been run."""
        return {g: enc.is_contrastive_fitted for g, enc in self._encoders.items()}

    def stats(self) -> Dict[str, Any]:
        """Stats for all registered encoders."""
        return {g: enc.stats() for g, enc in self._encoders.items()}

    def __repr__(self) -> str:
        return f"EncoderRegistry(games={self.games}, z_dim={self.z_dim}, out_dim={self.out_dim})"
