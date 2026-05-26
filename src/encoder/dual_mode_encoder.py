"""
dual_mode_encoder.py — Toggle between fast ASCII encoding and detailed CNN encoding.

Motivation (from architectural discussion):
    ASCII encoding works well for training — it strips noise and focuses on
    structure. But it destroys information needed for precision tasks
    (fine textures, wrinkles, delicate object shapes).

    A dual-mode encoder solves this by having TWO paths that both produce
    the SAME shaped z-vector, so everything downstream (world model, dreamer,
    meta-encoder, reward functions) is completely agnostic to which path ran.

Modes:
    "fast"    — AsciiEncoder + NumpyLinear projection (zero pytorch, ~0.01ms)
                Good for: exploration, bulk training, navigation, coarse tasks
    "detail"  — PixelEncoder CNN (requires pytorch, ~1-5ms CPU)
                Good for: precision tasks, fine motor control, texture-sensitive tasks

Switching:
    encoder.set_mode("detail")   # switch to CNN
    encoder.set_mode("fast")     # switch back to ASCII
    encoder.toggle()             # switch to the other

The mode can be switched MID-EPISODE. The z-vector shape never changes,
so no downstream reset is needed. The world model will see slightly higher
prediction error (surprise) for a few steps while the new features settle —
this is expected and useful as an intrinsic novelty signal.

Integration with existing pipeline:
    # Replace this:
    z = encoder.encode(obs)   # UniversalEncoder

    # With this (fully backward compatible):
    z = dual_enc.encode(obs)  # same interface, same output shape

Usage:
    enc = DualModeEncoder(
        game_name="mario",
        z_dim=32,
        frame_h=84, frame_w=84,
        initial_mode="fast",
    )

    # During coarse exploration:
    z = enc.encode(frame)           # fast (ASCII) path

    # Precision phase (e.g., fine-tuning for robot grasping):
    enc.set_mode("detail")
    z = enc.encode(frame)           # detail (CNN) path — same z_dim!

    # Check what mode is active:
    enc.mode       # "fast" or "detail"
    enc.stats()    # latency, encode count per mode, etc.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional, TYPE_CHECKING

import numpy as np

from src.encoder.universal_encoder import UniversalEncoder

if TYPE_CHECKING:
    from src.cell.world_model import MultiGameWorldModel

_TORCH_AVAILABLE = True
try:
    from src.encoder.pixel_encoder import PixelEncoder
except ImportError:
    _TORCH_AVAILABLE = False
    PixelEncoder = None  # type: ignore


# ═══════════════════════════════════════════════════════════════
# SURPRISE MAP — spatial surprise memory
# ═══════════════════════════════════════════════════════════════

class SurpriseMap:
    """
    Spatial memory that tracks rolling mean world-model surprise per z-cell.

    Maps discretized z-vectors → rolling mean surprise score.  When the
    agent revisits a z-cell, the pre-recorded surprise value lets the
    DualModeEncoder switch to detail mode *before* the world model fires,
    not just after.

    Cell keys are tuples of quantised z-components (same scheme as
    ZCellArchive so the two can share keys).

    Decay: each update blends new surprise with old via α::

        stored[key] = α * stored[key] + (1 - α) * new_surprise

    so stale entries fade gracefully and the map stays responsive to
    recent experience.
    """

    def __init__(
        self,
        resolution: int = 8,
        decay: float = 0.9,
        max_cells: int = 10_000,
    ):
        """
        Args:
            resolution: Quantisation factor for z-components.
                        Higher → finer grid, more cells, slower lookup.
            decay: EMA decay α ∈ (0, 1).  0.9 = slow decay, 0.5 = fast.
            max_cells: Cap on stored cells (LRU eviction when exceeded).
        """
        self.resolution = resolution
        self.decay = decay
        self.max_cells = max_cells
        self._map: dict = {}          # cell_key → float surprise
        self._age: dict = {}          # cell_key → int (update counter)
        self._global_age = 0

    def cell_key(self, z: np.ndarray) -> tuple:
        """Discretise a unit-sphere z-vector to a hashable cell key."""
        return tuple((z * self.resolution).round().astype(np.int16).tolist())

    def update(self, z: np.ndarray, surprise: float) -> None:
        """Record a surprise observation at the current z-cell location."""
        key = self.cell_key(z)
        self._global_age += 1
        if key in self._map:
            self._map[key] = self.decay * self._map[key] + (1 - self.decay) * surprise
        else:
            # Evict oldest cell if over capacity
            if len(self._map) >= self.max_cells:
                oldest = min(self._age, key=self._age.get)
                del self._map[oldest]
                del self._age[oldest]
            self._map[key] = surprise
        self._age[key] = self._global_age

    def lookup(self, z: np.ndarray) -> Optional[float]:
        """Return stored surprise for this z-cell, or None if unseen."""
        return self._map.get(self.cell_key(z))

    def predict_surprise(self, z: np.ndarray, default: float = 0.0) -> float:
        """Return stored surprise (or default for unseen cells)."""
        return self._map.get(self.cell_key(z), default)

    @property
    def n_cells(self) -> int:
        """Number of tracked cells."""
        return len(self._map)

    def stats(self) -> dict:
        """Summary statistics over stored surprise values."""
        if not self._map:
            return {"n_cells": 0, "mean_surprise": 0.0, "max_surprise": 0.0}
        vals = list(self._map.values())
        return {
            "n_cells": len(vals),
            "mean_surprise": round(float(np.mean(vals)), 4),
            "max_surprise": round(float(np.max(vals)), 4),
        }


# ═══════════════════════════════════════════════════════════════
# DUAL-MODE ENCODER
# ═══════════════════════════════════════════════════════════════

class DualModeEncoder:
    """
    Switchable encoder: FAST (ASCII) ↔ DETAIL (CNN), same z_dim output.

    Both paths produce unit-sphere-normalised (z_dim,) float32 vectors.
    Switch modes freely; downstream components never need to be reset.
    """

    MODES = ("fast", "detail")

    def __init__(
        self,
        game_name: str,
        z_dim: int = 32,
        frame_h: int = 84,
        frame_w: int = 84,
        in_channels: int = 3,
        initial_mode: str = "fast",
        include_game_id: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            game_name: Registered game name (see UniversalEncoder).
            z_dim: Output latent dimension for BOTH paths.
            frame_h: Frame height (required for pixel path).
            frame_w: Frame width (required for pixel path).
            in_channels: 3 for RGB, 1 for grayscale.
            initial_mode: Starting mode — "fast" or "detail".
            include_game_id: If True, append game-ID one-hot to z (same as UniversalEncoder).
            seed: RNG seed for fast-path NumpyLinear (reproducible init).
        """
        if initial_mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {initial_mode!r}")

        self.game_name = game_name
        self.z_dim = z_dim
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.in_channels = in_channels
        self._mode = initial_mode
        self.include_game_id = include_game_id

        # ── FAST path (always available) ──────────────────────
        self._fast_enc = UniversalEncoder(
            game_name=game_name,
            z_dim=z_dim,
            normalize_z=True,
            include_game_id=include_game_id,
            seed=seed,
        )

        # ── DETAIL path (requires torch) ──────────────────────
        self._detail_enc: Optional["PixelEncoder"] = None
        if _TORCH_AVAILABLE and PixelEncoder is not None:
            self._detail_enc = PixelEncoder(
                frame_h=frame_h,
                frame_w=frame_w,
                in_channels=in_channels,
                z_dim=z_dim,
                normalize_output=True,
            )
        elif initial_mode == "detail":
            raise ImportError(
                "mode='detail' requires torch. Install with: pip install torch"
            )

        # Output dim: z_dim (+ game_id one-hot if include_game_id, fast path only)
        self.out_dim = self._fast_enc.out_dim

        # Stats
        self._encode_count = {"fast": 0, "detail": 0}
        self._total_latency_ms = {"fast": 0.0, "detail": 0.0}
        self._mode_switches = 0
        self._last_z: Optional[np.ndarray] = None

        # Surprise-triggered mode switching
        self._surprise_buffer: deque = deque(maxlen=5)   # rolling window
        self._surprise_map = SurpriseMap()
        self._in_surprise_mode: bool = False

    # ── MODE CONTROL ───────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        """
        Switch encoding mode.

        Args:
            mode: "fast" or "detail".

        Raises:
            ImportError: if "detail" requested but torch not available.
            ValueError: if mode is not recognised.
        """
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if mode == "detail" and self._detail_enc is None:
            raise ImportError(
                "Detail mode requires torch. Install with: pip install torch"
            )
        if mode != self._mode:
            self._mode_switches += 1
        self._mode = mode

    def toggle(self):
        """Switch to the other mode."""
        other = "detail" if self._mode == "fast" else "fast"
        self.set_mode(other)

    @property
    def detail_available(self) -> bool:
        """True if the detail (CNN) path is available."""
        return self._detail_enc is not None

    # ── ENCODING ───────────────────────────────────────────────

    def encode(
        self,
        obs: np.ndarray,
        aux: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode an observation to a z-vector.

        Args:
            obs: Raw observation (pixel frame or flat vector depending on game config).
            aux: Optional auxiliary features to append (passed to fast path only).
            mode: Override the current mode for this call only.
                  Useful for one-off comparisons without changing state.

        Returns:
            z: (out_dim,) float32, L2-normalised.
        """
        active_mode = mode or self._mode
        t0 = time.perf_counter()

        if active_mode == "fast":
            z = self._fast_enc.encode(obs, aux=aux)
        else:
            # Detail path: use PixelEncoder for the core z, then optionally
            # append game_id one-hot to match fast path out_dim
            if self._detail_enc is None:
                raise RuntimeError("Detail encoder not available (torch missing)")
            z_core = self._detail_enc.encode(obs)           # (z_dim,)

            if self.include_game_id:
                # Append game_id one-hot (same as fast path)
                game_id_vec = self._fast_enc._game_id_vec   # reuse from fast encoder
                z = np.concatenate([z_core, game_id_vec], axis=0)
            else:
                z = z_core

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._encode_count[active_mode] += 1
        self._total_latency_ms[active_mode] += elapsed_ms
        self._last_z = z
        return z

    def encode_both(self, obs: np.ndarray) -> tuple:
        """
        Encode with both paths and return both z-vectors.

        Useful for:
         - Measuring divergence between fast and detail representations
         - Computing surprise signal from mode transition
         - Training a projection that maps detail-z → fast-z (knowledge distillation)

        Returns:
            (z_fast, z_detail) — both (out_dim,) float32
        """
        if self._detail_enc is None:
            raise RuntimeError("Detail encoder not available (torch missing)")
        z_fast = self.encode(obs, mode="fast")
        z_detail = self.encode(obs, mode="detail")
        return z_fast, z_detail

    def mode_divergence(self, obs: np.ndarray) -> float:
        """
        Cosine distance between fast-z and detail-z for a given observation.

        High divergence = the two paths see very different structure in this obs.
        Can be used as a signal for WHEN to switch to detail mode.

        Returns:
            distance ∈ [0, 2], where 0=identical, 2=opposite.
        """
        if self._detail_enc is None:
            return 0.0
        z_f, z_d = self.encode_both(obs)
        a = z_f / (np.linalg.norm(z_f) + 1e-8)
        b = z_d / (np.linalg.norm(z_d) + 1e-8)
        return float(1.0 - np.dot(a, b))

    def auto_mode(self, obs: np.ndarray, divergence_threshold: float = 0.3) -> str:
        """
        Heuristic: switch to "detail" if fast/detail encodings diverge strongly.

        Use this for tasks where the agent should automatically decide when
        it needs fine-grained vision (e.g., precision grasping).

        Returns the mode that was selected.
        """
        if not self.detail_available:
            return "fast"
        div = self.mode_divergence(obs)
        new_mode = "detail" if div > divergence_threshold else "fast"
        if new_mode != self._mode:
            self.set_mode(new_mode)
        return new_mode

    def surprise_auto_mode(
        self,
        world_model,
        game_id: int,
        obs: np.ndarray,
        threshold: float = 0.5,
        hysteresis: float = 0.7,
        window: int = 3,
        use_map: bool = True,
    ) -> str:
        """
        Switch to detail mode when world-model surprise is persistently high.

        This is the game-agnostic alternative to hardcoding "use detail near
        enemies".  High surprise means the world model cannot predict what
        happens next using the current (fast/ASCII) features — switching to
        detail gives it better signal.

        Two-layer trigger:
          1. **SurpriseMap pre-emption**: if the *stored* surprise for this
             z-cell exceeds `threshold`, switch to detail immediately (before
             the world model fires).  This is the "remember" part.
          2. **Rolling-window trigger**: maintains a deque of the last 5
             `world_model.surprise(game_id)` readings.  Switches to detail
             only when the mean of the last `window` readings > `threshold`,
             preventing thrashing on single spikes.
          3. **Hysteresis**: once in detail mode, only switch back to fast
             when mean_recent < `threshold * hysteresis` (default 0.35).

        After encoding, records (z_current, surprise) into the SurpriseMap
        so future visits to this region can pre-empt the rolling window.

        Args:
            world_model: Any object with `.surprise(game_id) -> float`.
            game_id: Current game ID.
            obs: Raw observation (used to encode and get z for the map).
            threshold: Surprise level at which to switch to detail mode.
            hysteresis: Fraction of threshold at which to release detail mode.
                        Smaller = more time in detail mode after surprise drops.
            window: Number of recent readings to average for the trigger.
            use_map: If True, also use SurpriseMap for pre-emptive switching.

        Returns:
            Active mode string: "fast" or "detail".
        """
        if not self.detail_available:
            return "fast"

        # Get current z (fast path, no side effects on encode_count)
        z_fast = self._fast_enc.encode(obs)

        # Layer 1: SurpriseMap pre-emption
        if use_map:
            map_surprise = self._surprise_map.predict_surprise(z_fast, default=0.0)
            if map_surprise > threshold and not self._in_surprise_mode:
                self.set_mode("detail")
                self._in_surprise_mode = True

        # Layer 2: rolling-window trigger
        current_surprise = 0.0
        if hasattr(world_model, "surprise"):
            try:
                current_surprise = float(world_model.surprise(game_id))
            except Exception:
                current_surprise = 0.0

        self._surprise_buffer.append(current_surprise)

        if len(self._surprise_buffer) >= window:
            mean_recent = float(np.mean(list(self._surprise_buffer)[-window:]))
            if not self._in_surprise_mode and mean_recent > threshold:
                self.set_mode("detail")
                self._in_surprise_mode = True
            elif self._in_surprise_mode and mean_recent < threshold * hysteresis:
                self.set_mode("fast")
                self._in_surprise_mode = False

        # Update SurpriseMap with current experience
        self._surprise_map.update(z_fast, current_surprise)

        return self._mode

    @property
    def surprise_map(self) -> "SurpriseMap":
        """The spatial surprise memory. Shared with GoExploreRunner if needed."""
        return self._surprise_map

    # ── STATS ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """Per-mode encoding statistics."""
        result = {
            "current_mode": self._mode,
            "mode_switches": self._mode_switches,
            "in_surprise_mode": self._in_surprise_mode,
            "detail_available": self.detail_available,
            "out_dim": self.out_dim,
        }
        for m in self.MODES:
            n = self._encode_count[m]
            avg_ms = self._total_latency_ms[m] / max(1, n)
            result[f"{m}_encodes"] = n
            result[f"{m}_avg_ms"] = round(avg_ms, 3)

        if self._detail_enc is not None:
            result["detail_params"] = self._detail_enc.stats()["n_params"]

        result["surprise_map"] = self._surprise_map.stats()
        return result

    def __repr__(self) -> str:
        return (
            f"DualModeEncoder(game={self.game_name!r}, z_dim={self.z_dim}, "
            f"mode={self._mode!r}, detail_available={self.detail_available})"
        )
