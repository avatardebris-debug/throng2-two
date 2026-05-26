"""
pixel_encoder.py — Small CNN encoder: raw pixel frames → z_dim feature vector.

This is the DETAIL path in the dual-mode encoder. It replaces AsciiEncoder
when the task requires fine-grained visual information (textures, wrinkles,
precise object boundaries) that ASCII grid compression would destroy.

Architecture (deliberately small — CPU/laptop-friendly):
    Input: (H, W, 3) uint8  OR  (H, W) grayscale uint8
    Conv1: 3 → 32, kernel 8, stride 4   → drastically downsample
    Conv2: 32 → 64, kernel 4, stride 2
    Conv3: 64 → 64, kernel 3, stride 1
    Global adaptive avg pool → (64, 1, 1)
    Flatten → (64,)
    Linear → z_dim
    LayerNorm + Tanh → bounded z for SNN/world-model compatibility

Memory footprint at 84×84 input: ~0.5MB params. Inference: ~0.3ms CPU.

Comparison to ASCII path:
    AsciiEncoder  — 15×20 × char categories → 300 floats
                    Zero pytorch dependency, 10× faster, no texture detail
    PixelEncoder  — full frame → z_dim floats
                    Requires pytorch, slower, captures fine visual detail

Both produce the same z_dim: downstream modules (world model, dreamer,
meta-encoder) are fully agnostic to which path produced z.

Usage:
    enc = PixelEncoder(frame_h=84, frame_w=84, z_dim=32)
    z = enc.encode(frame)   # (H, W, 3) uint8 → (32,) float32

For grayscale:
    enc = PixelEncoder(frame_h=210, frame_w=160, in_channels=1, z_dim=32)
"""
from __future__ import annotations


import numpy as np

_TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    _TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# PIXEL ENCODER (CNN)
# ═══════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class PixelEncoder(nn.Module):
        """
        Small convolutional encoder: pixel frame → z_dim latent vector.

        Designed for the DETAIL path where AsciiEncoder loses too much
        information (fine textures, wrinkles, precise object shapes).

        Output is L2-normalised (unit sphere), matching AsciiEncoder→NumpyLinear
        output convention and SNN input range.
        """

        def __init__(
            self,
            frame_h: int = 84,
            frame_w: int = 84,
            in_channels: int = 3,
            z_dim: int = 32,
            normalize_output: bool = True,
        ):
            """
            Args:
                frame_h: Input frame height in pixels.
                frame_w: Input frame width in pixels.
                in_channels: 3 for RGB, 1 for grayscale.
                z_dim: Output latent dimension (must match z_dim used elsewhere).
                normalize_output: If True, L2-normalise z to unit sphere.
            """
            super().__init__()
            self.frame_h = frame_h
            self.frame_w = frame_w
            self.in_channels = in_channels
            self.z_dim = z_dim
            self.normalize_output = normalize_output

            # Three convolutional layers — standard DQN-style spatial reduction
            # padding=1 on all layers keeps spatial dims from collapsing on small inputs
            self.convs = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )

            # Global average pool to (B, 64, 1, 1) then flatten → 64-dim vector
            # This makes the network input-size-agnostic.
            self.pool = nn.AdaptiveAvgPool2d(1)

            # MLP head: 64 (after global avg pool) → z_dim
            self.head = nn.Sequential(
                nn.Linear(64, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, z_dim),
                nn.LayerNorm(z_dim),
                nn.Tanh(),          # bounded output, SNN-friendly
            )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

            # Stats
            self._encode_count = 0

        def _conv_output_size(self, h: int, w: int, c: int) -> int:
            """Not used (global avg pool makes size adaptive), kept for reference."""
            return 64

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Forward pass.

            Args:
                x: (B, C, H, W) float32 tensor in [0, 1].

            Returns:
                z: (B, z_dim) float32 latent vectors.
            """
            # Conv → global avg pool → (B, 64) → MLP → z
            conv_out = self.convs(x)                 # (B, 64, H', W')
            pooled = self.pool(conv_out).squeeze(-1).squeeze(-1)  # (B, 64)
            features = self.head(pooled)             # (B, z_dim)
            if self.normalize_output:
                norm = features.norm(dim=1, keepdim=True).clamp(min=1e-8)
                features = features / norm
            return features

        def encode(self, frame: np.ndarray) -> np.ndarray:
            """
            Encode a single frame to a z-vector (numpy interface).

            Args:
                frame: (H, W, 3) uint8 RGB, or (H, W) uint8 grayscale,
                       or (H, W, 1) uint8 grayscale.

            Returns:
                z: (z_dim,) float32, L2-normalised.
            """
            self._encode_count += 1
            frame = np.asarray(frame, dtype=np.float32)

            # Normalise to [0, 1]
            if frame.max() > 1.0:
                frame = frame / 255.0

            # Handle shape: ensure (C, H, W)
            if frame.ndim == 2:
                # Grayscale (H, W) → (1, H, W)
                frame = frame[np.newaxis, :, :]
            elif frame.ndim == 3 and frame.shape[2] in (1, 3):
                # (H, W, C) → (C, H, W)
                frame = frame.transpose(2, 0, 1)

            # Pad or crop to expected (in_channels, frame_h, frame_w)
            c, h, w = frame.shape
            if h != self.frame_h or w != self.frame_w:
                from scipy.ndimage import zoom
                try:
                    frame = zoom(
                        frame,
                        (1, self.frame_h / h, self.frame_w / w),
                        order=1,
                    )
                except ImportError:
                    # No scipy — just crop/pad
                    frame = frame[:, :self.frame_h, :self.frame_w]
                    if frame.shape[1] < self.frame_h:
                        pad_h = self.frame_h - frame.shape[1]
                        frame = np.pad(frame, ((0,0),(0,pad_h),(0,0)))
                    if frame.shape[2] < self.frame_w:
                        pad_w = self.frame_w - frame.shape[2]
                        frame = np.pad(frame, ((0,0),(0,0),(0,pad_w)))

            # Apply channel conversion if mismatch
            if c != self.in_channels:
                if self.in_channels == 1 and c == 3:
                    # Convert RGB → grayscale (luminance)
                    frame = (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2])[np.newaxis]
                elif self.in_channels == 3 and c == 1:
                    # Duplicate grayscale to 3 channels
                    frame = np.repeat(frame, 3, axis=0)

            with torch.no_grad():
                t = torch.as_tensor(frame, dtype=torch.float32).unsqueeze(0).to(self.device)
                z = self.forward(t).squeeze(0).cpu().numpy()
            return z

        def stats(self) -> dict:
            n_params = sum(p.numel() for p in self.parameters())
            return {
                "encode_count": self._encode_count,
                "n_params": n_params,
                "frame_shape": (self.in_channels, self.frame_h, self.frame_w),
                "z_dim": self.z_dim,
                "device": str(self.device),
            }

        def __repr__(self) -> str:
            n = sum(p.numel() for p in self.parameters())
            return (
                f"PixelEncoder(frame={self.frame_h}×{self.frame_w}, "
                f"in_ch={self.in_channels}, z_dim={self.z_dim}, "
                f"params={n/1e3:.1f}k)"
            )

else:
    # Dummy class when torch is not available
    class PixelEncoder:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PixelEncoder requires PyTorch. Install with: pip install torch"
            )
