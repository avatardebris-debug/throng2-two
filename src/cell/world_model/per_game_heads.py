"""Per-game ModuleList head routing for multi-game world models."""
from __future__ import annotations

import torch
import torch.nn as nn


def scatter_per_game_heads(
    encoded: torch.Tensor,
    game_ids: torch.Tensor,
    heads: nn.ModuleList,
    trailing_shape: tuple,
    *,
    device: torch.device,
    n_games: int,
) -> torch.Tensor:
    """Route each batch row through heads[game_id] (skips unknown game ids)."""
    batch = encoded.shape[0]
    out = torch.zeros(batch, *trailing_shape, device=device, dtype=encoded.dtype)
    for gid in game_ids.unique():
        gid_int = int(gid.item())
        if gid_int >= n_games:
            continue
        mask = game_ids == gid
        out[mask] = heads[gid_int](encoded[mask])
    return out
