# -*- coding: utf-8 -*-
"""
checkpoint_manager.py -- Save/load complete training state.

Saves:
  - World model weights  (torch state_dict or numpy fallback)
  - EncoderRegistry projections  (per-game W/b arrays)
  - GoExplore ZCellArchive  (per-game cells)
  - EliteReplayManager  (top-N trajectories per game)
  - Meta JSON  (episode count, game list, scores, timestamp)

Rolling window: keeps the last 3 checkpoints by default; the one
with highest world-model confidence is flagged as "best".
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

def _ckpt_dir(base: str, episode: int) -> str:
    return os.path.join(base, f"ep{episode:06d}")


def _meta_path(directory: str) -> str:
    return os.path.join(directory, "meta.json")


# ---------------------------------------------------------------
# CHECKPOINT MANAGER
# ---------------------------------------------------------------

class CheckpointManager:
    """
    Save and restore complete training state so runs are cumulative.

    Usage pattern:
        ckpt = CheckpointManager("results/checkpoints")

        # --- save ---
        ckpt.save(ep, world_model, enc, go_archives, elite_replay)

        # --- resume ---
        ep0, world_model, enc, go_archives, elite_replay = ckpt.load("results/checkpoints/latest")
    """

    def __init__(self, base_dir: str, keep_last: int = 3):
        self.base_dir  = base_dir
        self.keep_last = keep_last
        os.makedirs(base_dir, exist_ok=True)

    # -- SAVE --------------------------------------------------------

    def save(
        self,
        episode:      int,
        world_model,                   # MultiGameWorldModel or None
        enc,                           # EncoderRegistry
        go_archives:  Dict[str, Any],  # {game: ZCellArchive}
        elite_replay,                  # EliteReplayManager
        extra_meta:   Optional[dict] = None,
    ) -> str:
        """Save a checkpoint and return the directory path."""
        ckpt_dir = _ckpt_dir(self.base_dir, episode)
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. World model
        self._save_world_model(world_model, os.path.join(ckpt_dir, "world_model"))

        # 2. Encoder projections
        self._save_encoders(enc, os.path.join(ckpt_dir, "encoders"))

        # 3. GoExplore archives
        self._save_archives(go_archives, os.path.join(ckpt_dir, "go_archives"))

        # 4. Elite replay
        elite_replay.save(os.path.join(ckpt_dir, "elite_replay"))

        # 5. Meta
        meta = {
            "episode":    episode,
            "games":      enc.games,
            "z_dim":      enc.z_dim,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wm_ready":   bool(world_model and getattr(world_model, "is_ready", False)),
            "wm_confidence": float(getattr(world_model, "confidence", 0.0)) if world_model else 0.0,
            "elite_stats": elite_replay.stats(),
            "archive_sizes": {g: getattr(ar, "size", 0) for g, ar in go_archives.items()},
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(_meta_path(ckpt_dir), "w") as f:
            json.dump(meta, f, indent=2, default=float)

        # Update symlink "latest"
        latest = os.path.join(self.base_dir, "latest")
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.remove(latest)
            except Exception:
                pass
        try:
            # relative symlink so the whole folder is moveable
            os.symlink(os.path.basename(ckpt_dir), latest)
        except Exception:
            # Windows may not allow symlinks without privileges — write a text file
            with open(latest + ".txt", "w") as f:
                f.write(ckpt_dir)

        # Rolling eviction
        self._evict_old(episode)

        return ckpt_dir

    # -- LOAD --------------------------------------------------------

    def load(
        self,
        directory: str,
        world_model,
        enc,
        go_archives: Dict[str, Any],
        elite_replay,
    ) -> int:
        """
        Restore state in-place. Returns the episode number saved.
        Modifies world_model, enc, go_archives, elite_replay in-place.
        """
        if not os.path.isdir(directory):
            # Might be latest.txt fallback
            txt = directory + ".txt"
            if os.path.exists(txt):
                with open(txt) as f:
                    directory = f.read().strip()
            else:
                raise FileNotFoundError(f"Checkpoint directory not found: {directory!r}")

        with open(_meta_path(directory)) as f:
            meta = json.load(f)

        episode = meta["episode"]

        # 1. World model
        self._load_world_model(world_model, os.path.join(directory, "world_model"))

        # 2. Encoders
        self._load_encoders(enc, os.path.join(directory, "encoders"))

        # 3. GoExplore archives
        self._load_archives(go_archives, os.path.join(directory, "go_archives"))

        # 4. Elite replay (reload in-place by copying buffers)
        from src.learning.elite_replay import EliteReplayManager
        loaded_mgr = EliteReplayManager.load(os.path.join(directory, "elite_replay"))
        for game, buf in loaded_mgr._buffers.items():
            elite_replay._buffers[game] = buf

        return episode

    # -- WORLD MODEL -------------------------------------------------

    def _save_world_model(self, wm, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        if wm is None:
            return
        pt_path = os.path.join(directory, "wm.pt")
        try:
            if hasattr(wm, "save"):
                wm.save(pt_path)
                return
            if hasattr(wm, "save_weights"):
                wm.save_weights(pt_path)
                return
            if hasattr(wm, "state_dict_all"):
                import torch
                torch.save(wm.state_dict_all(), pt_path)
        except Exception:
            _log.warning("Could not save world model weights; falling back to stats JSON", exc_info=True)
            try:
                stats = wm.multi_stats() if hasattr(wm, "multi_stats") else wm.stats()
                with open(os.path.join(directory, "wm_stats.json"), "w") as f:
                    json.dump(stats, f, default=float)
            except Exception:
                _log.warning("Could not save world model stats JSON either", exc_info=True)

    def _load_world_model(self, wm, directory: str) -> None:
        if wm is None or not os.path.isdir(directory):
            return
        pt_path = os.path.join(directory, "wm.pt")
        if not os.path.exists(pt_path):
            return
        try:
            if hasattr(wm, "load"):
                wm.load(pt_path)
                return
            if hasattr(wm, "load_weights"):
                wm.load_weights(pt_path)
                return
            if hasattr(wm, "load_state_dict_all"):
                import torch
                wm.load_state_dict_all(
                    torch.load(pt_path, map_location="cpu", weights_only=False)
                )
        except Exception:
            _log.warning("Could not load world model weights from %s", pt_path, exc_info=True)

    # -- ENCODERS ----------------------------------------------------

    def _save_encoders(self, enc, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for game, encoder in enc._encoders.items():
            safe  = game.replace("/", "_")
            proj  = getattr(encoder, "_projection", None)
            if proj is None:
                continue
            np.savez(
                os.path.join(directory, f"{safe}.npz"),
                W            = proj.W,
                b            = proj.b,
                is_pca       = np.array([getattr(proj, "is_pca_fitted", False)]),
                is_contrast  = np.array([getattr(encoder, "_is_contrastive_fitted", False)]),
            )

    def _load_encoders(self, enc, directory: str) -> None:
        if not os.path.isdir(directory):
            return
        for game, encoder in enc._encoders.items():
            safe = game.replace("/", "_")
            path = os.path.join(directory, f"{safe}.npz")
            if not os.path.exists(path):
                continue
            proj = getattr(encoder, "_projection", None)
            if proj is None:
                continue
            try:
                d = np.load(path, allow_pickle=False)
                if d["W"].shape == proj.W.shape:
                    proj.W = d["W"].astype(np.float32)
                    proj.b = d["b"].astype(np.float32)
                    proj._is_pca_fitted = bool(d["is_pca"][0])
                    encoder._is_contrastive_fitted = bool(d["is_contrast"][0])
            except Exception:
                _log.warning("Could not load encoder projection for game %r from %s", game, path, exc_info=True)

    # -- GO-EXPLORE ARCHIVES -----------------------------------------

    def _save_archives(self, go_archives: dict, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for game, archive in go_archives.items():
            safe = game.replace("/", "_")
            try:
                archive.save(os.path.join(directory, f"{safe}.npz"))
            except Exception:
                _log.warning("Could not save GoExplore archive for game %r", game, exc_info=True)

    def _load_archives(self, go_archives: dict, directory: str) -> None:
        if not os.path.isdir(directory):
            return
        for game, archive in go_archives.items():
            safe = game.replace("/", "_")
            path = os.path.join(directory, f"{safe}.npz")
            if not os.path.exists(path):
                continue
            try:
                archive.load(path)
            except Exception:
                _log.warning("Could not load GoExplore archive for game %r from %s", game, path, exc_info=True)

    # -- ROLLING EVICTION --------------------------------------------

    def _evict_old(self, current_ep: int) -> None:
        """Keep only the last keep_last checkpoints."""
        dirs = []
        for name in os.listdir(self.base_dir):
            if name.startswith("ep") and os.path.isdir(
                os.path.join(self.base_dir, name)
            ):
                try:
                    ep = int(name[2:])
                    dirs.append((ep, name))
                except ValueError:
                    pass
        dirs.sort()
        while len(dirs) > self.keep_last:
            _, old_name = dirs.pop(0)
            try:
                shutil.rmtree(os.path.join(self.base_dir, old_name))
            except Exception:
                _log.warning("Could not evict old checkpoint %r", old_name, exc_info=True)

    # -- UTILITIES ---------------------------------------------------

    @staticmethod
    def list_checkpoints(base_dir: str) -> List[dict]:
        """Return metadata for all checkpoints sorted by episode."""
        results = []
        if not os.path.isdir(base_dir):
            return results
        for name in sorted(os.listdir(base_dir)):
            d = os.path.join(base_dir, name)
            mp = _meta_path(d)
            if os.path.isdir(d) and os.path.exists(mp):
                with open(mp) as f:
                    results.append(json.load(f))
        return results

    @staticmethod
    def best_checkpoint(base_dir: str) -> Optional[str]:
        """Return directory of checkpoint with highest WM confidence."""
        ckpts = CheckpointManager.list_checkpoints(base_dir)
        if not ckpts:
            return None
        best = max(ckpts, key=lambda m: m.get("wm_confidence", 0.0))
        return _ckpt_dir(base_dir, best["episode"])

    def __repr__(self) -> str:
        return f"CheckpointManager({self.base_dir!r}, keep_last={self.keep_last})"
