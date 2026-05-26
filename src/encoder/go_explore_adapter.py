"""
go_explore_adapter.py — Pure-numpy Go-Explore adapter for Throng2.

Strips all Horovod / MPI / TF dependencies from the Uber implementation.
Cell keys are discretised z-vectors from UniversalEncoder (not RAM positions),
making this game-agnostic.

Components
----------
CellEntry
    Lightweight dataclass: score, trajectory_len, nb_visits, z_centroid.

ZCellArchive
    Single-process archive.  Maps cell_key → CellEntry.
    Cell selection weighted by 1 / (nb_visits + 1)  (novel cells win).
    Accepts a cell if: better score OR shorter path at same score.

GoExploreRunner
    Runs one "explore episode": sample a cell from the archive, attempt
    to navigate there via a short random roll-out, then explore from
    wherever we land and record any new cells discovered.

GoExploreMetaRouter
    Uses MetaEncoder.similarity_matrix() to find the most similar
    game that has a populated archive, then seeds the new game's archive
    with the top-K cells from the source game.  This is the cross-game
    knowledge-transfer layer described in P3.2.

Usage
-----
    from src.encoder.go_explore_adapter import ZCellArchive, GoExploreRunner, GoExploreMetaRouter

    archive = ZCellArchive(z_dim=32)
    runner  = GoExploreRunner(archive)

    # After each regular training episode:
    runner.run_explore_episode(env_runner, encoder, game_name, n_random_steps=50)

    # When starting a new game with MetaEncoder available:
    router = GoExploreMetaRouter(meta_encoder, archives_by_game)
    seed_cells = router.seed_for(new_game_name, top_k=20)
    archives_by_game[new_game_name].seed_from(seed_cells)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# ═══════════════════════════════════════════════════════════════
# CELL ENTRY
# ═══════════════════════════════════════════════════════════════

@dataclass
class CellEntry:
    """
    One entry in the Go-Explore archive.

    Attributes
    ----------
    score : float
        Best cumulative reward seen on a trajectory reaching this cell.
    trajectory_len : int
        Length of the shortest trajectory that reached this cell with
        `score` or better.
    nb_visits : int
        How many times this cell has been sampled for exploration.
    z_centroid : np.ndarray
        Running mean of the z-vectors that mapped to this cell key.
        Used to reconstruct a target embedding for goal-conditioned return.
    """
    score: float = 0.0
    trajectory_len: int = 0
    nb_visits: int = 0
    z_centroid: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))

    def update_centroid(self, z: np.ndarray, alpha: float = 0.1) -> None:
        """EMA update of the centroid."""
        if self.z_centroid.shape != z.shape:
            self.z_centroid = z.copy()
        else:
            self.z_centroid = (1 - alpha) * self.z_centroid + alpha * z


# ═══════════════════════════════════════════════════════════════
# Z-CELL ARCHIVE
# ═══════════════════════════════════════════════════════════════

class ZCellArchive:
    """
    Single-process Go-Explore archive keyed by discretised z-vectors.

    Cell keys
    ---------
    A cell key is a tuple of int16 values: the z-vector components
    multiplied by `resolution` and rounded.  At resolution=8 and
    z_dim=32 this gives 32-dimensional integer cells, which is a
    coarser but generalising representation compared to Atari RAM state.

    Selection strategy
    ------------------
    Cells are sampled with probability ∝ 1 / (nb_visits + 1) so that
    novel (rarely visited) cells are prioritised for exploration.
    """

    def __init__(
        self,
        z_dim: int = 32,
        resolution: int = 8,
        max_cells: int = 50_000,
    ):
        self.z_dim = z_dim
        self.resolution = resolution
        self.max_cells = max_cells
        self._archive: Dict[tuple, CellEntry] = {}
        self._total_adds = 0

    # ── cell key ──────────────────────────────────────────────

    def cell_key(self, z: np.ndarray) -> tuple:
        """Discretise a unit-sphere z-vector to a hashable cell key."""
        return tuple((z * self.resolution).round().astype(np.int16).tolist())

    # ── accept / add ──────────────────────────────────────────

    def should_accept(
        self,
        key: tuple,
        score: float,
        trajectory_len: int,
    ) -> bool:
        """Return True if this (score, length) pair improves the stored entry."""
        if key not in self._archive:
            return True
        existing = self._archive[key]
        if score > existing.score:
            return True
        if score == existing.score and trajectory_len < existing.trajectory_len:
            return True
        return False

    def add(
        self,
        z: np.ndarray,
        score: float,
        trajectory_len: int,
    ) -> bool:
        """
        Add or update a cell.

        Returns True if the archive was updated (new or improved entry).
        """
        key = self.cell_key(z)
        if not self.should_accept(key, score, trajectory_len):
            return False

        if key in self._archive:
            entry = self._archive[key]
            entry.score = score
            entry.trajectory_len = trajectory_len
            entry.update_centroid(z)
        else:
            if len(self._archive) >= self.max_cells:
                # Evict the most-visited cell (least novel)
                worst = max(self._archive, key=lambda k: self._archive[k].nb_visits)
                del self._archive[worst]
            self._archive[key] = CellEntry(
                score=score,
                trajectory_len=trajectory_len,
                nb_visits=0,
                z_centroid=z.copy(),
            )

        self._total_adds += 1
        return True

    def seed_from(self, cells: List[Tuple[tuple, CellEntry]]) -> int:
        """
        Bulk-insert cells from another archive (cross-game seeding).

        Returns number of cells actually added.
        """
        added = 0
        for key, entry in cells:
            z_approx = entry.z_centroid
            if self.add(z_approx, entry.score, entry.trajectory_len):
                added += 1
        return added

    # ── sampling ──────────────────────────────────────────────

    def sample_cell(self, rng: Optional[np.random.RandomState] = None) -> Optional[tuple]:
        """
        Sample a cell key weighted by novelty (1 / (nb_visits + 1)).

        Returns None if archive is empty.
        """
        if not self._archive:
            return None
        keys = list(self._archive.keys())
        weights = np.array(
            [1.0 / (self._archive[k].nb_visits + 1) for k in keys],
            dtype=np.float64,
        )
        weights /= weights.sum()
        if rng is not None:
            idx = rng.choice(len(keys), p=weights)
        else:
            idx = np.random.choice(len(keys), p=weights)
        chosen = keys[idx]
        self._archive[chosen].nb_visits += 1
        return chosen

    def get_z_target(self, key: tuple) -> Optional[np.ndarray]:
        """Return the stored z-centroid for a given cell key."""
        if key not in self._archive:
            return None
        return self._archive[key].z_centroid.copy()

    def top_k_cells(self, k: int = 20) -> List[Tuple[tuple, CellEntry]]:
        """Return top-K cells by score (descending), for cross-game seeding."""
        sorted_cells = sorted(
            self._archive.items(),
            key=lambda item: item[1].score,
            reverse=True,
        )
        return sorted_cells[:k]

    # ── stats ─────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._archive)

    def stats(self) -> dict:
        if not self._archive:
            return {"n_cells": 0, "max_score": 0.0, "mean_visits": 0.0}
        scores = [e.score for e in self._archive.values()]
        visits = [e.nb_visits for e in self._archive.values()]
        return {
            "n_cells": len(self._archive),
            "total_adds": self._total_adds,
            "max_score": round(float(max(scores)), 4),
            "mean_visits": round(float(np.mean(visits)), 2),
        }

    def __repr__(self) -> str:
        return f"ZCellArchive(size={self.size}, z_dim={self.z_dim}, res={self.resolution})"

    # ── persistence ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save archive to a numpy .npz file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not self._archive:
            np.savez(path, keys=np.zeros((0, self.z_dim), dtype=np.int16),
                     scores=np.array([]), visits=np.array([]),
                     centroids=np.zeros((0, self.z_dim), dtype=np.float32),
                     meta=np.array([self.z_dim, self.resolution, self.max_cells,
                                    self._total_adds], dtype=np.int64))
            return
        keys     = np.array(list(self._archive.keys()), dtype=np.int16)
        entries  = list(self._archive.values())
        scores   = np.array([e.score for e in entries], dtype=np.float32)
        visits   = np.array([e.nb_visits for e in entries], dtype=np.int32)
        traj_len = np.array([e.trajectory_len for e in entries], dtype=np.int32)
        centroids = np.stack([e.z_centroid for e in entries], axis=0)
        np.savez(path,
                 keys=keys, scores=scores, visits=visits,
                 traj_len=traj_len, centroids=centroids,
                 meta=np.array([self.z_dim, self.resolution, self.max_cells,
                                self._total_adds], dtype=np.int64))

    def load(self, path: str) -> None:
        """Load archive from a numpy .npz file (in-place, clears existing data)."""
        d = np.load(path, allow_pickle=False)
        meta = d["meta"]
        self.z_dim      = int(meta[0])
        self.resolution = int(meta[1])
        self.max_cells  = int(meta[2])
        self._total_adds = int(meta[3])
        self._archive   = {}
        keys     = d["keys"]
        if len(keys) == 0:
            return
        scores   = d["scores"]
        visits   = d["visits"]
        traj_len = d["traj_len"]
        centroids = d["centroids"]
        for i in range(len(keys)):
            key = tuple(keys[i].tolist())
            self._archive[key] = CellEntry(
                score=float(scores[i]),
                trajectory_len=int(traj_len[i]),
                nb_visits=int(visits[i]),
                z_centroid=centroids[i].astype(np.float32),
            )



# ═══════════════════════════════════════════════════════════════
# GO-EXPLORE RUNNER
# ═══════════════════════════════════════════════════════════════

class GoExploreRunner:
    """
    Single-process Go-Explore exploration loop.

    Each call to `run_explore_episode()`:
      1. Samples a target cell from the archive.
      2. Rolls out N random steps to explore from the current state,
         since true state restoration (reset-to-position) isn't guaranteed.
      3. Records every z-vector seen as a candidate cell, adding any
         improvements to the archive.

    This is a simplified "Phase 1" of Go-Explore — no self-imitation
    learning or goal-conditioned return policy yet.  The archive
    produced here feeds the MetaRouter for cross-game transfer.
    """

    def __init__(
        self,
        archive: ZCellArchive,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.archive = archive
        self.rng = rng or np.random.RandomState(0)
        self._episodes_run = 0

    def run_explore_episode(
        self,
        runner,
        encoder,
        game_name: str,
        n_random_steps: int = 50,
        max_episode_steps: int = 200,
    ) -> dict:
        """
        Run one Go-Explore episode: explore and update archive.

        Args:
            runner: Game runner with .reset(), .step(action), .n_actions.
            encoder: EncoderRegistry or UniversalEncoder with .encode(game_name, obs).
            game_name: Name of the current game.
            n_random_steps: Steps to take from sampled start point.
            max_episode_steps: Hard cap on episode length.

        Returns:
            dict with n_new_cells, n_updated_cells, episode_score.
        """
        n_new = 0
        n_updated = 0
        total_reward = 0.0

        try:
            obs = runner.reset()
        except Exception:
            return {"n_new_cells": 0, "n_updated_cells": 0, "episode_score": 0.0}

        # Encode start state and add to archive
        if hasattr(encoder, "encode") and callable(encoder.encode):
            try:
                # EncoderRegistry signature
                z = encoder.encode(game_name, obs)
            except TypeError:
                z = encoder.encode(obs)
        else:
            z = np.zeros(self.archive.z_dim, dtype=np.float32)

        added = self.archive.add(z, score=0.0, trajectory_len=0)
        if added:
            n_new += 1

        step = 0
        cumulative_reward = 0.0
        trajectory_len = 0

        while step < max_episode_steps:
            n_actions = getattr(runner, "n_actions", 4)
            action = self.rng.randint(n_actions)
            try:
                result = runner.step(action)
                next_obs, reward, done = result[0], result[1], result[2]
            except Exception:
                break

            cumulative_reward += float(reward)
            total_reward += float(reward)
            trajectory_len += 1
            step += 1

            try:
                if hasattr(encoder, "encode") and callable(encoder.encode):
                    try:
                        z_next = encoder.encode(game_name, next_obs)
                    except TypeError:
                        z_next = encoder.encode(next_obs)
                else:
                    z_next = np.zeros(self.archive.z_dim, dtype=np.float32)
            except Exception:
                z_next = np.zeros(self.archive.z_dim, dtype=np.float32)

            was_new_before = self.archive.cell_key(z_next) not in self.archive._archive
            accepted = self.archive.add(z_next, cumulative_reward, trajectory_len)
            if accepted:
                if was_new_before:
                    n_new += 1
                else:
                    n_updated += 1

            obs = next_obs
            if done:
                break

        self._episodes_run += 1
        return {
            "n_new_cells": n_new,
            "n_updated_cells": n_updated,
            "episode_score": total_reward,
        }

    @property
    def episodes_run(self) -> int:
        return self._episodes_run


# ═══════════════════════════════════════════════════════════════
# GO-EXPLORE META ROUTER
# ═══════════════════════════════════════════════════════════════

class GoExploreMetaRouter:
    """
    Cross-game knowledge transfer using MetaEncoder similarity.

    When starting exploration on a new game, seeding its archive from
    the most similar existing game's archive gives the agent a head start:
    it begins exploration from known interesting regions rather than from
    scratch at the start state.

    Uses MetaEncoder.similarity_matrix() which returns a {game_name → score}
    cosine similarity dict relative to a query game.
    """

    def __init__(
        self,
        meta_encoder,
        archives_by_game: Dict[str, ZCellArchive],
    ):
        """
        Args:
            meta_encoder: MetaEncoder instance with .similarity_matrix().
            archives_by_game: Dict mapping game_name → ZCellArchive.
        """
        self.meta_encoder = meta_encoder
        self.archives = archives_by_game

    def best_source_game(
        self,
        target_game: str,
        min_cells: int = 5,
    ) -> Optional[str]:
        """
        Find the most similar game that has a populated archive.

        Args:
            target_game: The game we want to seed.
            min_cells: Minimum archive size to be considered a valid source.

        Returns:
            Name of the best source game, or None if no suitable game found.
        """
        try:
            sim_matrix = self.meta_encoder.similarity_matrix()
        except Exception:
            return None

        if target_game not in sim_matrix:
            return None

        similarities = sim_matrix[target_game]  # {other_game → float}
        best_game = None
        best_sim = -1.0

        for game_name, sim in similarities.items():
            if game_name == target_game:
                continue
            archive = self.archives.get(game_name)
            if archive is None or archive.size < min_cells:
                continue
            if sim > best_sim:
                best_sim = sim
                best_game = game_name

        return best_game

    def seed_for(
        self,
        target_game: str,
        top_k: int = 20,
        min_cells: int = 5,
    ) -> int:
        """
        Seed the target game's archive from the most similar source game.

        Args:
            target_game: Game to seed.
            top_k: Number of top-scoring cells to copy.
            min_cells: Minimum source archive size.

        Returns:
            Number of cells actually inserted into target archive.
        """
        source_game = self.best_source_game(target_game, min_cells)
        if source_game is None:
            return 0

        source_archive = self.archives.get(source_game)
        target_archive = self.archives.get(target_game)
        if source_archive is None or target_archive is None:
            return 0

        top_cells = source_archive.top_k_cells(top_k)
        n_seeded = target_archive.seed_from(top_cells)
        return n_seeded

    def stats(self) -> dict:
        """Archive sizes for all registered games."""
        return {g: a.stats() for g, a in self.archives.items()}
