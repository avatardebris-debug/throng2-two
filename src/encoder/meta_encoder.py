"""
meta_encoder.py — Meta-level encoder that learns a challenge descriptor for each game.

Motivation (GAN-of-GANs concept):
    Each game has unique physics, reward structure, and action dynamics.
    A meta-encoder distills this into a compact "challenge descriptor" vector c_game
    so that:
      1. Similar games (e.g., CartPole ≈ MountainCar — both 1D balance tasks) have
         similar c_game vectors.
      2. When a NEW game arrives, we find its nearest c_game neighbour and warm-start
         the world model head from that similar game → faster adaptation.
      3. A meta-policy can choose which world model head to use for action selection.

Architecture:
    Challenge Descriptor Encoder:
        episode_summary → c_game (challenge_dim, e.g., 16)

    Episode summary contains:
        - mean/std of z-vectors across the episode (2 × z_dim)
        - action entropy (1)
        - reward statistics: mean, std, max, cumulative (4)
        - survival fraction: steps/max_steps (1)
        - surprise signal: world model prediction error per step (1)
        Total: 2*z_dim + 7

    Learning:
        - No PyTorch required — NumpyLinear projection (same as UniversalEncoder)
        - Optional: online triplet update:
            c_gA closer to c_gB (same game) than to c_gC (different game)
        - Stored as a persistent registry: game_name → c_game (averaged over N episodes)

    Game routing:
        meta.nearest_game(c_query) → game_name with highest cosine similarity
        meta.recommend_head(c_query) → game_id to warm-start from

    Transfer:
        meta.warm_start_agent(new_game_obs_summary, world_model) → suggested game_id
        This is the key transfer learning hook: an unseen game gets its world model
        head seeded from the most similar known game.

Usage:
    meta = MetaEncoder(z_dim=32, challenge_dim=16)

    # After each episode:
    meta.update(game_name="mario", episode_summary)

    # Before training a new game:
    best_game = meta.recommend_transfer_source("lunarlander", query_summary)
    world_model.warm_start_from(best_game_id)

    # Visualise game clustering:
    meta.cluster_report()
"""
from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════
# EPISODE SUMMARY BUILDER
# ═══════════════════════════════════════════════════════════════

class EpisodeSummary:
    """
    Accumulates per-step statistics and produces a fixed-dim episode summary vector.

    Summary layout (for z_dim=32):
        [0:32]   mean(z) over episode
        [32:64]  std(z)  over episode
        [64]     action entropy
        [65]     mean reward
        [66]     std reward
        [67]     max reward
        [68]     cumulative reward
        [69]     survival fraction (steps / max_steps)
        [70]     mean world model surprise
        Total: 2*z_dim + 7 = 71 for z_dim=32
    """

    def __init__(self, z_dim: int = 32, max_steps: int = 500, n_actions: int = 0):
        self.z_dim = z_dim
        self.max_steps = max_steps
        # n_actions: actual game action space size, for correct entropy computation.
        # If 0 (default), inferred from observed actions only — may be biased if
        # not all actions were explored.
        self.n_actions = n_actions
        self._zs: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._surprises: List[float] = []
        self.steps = 0

    def record(
        self,
        z: np.ndarray,
        action: int,
        reward: float,
        surprise: float = 0.0,
    ):
        """Record one step."""
        self._zs.append(np.asarray(z, dtype=np.float32).flatten()[:self.z_dim])
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._surprises.append(float(surprise))
        self.steps += 1

    def build(self) -> np.ndarray:
        """
        Convert accumulated episode data into a fixed-dim summary vector.

        Returns:
            (2*z_dim + 7,) float32 vector
        """
        if not self._zs:
            return np.zeros(2 * self.z_dim + 7, dtype=np.float32)

        zs = np.stack(self._zs, axis=0)                 # (T, z_dim)
        z_mean = zs.mean(axis=0)                         # (z_dim,)
        z_std  = zs.std(axis=0) + 1e-8                  # (z_dim,)

        # Action entropy: use the game's actual action space size, not just
        # what was observed (avoids bias when some actions were never tried).
        actions_arr = np.array(self._actions)
        # Use game's known n_actions if provided, otherwise infer from observed
        hist_n = max(1, int(actions_arr.max()) + 1)
        n_act = self.n_actions if self.n_actions > 0 else hist_n
        counts = np.bincount(actions_arr, minlength=n_act).astype(float)
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        rwds = np.array(self._rewards, dtype=np.float32)
        r_mean = float(rwds.mean())
        r_std  = float(rwds.std())
        r_max  = float(rwds.max())
        r_cum  = float(rwds.sum())

        survival = min(1.0, self.steps / max(1, self.max_steps))
        surprise = float(np.mean(self._surprises)) if self._surprises else 0.0

        return np.concatenate([
            z_mean,
            z_std,
            [entropy, r_mean, r_std, r_max, r_cum, survival, surprise],
        ], axis=0).astype(np.float32)

    @property
    def summary_dim(self) -> int:
        return 2 * self.z_dim + 7

    def reset(self):
        """Reset for a new episode."""
        self._zs = []
        self._actions = []
        self._rewards = []
        self._surprises = []
        self.steps = 0


# ═══════════════════════════════════════════════════════════════
# NUMPY LINEAR (reused from universal_encoder pattern)
# ═══════════════════════════════════════════════════════════════

class _TripletProjection:
    """
    2-layer numpy MLP trained with triplet margin loss.

    Replaces the random-projection _NumpyLinear so that episode summary
    vectors from the SAME game project closer together than those from
    DIFFERENT games — giving MetaEncoder's cosine similarity real meaning.

    Architecture:   in_dim ──ReLU──> hidden_dim ──> out_dim (L2-normed)
    Training:       Triplet margin loss with cosine distance
    Interface:      __call__(x) same as _NumpyLinear — drop-in replacement.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        seed: int = 0,
    ):
        rng = np.random.RandomState(seed)
        # Layer 1: in → hidden
        lim1 = np.sqrt(6.0 / (in_dim + hidden_dim))
        self.W1 = rng.uniform(-lim1, lim1, (in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        # Layer 2: hidden → out
        lim2 = np.sqrt(6.0 / (hidden_dim + out_dim))
        self.W2 = rng.uniform(-lim2, lim2, (hidden_dim, out_dim)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self._fitted = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: in_dim → out_dim (L2-normalised)."""
        h = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        out = h @ self.W2 + self.b2
        norm = np.linalg.norm(out) + 1e-8
        return out / norm

    def _forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch forward pass: (N, in_dim) → (N, out_dim) unit-normed."""
        H = np.maximum(0, X @ self.W1 + self.b1)   # (N, hidden)
        O = H @ self.W2 + self.b2                   # (N, out)
        norms = np.linalg.norm(O, axis=1, keepdims=True) + 1e-8
        return O / norms, H

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        summaries_by_game: Dict[str, np.ndarray],
        n_epochs: int = 30,
        lr: float = 0.02,
        margin: float = 0.5,
        batch_size: int = 32,
        triplets_per_pair: int = 20,
        seed: int = 42,
    ) -> float:
        """
        Train projection with triplet margin loss.

        Triplet construction (no external data needed):
          Anchor  : summary from game A, episode i
          Positive: summary from game A, episode j  (same game, different ep)
          Negative: summary from game B             (different game)

        Loss: mean(max(0,  d(a,p) - d(a,n) + margin))
              where d is cosine distance = 1 - cosine_similarity

        Args:
            summaries_by_game: {game_name: (N_episodes, in_dim)} episode summaries.
            n_epochs: SGD epochs over the generated triplet set.
            lr: Learning rate.
            margin: Triplet margin (encourage d(a,p) + margin < d(a,n)).
            batch_size: Mini-batch size.
            triplets_per_pair: Triplets to generate per game-pair.
            seed: RNG seed for triplet sampling.

        Returns:
            Final training loss (float).
        """
        rng = np.random.RandomState(seed)
        games = list(summaries_by_game.keys())

        if len(games) < 2:
            return 0.0   # Need ≥2 games for meaningful triplets

        # ── Build triplet set ─────────────────────────────────────
        anchors, positives, negatives = [], [], []
        for i, g_a in enumerate(games):
            S_a = summaries_by_game[g_a]
            n_a = len(S_a)
            if n_a < 2:
                continue
            for g_b in games:
                if g_b == g_a:
                    continue
                S_b = summaries_by_game[g_b]
                if len(S_b) == 0:
                    continue
                for _ in range(triplets_per_pair):
                    ai, pi = rng.choice(n_a, 2, replace=False)
                    ni    = rng.randint(len(S_b))
                    anchors.append(S_a[ai])
                    positives.append(S_a[pi])
                    negatives.append(S_b[ni])

        if len(anchors) < batch_size:
            return 0.0   # Not enough triplets — skip training

        A = np.stack(anchors, axis=0).astype(np.float32)   # (T, in_dim)
        P = np.stack(positives, axis=0).astype(np.float32)
        N_ = np.stack(negatives, axis=0).astype(np.float32)
        T = len(A)

        final_loss = 0.0

        # ── SGD via finite-difference gradients on W2 and W1 ──────
        # We use analytical backprop for speed.
        for epoch in range(n_epochs):
            perm = rng.permutation(T)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, T, batch_size):
                idx = perm[start: start + batch_size]
                a, p, n = A[idx], P[idx], N_[idx]  # (B, in_dim)
                B = len(idx)

                # Forward pass
                za, Ha = self._forward_batch(a)     # (B, out_dim), (B, hidden)
                zp, _  = self._forward_batch(p)
                zn, _  = self._forward_batch(n)

                # Cosine distances (1 - sim); vectors already unit-normed
                d_ap = 1.0 - np.sum(za * zp, axis=1)   # (B,)
                d_an = 1.0 - np.sum(za * zn, axis=1)

                # Triplet loss: max(0, d_ap - d_an + margin)
                loss_vec = np.maximum(0.0, d_ap - d_an + margin)
                loss = loss_vec.mean()
                epoch_loss += loss
                n_batches += 1

                # Backward only for active triplets
                active = loss_vec > 0                    # (B,) bool mask
                if not active.any():
                    continue

                # Gradient of loss w.r.t. za:
                # dL/dza = (1/B) * [active] * (-zp + zn)  (cosine grad simplified)
                scale = active.astype(np.float32) / B
                dza = scale[:, None] * (-zp + zn)        # (B, out_dim)

                # Backprop through Layer 2 (za = (W2ᵀ Ha) / norm)
                # Using approximate grad (ignore norm denominator for stability)
                dL_dO2 = dza                             # approx: ignore norm grad
                dW2 = Ha.T @ dL_dO2                     # (hidden, out_dim)
                db2 = dL_dO2.sum(axis=0)
                dH  = dL_dO2 @ self.W2.T                # (B, hidden)

                # Backprop through ReLU Layer 1
                relu_mask = (Ha > 0).astype(np.float32)
                dH_pre = dH * relu_mask                  # (B, hidden)
                dW1 = a.T @ dH_pre                       # (in_dim, hidden)
                db1 = dH_pre.sum(axis=0)

                # SGD step
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            final_loss = epoch_loss / max(1, n_batches)

        self._fitted = True
        return float(final_loss)


# ═══════════════════════════════════════════════════════════════
# META ENCODER
# ═══════════════════════════════════════════════════════════════

class MetaEncoder:
    """
    Learns a compact challenge descriptor c_game ∈ R^challenge_dim for each game.

    After enough episodes, games with similar dynamics cluster together:
        CartPole ≈ MountainCar  (low-dim, balance/momentum)
        Mario ≈ Montezuma       (2D navigation, platformer)
        MuJoCo ≈ MuJoCo         (continuous physics)

    This clustering is used to:
        1. Route new games to the most similar world model head (warm-start).
        2. Visualise which games transfer to each other.
        3. Select which learned policy to fine-tune for a new task.
    """

    def __init__(
        self,
        z_dim: int = 32,
        challenge_dim: int = 16,
        window: int = 20,         # Rolling window of episodes to average
        seed: int = 0,
    ):
        """
        Args:
            z_dim: z-vector dimension (must match UniversalEncoder z_dim).
            challenge_dim: Output challenge descriptor dimension.
            window: Number of recent episodes to average for stable c_game.
            seed: RNG seed for projection matrix.
        """
        self.z_dim = z_dim
        self.challenge_dim = challenge_dim
        self.window = window

        # Input dim = 2*z_dim + 7 (from EpisodeSummary.build())
        self._in_dim = 2 * z_dim + 7

        # Triplet-trained projection: summary_dim → challenge_dim
        # Starts as random MLP; call fit_projection() to train it.
        self._proj = _TripletProjection(self._in_dim, challenge_dim, seed=seed)

        # Per-game episode summaries — projected c-vectors (for descriptor averages)
        self._episode_summaries: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )

        # Per-game RAW summaries (pre-projection) — used for triplet training
        self._raw_summaries: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max(window, 50))  # keep more for triplets
        )

        # Per-game challenge descriptor (averaged, updated on each episode)
        self._descriptors: Dict[str, np.ndarray] = {}

        # Per-game ID mapping (for world model head warm-start)
        self._game_ids: Dict[str, int] = {}

        # Episode count per game
        self._episode_counts: Dict[str, int] = defaultdict(int)

    # ── REGISTRATION ───────────────────────────────────────────

    def register_game(self, game_name: str, game_id: int):
        """Register a game name → world model head ID mapping."""
        self._game_ids[game_name] = game_id

    def register_games(self, game_id_map: Dict[str, int]):
        """Bulk-register game → game_id mappings."""
        self._game_ids.update(game_id_map)

    # ── ENCODING ───────────────────────────────────────────────

    def encode_summary(self, summary: np.ndarray) -> np.ndarray:
        """
        Project an episode summary to a challenge descriptor.

        Args:
            summary: (summary_dim,) float32 from EpisodeSummary.build()

        Returns:
            c_game: (challenge_dim,) float32, L2-normalised to unit sphere
        """
        summary = np.asarray(summary, dtype=np.float32).flatten()
        # Pad/truncate to expected in_dim
        if len(summary) < self._in_dim:
            summary = np.pad(summary, (0, self._in_dim - len(summary)))
        elif len(summary) > self._in_dim:
            summary = summary[:self._in_dim]

        c = self._proj(summary)
        norm = np.linalg.norm(c) + 1e-8
        return c / norm

    # ── ONLINE UPDATE ──────────────────────────────────────────

    def update(self, game_name: str, episode_summary: np.ndarray):
        """
        Update the challenge descriptor for a game using a new episode summary.

        Args:
            game_name: Name of the game (e.g., "mario").
            episode_summary: (summary_dim,) array from EpisodeSummary.build()
        """
        c = self.encode_summary(episode_summary)
        self._episode_summaries[game_name].append(c)
        self._episode_counts[game_name] += 1

        # Cache raw summary for triplet training
        raw = np.asarray(episode_summary, dtype=np.float32).flatten()
        if len(raw) < self._in_dim:
            raw = np.pad(raw, (0, self._in_dim - len(raw)))
        elif len(raw) > self._in_dim:
            raw = raw[:self._in_dim]
        self._raw_summaries[game_name].append(raw)

        # Update the rolling-average descriptor
        stack = np.stack(list(self._episode_summaries[game_name]), axis=0)
        avg = stack.mean(axis=0)
        norm = np.linalg.norm(avg) + 1e-8
        self._descriptors[game_name] = avg / norm

    @property
    def projection_fitted(self) -> bool:
        """True if the triplet projection has been trained (not just random init)."""
        return self._proj.is_fitted

    def fit_projection(
        self,
        min_episodes_per_game: int = 5,
        n_epochs: int = 30,
        lr: float = 0.02,
        margin: float = 0.5,
        verbose: bool = False,
    ) -> float:
        """
        Train the triplet projection on accumulated episode summaries.

        Constructs same-game (positive) and cross-game (negative) pairs from
        the raw episode summaries collected so far, then runs mini-batch SGD
        to make same-game descriptors cluster together in c_game space.

        Guarded: silently skips if < 2 games or < min_episodes_per_game each.
        After fitting, updates all existing descriptors to use the new projection.

        Args:
            min_episodes_per_game: Minimum episodes per game before fitting.
            n_epochs: Training epochs.
            lr: SGD learning rate.
            margin: Triplet margin.
            verbose: Print training summary.

        Returns:
            Final triplet loss, or 0.0 if skipped.
        """
        # Check we have enough games and episodes
        eligible = {
            g: np.stack(list(sums), axis=0)
            for g, sums in self._raw_summaries.items()
            if len(sums) >= min_episodes_per_game
        }
        if len(eligible) < 2:
            if verbose:
                n = {g: len(s) for g, s in self._raw_summaries.items()}
                print(f"  fit_projection: skipped (need ≥2 games with ≥{min_episodes_per_game} eps each, have {n})")
            return 0.0

        loss = self._proj.fit(eligible, n_epochs=n_epochs, lr=lr, margin=margin)

        # Recompute all descriptors with the newly trained projection
        for game_name, raw_deque in self._raw_summaries.items():
            if game_name not in self._episode_summaries:
                continue
            # Re-project stored raw summaries
            new_cs = [self.encode_summary(r) for r in raw_deque]
            if new_cs:
                stack = np.stack(new_cs, axis=0)
                avg = stack.mean(axis=0)
                norm = np.linalg.norm(avg) + 1e-8
                self._descriptors[game_name] = avg / norm

        if verbose:
            print(f"  fit_projection: {len(eligible)} games, "
                  f"loss={loss:.4f}, descriptors refreshed")
        return loss

    def descriptor(self, game_name: str) -> Optional[np.ndarray]:
        """Return the current challenge descriptor for a game, or None if not seen."""
        return self._descriptors.get(game_name)

    def known_games(self) -> List[str]:
        """List of games with at least one registered episode."""
        return sorted(self._descriptors.keys())

    # ── SIMILARITY ─────────────────────────────────────────────

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit vectors."""
        return float(np.dot(a, b))

    def similarity_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Compute pairwise cosine similarity across all known games.

        Returns:
            (game_names, (N×N) similarity matrix)
        """
        games = self.known_games()
        n = len(games)
        mat = np.zeros((n, n), dtype=np.float32)
        for i, g1 in enumerate(games):
            for j, g2 in enumerate(games):
                d1, d2 = self._descriptors[g1], self._descriptors[g2]
                mat[i, j] = self.cosine_similarity(d1, d2)
        return games, mat

    def nearest_game(
        self,
        query: np.ndarray,
        exclude: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Find the registered game with the highest cosine similarity to query.

        Args:
            query: (challenge_dim,) challenge descriptor
            exclude: Game to exclude (e.g., to find transfer source ≠ self)

        Returns:
            (game_name, similarity_score)
        """
        best_game = ""
        best_sim = -2.0
        q = np.asarray(query, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        for game, desc in self._descriptors.items():
            if game == exclude:
                continue
            sim = self.cosine_similarity(q, desc)
            if sim > best_sim:
                best_sim = sim
                best_game = game

        return best_game, best_sim

    # ── TRANSFER ROUTING ───────────────────────────────────────

    def recommend_transfer_source(
        self,
        new_game_name: str,
        episode_summary: np.ndarray,
    ) -> Tuple[str, int, float]:
        """
        For a new game, find the best world model head to warm-start from.

        Call this before (or early in) training a new game to get:
            - Which known game is most similar
            - That game's world model head ID
            - Similarity score (confidence)

        Args:
            new_game_name: Name of the new/target game.
            episode_summary: Episode summary from first few episodes on new game.

        Returns:
            (source_game_name, source_game_id, similarity_score)
        """
        c_new = self.encode_summary(episode_summary)
        source_game, sim = self.nearest_game(c_new, exclude=new_game_name)

        if not source_game:
            # No known games — default to game_id 0
            return "none", 0, 0.0

        source_id = self._game_ids.get(source_game, 0)
        return source_game, source_id, sim

    # ── CLUSTERING REPORT ──────────────────────────────────────

    def cluster_report(self, top_k: int = 3) -> str:
        """
        Human-readable report showing game similarity clusters.

        Format:
            mario       → (mario: 1.00, montezuma: 0.72, gridworld: 0.41)
            cartpole    → (mountaincar: 0.89, lunarlander: 0.65, mario: 0.31)
            ...

        Returns:
            Formatted string.
        """
        games = self.known_games()
        if not games:
            return "No games registered yet."

        lines = ["Game Challenge Descriptor Clusters:", ""]
        _, mat = self.similarity_matrix()

        for i, game in enumerate(games):
            sims = [(games[j], mat[i, j]) for j in range(len(games)) if j != i]
            sims.sort(key=lambda x: -x[1])
            top = sims[:top_k]
            ep = self._episode_counts[game]
            top_str = ", ".join(f"{g}: {s:.2f}" for g, s in top)
            lines.append(f"  {game:12s} (ep={ep:4d}): top-{top_k} = [{top_str}]")

        return "\n".join(lines)

    def stats(self) -> Dict:
        """Summary statistics."""
        return {
            "z_dim": self.z_dim,
            "challenge_dim": self.challenge_dim,
            "n_games": len(self._descriptors),
            "games": self.known_games(),
            "episode_counts": dict(self._episode_counts),
        }

    def save(self, path: str):
        """Save descriptors to a numpy .npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            f"desc_{name}": desc
            for name, desc in self._descriptors.items()
        }
        data["game_ids"] = np.array(
            list(self._game_ids.values()), dtype=np.int32
        )
        data["game_names"] = np.array(
            list(self._game_ids.keys()), dtype=object
        )
        np.savez(path, **data)

    def load(self, path: str):
        """Load descriptors from a saved .npz file."""
        ckpt = np.load(path, allow_pickle=True)
        # Restore game_ids
        if "game_names" in ckpt and "game_ids" in ckpt:
            for name, gid in zip(
                ckpt["game_names"].tolist(), ckpt["game_ids"].tolist()
            ):
                self._game_ids[str(name)] = int(gid)
        # Restore descriptors
        for key in ckpt.files:
            if key.startswith("desc_"):
                game = key[5:]
                self._descriptors[game] = ckpt[key].astype(np.float32)
                # Synthetic episode count (unknown after load)
                if game not in self._episode_counts:
                    self._episode_counts[game] = self.window  # assume window full

    def __repr__(self) -> str:
        return (
            f"MetaEncoder(z_dim={self.z_dim}, challenge_dim={self.challenge_dim}, "
            f"n_games={len(self._descriptors)})"
        )
