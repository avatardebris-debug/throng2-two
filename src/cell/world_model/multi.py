"""Multi-game world model."""
import os
import numpy as np
from collections import deque
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import WorldModelCore
from .buffer import MultiGameReplayBuffer
from .per_game_heads import scatter_per_game_heads
from .rollout import greedy_action_values

_MULTI_GAME_ONLY_MSG = (
    "MultiGameWorldModel does not implement single-game APIs. "
    "Use train_step_multi_game(), predict_multi(), and store_transition(..., game_id=)."
)


class MultiGameWorldModel(WorldModelCore):
    """
    Cross-game world model that conditions on game identity.

    Extends WorldModelCore with:
      1. Game ID embedding — small learned vector concatenated to input
      2. Per-game dynamics heads — shared encoder + game-specific delta/reward heads
      3. Multi-game replay buffer — balanced sampling across games
      4. Surprise signal — ||predicted_z - actual_z|| as transferable curiosity

    Architecture:
        input: [z_state (feature_dim) | action_onehot (n_actions) | game_embed (game_embed_dim)]
                                             ↓ shared encoder
                                        hidden representation
                          ┌──────────────────┤
                          ▼                  ▼
                    per-game delta head   per-game reward head
                    (game_id selects)     (game_id selects)

    Backward compatible: store_transition(s,a,ns,r) still works using game_id=0.
    For multi-game use: store_transition_multi(s,a,ns,r,game_id).
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        n_games: int = 6,
        game_embed_dim: int = 8,
        hidden_size: int = 128,
        lr: float = 1e-3,
        buffer_size: int = 5000,
        batch_size: int = 64,
        min_transitions: int = 100,
    ):
        """
        Args:
            feature_dim: Dimension of z-vectors from UniversalEncoder.
            n_actions: Max actions across all games (use the largest action space).
            n_games: Number of distinct games.
            game_embed_dim: Dimension of game ID embedding.
            hidden_size: Shared encoder hidden size.
            lr: Learning rate.
            buffer_size: Per-game replay buffer capacity.
            batch_size: Training batch size (split across games).
            min_transitions: Minimum per-game transitions before model is ready.
        """
        super().__init__(
            feature_dim=feature_dim,
            n_actions=n_actions,
            batch_size=batch_size,
            min_transitions=min_transitions,
        )

        self.n_games = n_games
        self.game_embed_dim = game_embed_dim

        device = self.device

        # Game ID embedding table (trainable)
        self._game_embed = nn.Embedding(n_games, game_embed_dim).to(device)

        # Rebuild shared encoder with extended input (includes game embedding)
        input_dim = feature_dim + n_actions + game_embed_dim
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        ).to(device)

        # Per-game dynamics heads (list of Linears, indexed by game_id)
        # delta heads: hidden → feature_dim (residual state prediction)
        self._game_delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, feature_dim),
            )
            for _ in range(n_games)
        ]).to(device)

        # reward heads: hidden → 1
        self._game_reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
            )
            for _ in range(n_games)
        ]).to(device)

        # Horizon heads: hidden → (feature_dim + 1)
        # Output layout: [:feature_dim] = z_delta_N, [feature_dim] = cum_reward_N
        self.horizon_n = 16
        self.min_horizon_n = 4
        self._game_horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, feature_dim + 1),  # z_delta_N + cum_r
            )
            for _ in range(n_games)
        ]).to(device)

        # Rebuild optimizer (all parameters including new horizon heads)
        all_params = (
            list(self._encoder.parameters())
            + list(self._game_embed.parameters())
            + list(self._game_delta_heads.parameters())
            + list(self._game_reward_heads.parameters())
            + list(self._game_horizon_heads.parameters())
        )
        self._optimizer = torch.optim.Adam(all_params, lr=lr)

        # Multi-game buffer (replaces parent's single buffer)
        self._multi_buffer = MultiGameReplayBuffer(
            capacity_per_game=buffer_size,
            sampling="balanced",
            horizon_n=self.horizon_n,   # buffer always accumulates max-length windows
        )
        self._min_per_game = min_transitions

        # Prediction error per game (surprise signal)
        self._game_surprises: Dict[int, deque] = {
            gid: deque(maxlen=100) for gid in range(n_games)
        }

    def replay_size(self) -> int:
        return self._multi_buffer.size

    def _scatter_per_game_heads(
        self,
        encoded: "torch.Tensor",
        game_ids: "torch.Tensor",
        heads: nn.ModuleList,
        trailing_shape: tuple,
    ) -> "torch.Tensor":
        return scatter_per_game_heads(
            encoded,
            game_ids,
            heads,
            trailing_shape,
            device=self.device,
            n_games=self.n_games,
        )

    @property
    def is_ready(self) -> bool:
        """True when every game has enough data and model has been updated."""
        return (
            self._multi_buffer.is_ready(self._min_per_game)
            and self._total_updates >= 10
        )

    def is_ready_for(self, game_id: int) -> bool:
        """
        Per-game readiness check.

        Returns True when THIS specific game has enough transitions and the
        model has seen at least one training update — regardless of whether
        OTHER games have filled their buffers.

        Use this instead of is_ready when running multiple games at different
        speeds (e.g. Mario at 60ep/s, MuJoCo at 0.5ep/s) so the fast games
        can start dreaming without waiting for the slow ones.

        Args:
            game_id: The game to check readiness for.

        Returns:
            bool
        """
        return (
            self._multi_buffer.is_game_ready(game_id, self._min_per_game)
            and self._total_updates >= 1
        )

    # Override confidence to use multi-buffer stats (base class confidence
    # was always 0.0 because _losses is only populated by train_step_multi_game)
    @property
    def confidence(self) -> float:
        """0-1 confidence based on multi-game training progress and loss."""
        if not self._losses or not self.is_ready:
            return 0.0
        avg_loss = float(np.mean(list(self._losses)[-20:]))
        return float(min(1.0, 1.0 / (1.0 + avg_loss)))

    def confidence_for(self, game_id: int) -> float:
        """
        Per-game confidence estimate.

        Uses the game's own surprise history: lower surprise = higher confidence.
        Useful for per-game dream blending weights.

        Returns:
            float in [0, 1]
        """
        surprises = self._game_surprises.get(game_id)
        if not surprises or not self.is_ready_for(game_id):
            return 0.0
        avg_surprise = float(np.mean(list(surprises)[-20:]))
        # Normalize: surprise of 0 → confidence 1.0, surprise of 1+ → approaching 0
        return float(min(1.0, 1.0 / (1.0 + avg_surprise)))

    def adaptive_horizon_n(self, game_id: int = 0) -> int:
        """
        Return the current effective horizon length for a game.

        Adapts based on per-game confidence:
          - Low confidence (early training): min_horizon_n (short, conservative)
          - High confidence (well trained):  horizon_n    (long, ambitious)

        The horizon buffer always stores max-horizon entries — only the effective
        N changes, so no extra buffers are needed. The horizon head is trained on
        the full sequence but inference uses the adaptive N as a signal.

        This is the dynamic/adaptive design: one buffer, confidence-gated depth.

        Returns:
            int in [min_horizon_n, horizon_n]
        """
        conf = self.confidence_for(game_id)
        span = self.horizon_n - self.min_horizon_n
        # Linear interpolation: conf=0 → min, conf=1 → max; round to int
        return int(self.min_horizon_n + round(conf * span))

    def store_transition_multi(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        game_id: int,
    ):
        """Store a transition with explicit game_id for multi-game training."""
        self._multi_buffer.add(state, action, next_state, reward, game_id)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        game_id: int = 0,
    ):
        """Backward-compatible override: also stores in multi-game buffer."""
        if game_id >= self.n_games:
            import warnings
            warnings.warn(
                f"store_transition: game_id={game_id} >= n_games={self.n_games}. "
                "Transition stored in buffer but no world model head will predict it. "
                "Increase n_games or re-check game_id.",
                stacklevel=2,
            )
        self._multi_buffer.add(state, action, next_state, reward, game_id)
        # Don't call super (we manage our own buffer)

    def _forward_multi(
        self,
        states: "torch.Tensor",
        actions_oh: "torch.Tensor",
        game_ids: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Forward pass through shared encoder + per-game heads.

        Returns:
            (delta_preds, reward_preds) — both (batch,) shaped tensors.
        """
        embed = self._game_embed(game_ids)
        x = torch.cat([states, actions_oh, embed], dim=1)
        encoded = self._encoder(x)

        delta_preds = self._scatter_per_game_heads(
            encoded, game_ids, self._game_delta_heads, (self.feature_dim,)
        )
        reward_preds = self._scatter_per_game_heads(
            encoded, game_ids, self._game_reward_heads, (1,)
        )
        return delta_preds, reward_preds

    def train_step_multi_game(self) -> Dict[str, float]:
        """
        Train on a balanced batch from all games.

        Returns:
            Dictionary of training metrics including per-game surprise.
        """
        batch = self._multi_buffer.sample(self.batch_size)
        if len(batch) < 8:
            return {"wm_loss": 0.0, "wm_buffer": self._multi_buffer.size}

        states = torch.FloatTensor(
            np.array([b[0] for b in batch])
        ).to(self.device)
        actions_idx = [b[1] for b in batch]
        next_states = torch.FloatTensor(
            np.array([b[2] for b in batch])
        ).to(self.device)
        rewards = torch.FloatTensor(
            [b[3] for b in batch]
        ).unsqueeze(1).to(self.device)
        game_ids = torch.LongTensor(
            [b[4] for b in batch]
        ).to(self.device)

        # One-hot encode actions
        B = len(batch)
        actions_oh = torch.zeros(B, self.n_actions).to(self.device)
        for i, a in enumerate(actions_idx):
            actions_oh[i, min(int(a), self.n_actions - 1)] = 1.0

        # Forward through multi-game model
        delta_preds, reward_preds = self._forward_multi(states, actions_oh, game_ids)

        # Targets
        delta_target = next_states - states
        reward_target = rewards

        # Losses (step-level)
        state_loss = F.mse_loss(delta_preds, delta_target)
        reward_loss = F.smooth_l1_loss(reward_preds, reward_target)
        total_loss = state_loss + reward_loss * 0.1

        # ── Horizon loss (N-step) ────────────────────────────────
        horizon_loss_val = 0.0
        h_batch = self._multi_buffer.sample_horizon(max(8, self.batch_size // 2))
        if h_batch:
            h_states = torch.FloatTensor(
                np.array([h[0] for h in h_batch])
            ).to(self.device)
            h_actions_idx = [h[1] for h in h_batch]
            h_z_N = torch.FloatTensor(
                np.array([h[2] for h in h_batch])
            ).to(self.device)                      # target: z after N steps
            h_cum_r = torch.FloatTensor(
                [h[3] for h in h_batch]
            ).unsqueeze(1).to(self.device)          # target: N-step return
            h_game_ids = torch.LongTensor(
                [h[4] for h in h_batch]
            ).to(self.device)

            Bh = len(h_batch)
            h_actions_oh = torch.zeros(Bh, self.n_actions).to(self.device)
            for i, a in enumerate(h_actions_idx):
                h_actions_oh[i, min(int(a), self.n_actions - 1)] = 1.0

            # Shared encoder (same forward path, different head)
            h_embed = self._game_embed(h_game_ids)
            h_x = torch.cat([h_states, h_actions_oh, h_embed], dim=1)
            h_encoded = self._encoder(h_x)

            h_out = self._scatter_per_game_heads(
                h_encoded,
                h_game_ids,
                self._game_horizon_heads,
                (self.feature_dim + 1,),
            )
            h_delta_N = h_out[:, : self.feature_dim]
            h_r_pred = h_out[:, self.feature_dim :]

            # Targets for horizon head
            h_delta_N_target = h_z_N - h_states   # actual z_N - z_0 (residual)
            horizon_state_loss = F.mse_loss(h_delta_N, h_delta_N_target)
            horizon_reward_loss = F.smooth_l1_loss(h_r_pred, h_cum_r)
            horizon_loss = horizon_state_loss + horizon_reward_loss * 0.1
            horizon_loss_val = horizon_loss.item()
            total_loss = total_loss + 0.5 * horizon_loss

        # Optimize
        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._encoder.parameters())
            + list(self._game_embed.parameters())
            + list(self._game_delta_heads.parameters())
            + list(self._game_reward_heads.parameters())
            + list(self._game_horizon_heads.parameters()),
            5.0,
        )
        self._optimizer.step()

        self._total_updates += 1
        loss_val = total_loss.item()
        self._losses.append(loss_val)  # feeds confidence property

        # Track per-game surprise (prediction error)
        with torch.no_grad():
            surprise = (delta_preds - delta_target).pow(2).mean(dim=1)
            for i, gid in enumerate(game_ids.tolist()):
                # setdefault handles dynamic game_ids that exceed n_games
                if gid not in self._game_surprises:
                    self._game_surprises[gid] = deque(maxlen=100)
                self._game_surprises[gid].append(float(surprise[i]))

        return {
            "wm_loss": round(loss_val, 4),
            "wm_state_loss": round(state_loss.item(), 4),
            "wm_horizon_loss": round(horizon_loss_val, 4),
            "wm_updates": self._total_updates,
            "wm_buffer": self._multi_buffer.size,
            "horizon_size": self._multi_buffer.horizon_size(),
            "game_sizes": self._multi_buffer.game_sizes(),
        }

    def dream_horizon(
        self,
        features: np.ndarray,
        game_id: int = 0,
    ) -> np.ndarray:
        """
        Estimate N-step returns for each action using the horizon head.

        Complexity: O(n_actions) — one encoder forward pass per action,
        no rollout chain. Covers horizon_n steps of lookahead in the same
        cost as dream_all_actions(depth=1).

        This is the SLOW PATH in multi-timescale dreaming:
          - Called every K steps (not every step)
          - Covers self.horizon_n steps (default 8)
          - Cached by CellDreamer between refreshes

        Args:
            features: (feature_dim,) current state z-vector.
            game_id: Game to use the horizon head for.

        Returns:
            (n_actions,) float32 estimated N-step cumulative returns.
            Returns zeros if not ready or game_id out of range.
        """
        if not self.is_ready_for(game_id) or game_id >= self.n_games:
            return np.zeros(self.n_actions, dtype=np.float32)

        action_values = np.zeros(self.n_actions, dtype=np.float32)

        with torch.inference_mode():
            state_t = torch.as_tensor(
                features, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            gid_t = torch.LongTensor([game_id]).to(self.device)
            embed = self._game_embed(gid_t)                 # (1, embed_dim)

            for a in range(self.n_actions):
                action_oh = torch.zeros(1, self.n_actions, device=self.device)
                action_oh[0, min(a, self.n_actions - 1)] = 1.0

                x = torch.cat([state_t, action_oh, embed], dim=1)
                encoded = self._encoder(x)
                out = self._game_horizon_heads[game_id](encoded)  # (1, feat+1)

                # Output layout: [:feature_dim] = z_delta_N, [feature_dim] = cum_reward
                cum_reward = out[0, self.feature_dim].item()
                action_values[a] = cum_reward

        return action_values

    def predict_multi(
        self,
        features: np.ndarray,
        action: int,
        game_id: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict next state and reward, conditioned on game_id.

        Returns: (predicted_next_features, predicted_reward)
        """
        with torch.inference_mode():
            state_t = torch.as_tensor(
                features, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            action_oh = torch.zeros(1, self.n_actions).to(self.device)
            action_oh[0, min(action, self.n_actions - 1)] = 1.0
            gid_t = torch.LongTensor([game_id]).to(self.device)

            delta, reward = self._forward_multi(state_t, action_oh, gid_t)
            next_features = (state_t + delta).squeeze(0).cpu().numpy()
            pred_reward = reward.item()

        return next_features, pred_reward

    def dream_all_actions(
        self,
        features: np.ndarray,
        depth: int = 1,
        gamma: float = 0.99,
        game_id: int = 0,
    ) -> np.ndarray:
        del gamma
        if not self.is_ready_for(game_id):
            return np.zeros(self.n_actions, dtype=np.float32)

        def predict_step(state: np.ndarray, action: int):
            return self.predict_multi(state, action, game_id)

        return greedy_action_values(
            predict_step, features, self.n_actions, depth=depth
        )

    def surprise(self, game_id: int) -> float:
        """Average recent prediction error (surprise) for one game."""
        buf = self._game_surprises.get(game_id, deque())
        if not buf:
            return 0.0
        return float(np.mean(list(buf)[-20:]))

    def save(self, path: str):
        """Save multi-game WM weights and optimizer state to a .pt checkpoint."""
        import torch

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "encoder": self._encoder.state_dict(),
                "game_embed": self._game_embed.state_dict(),
                "game_delta_heads": self._game_delta_heads.state_dict(),
                "game_reward_heads": self._game_reward_heads.state_dict(),
                "game_horizon_heads": self._game_horizon_heads.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "total_updates": self._total_updates,
                "losses": list(self._losses),
                "n_games": self.n_games,
            },
            path,
        )

    def load(self, path: str):
        """Load multi-game WM weights from a checkpoint saved by save()."""
        import torch

        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._encoder.load_state_dict(ckpt["encoder"])
        self._game_embed.load_state_dict(ckpt["game_embed"])
        self._game_delta_heads.load_state_dict(ckpt["game_delta_heads"])
        self._game_reward_heads.load_state_dict(ckpt["game_reward_heads"])
        self._game_horizon_heads.load_state_dict(ckpt["game_horizon_heads"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        self._total_updates = ckpt.get("total_updates", 0)
        self._losses.extend(ckpt.get("losses", []))

    def train_step(self) -> Dict[str, float]:
        raise NotImplementedError(_MULTI_GAME_ONLY_MSG)

    def predict(self, features: np.ndarray, action: int, game_id: int = 0):
        """Delegate to predict_multi (done_prob=0; multi model has no done head)."""
        next_features, pred_reward = self.predict_multi(features, action, game_id)
        return next_features, pred_reward, 0.0

    def save_weights(self, path: str) -> None:
        raise NotImplementedError("Use MultiGameWorldModel.save(path) instead.")

    def load_weights(self, path: str) -> None:
        raise NotImplementedError("Use MultiGameWorldModel.load(path) instead.")

    def state_dict_all(self) -> dict:
        raise NotImplementedError("Use MultiGameWorldModel.save(path) instead.")

    def load_state_dict_all(self, state: dict) -> None:
        raise NotImplementedError("Use MultiGameWorldModel.load(path) instead.")

    def multi_stats(self) -> Dict[str, object]:
        """Extended stats for the multi-game model."""
        base = self.stats()
        base.update({
            "multi_buffer_size": self._multi_buffer.size,
            "game_buffer_sizes": self._multi_buffer.game_sizes(),
            "multi_ready": self.is_ready,
            "game_surprise": {
                gid: round(self.surprise(gid), 4)
                for gid in self._game_surprises
            },
        })
        return base

