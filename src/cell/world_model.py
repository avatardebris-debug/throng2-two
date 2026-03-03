"""
CellWorldModel — Lightweight learned dynamics for the ThrongletCell.

Adapted from throng5's WorldModel pattern:
  - Residual MLP: predicts (next_state = state + delta, reward)
  - Replay buffer for experience storage
  - dream_all_actions() for hypothesis evaluation

Sized for cell's compressed feature space (~20-37 dim, 2-6 actions),
not throng5's 84-dim CNN features.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CellWorldModel:
    """
    Learned dynamics model for the ThrongletCell.

    Predicts (next_state, reward) given (state, action).
    Uses residual state prediction: next_state = state + delta.
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        hidden_size: int = 128,
        lr: float = 1e-3,
        buffer_size: int = 10000,
        batch_size: int = 64,
        min_transitions: int = 200,
    ):
        """
        Args:
            feature_dim: Dimension of the combined feature vector.
            n_actions: Number of discrete actions.
            hidden_size: Hidden layer size.
            lr: Learning rate.
            buffer_size: Replay buffer capacity.
            batch_size: Training batch size.
            min_transitions: Minimum transitions before model is ready.
        """
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.min_transitions = min_transitions

        # Replay buffer
        self._replay = deque(maxlen=buffer_size)
        self._total_updates = 0
        self._losses = deque(maxlen=100)

        # Architecture: (features + action_onehot) → hidden → (delta, reward)
        input_dim = feature_dim + n_actions

        # Device (auto-detect GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        ).to(self.device)

        # State delta head
        self._state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, feature_dim),
        ).to(self.device)

        # Reward head
        self._reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        ).to(self.device)

        # Optimizer
        all_params = (
            list(self._encoder.parameters())
            + list(self._state_head.parameters())
            + list(self._reward_head.parameters())
        )
        self._optimizer = optim.Adam(all_params, lr=lr)

    @property
    def is_ready(self) -> bool:
        """True if model has enough data to produce useful predictions."""
        return len(self._replay) >= self.min_transitions and self._total_updates >= 10

    @property
    def confidence(self) -> float:
        """0-1 confidence based on training progress and loss."""
        if not self._losses or not self.is_ready:
            return 0.0
        avg_loss = np.mean(list(self._losses)[-20:])
        # Confidence inversely proportional to loss, capped at 1.0
        return float(min(1.0, 1.0 / (1.0 + avg_loss)))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
    ):
        """Store a real transition for training."""
        self._replay.append((
            np.asarray(state, dtype=np.float32),
            action,
            np.asarray(next_state, dtype=np.float32),
            float(reward),
        ))

    def train_step(self) -> Dict[str, float]:
        """Train on a batch of real transitions."""
        if len(self._replay) < self.batch_size:
            return {"wm_loss": 0.0, "wm_buffer": len(self._replay)}

        # Sample batch
        indices = np.random.choice(len(self._replay), self.batch_size, replace=False)
        batch = [self._replay[i] for i in indices]

        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions_idx = [b[1] for b in batch]
        next_states = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)
        rewards = torch.FloatTensor([b[3] for b in batch]).unsqueeze(1).to(self.device)

        # One-hot encode actions
        actions_oh = torch.zeros(self.batch_size, self.n_actions).to(self.device)
        for i, a in enumerate(actions_idx):
            actions_oh[i, a] = 1.0

        # Forward
        x = torch.cat([states, actions_oh], dim=1)
        encoded = self._encoder(x)
        delta_pred = self._state_head(encoded)
        reward_pred = self._reward_head(encoded)

        # Targets
        delta_target = next_states - states
        reward_target = rewards

        # Losses
        state_loss = F.mse_loss(delta_pred, delta_target)
        reward_loss = F.smooth_l1_loss(reward_pred, reward_target)
        total_loss = state_loss + reward_loss * 0.1

        # Optimize
        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._encoder.parameters())
            + list(self._state_head.parameters())
            + list(self._reward_head.parameters()),
            5.0,
        )
        self._optimizer.step()

        self._total_updates += 1
        loss_val = total_loss.item()
        self._losses.append(loss_val)

        return {
            "wm_loss": round(loss_val, 4),
            "wm_state_loss": round(state_loss.item(), 4),
            "wm_reward_loss": round(reward_loss.item(), 4),
            "wm_updates": self._total_updates,
        }

    def predict(
        self, features: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float]:
        """
        Predict next state and reward for (state, action).

        Returns: (predicted_next_features, predicted_reward)
        """
        with torch.inference_mode():
            state_t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_oh = torch.zeros(1, self.n_actions).to(self.device)
            action_oh[0, action] = 1.0

            x = torch.cat([state_t, action_oh], dim=1)
            encoded = self._encoder(x)
            delta = self._state_head(encoded)
            reward = self._reward_head(encoded)

            next_features = (state_t + delta).squeeze(0).cpu().numpy()
            pred_reward = reward.item()

        return next_features, pred_reward

    def dream_all_actions(
        self,
        features: np.ndarray,
        depth: int = 3,
        gamma: float = 0.99,
    ) -> np.ndarray:
        """
        Evaluate all actions by dreaming depth steps ahead.

        For each action: take it, then greedily pick best actions
        for remaining steps. Accumulate discounted reward.

        Returns: action_values (n_actions,) — estimated return per action.
        """
        if not self.is_ready:
            return np.zeros(self.n_actions)

        action_values = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # Simulate first step
            next_feat, r = self.predict(features, a)
            total_return = r

            # Continue greedily for remaining steps
            feat = next_feat
            for d in range(1, depth):
                best_r = -float("inf")
                best_a = 0
                for a2 in range(self.n_actions):
                    _, pr = self.predict(feat, a2)
                    if pr > best_r:
                        best_r = pr
                        best_a = a2
                feat, r = self.predict(feat, best_a)
                total_return += (gamma ** d) * r

            action_values[a] = total_return

        return action_values

    def stats(self) -> dict:
        """World model statistics."""
        return {
            "buffer_size": len(self._replay),
            "total_updates": self._total_updates,
            "is_ready": self.is_ready,
            "confidence": round(self.confidence, 3),
            "avg_loss": round(float(np.mean(list(self._losses)[-20:])), 4)
            if self._losses else 0.0,
        }

    def state_dict_all(self) -> dict:
        """Full state for checkpointing."""
        return {
            "encoder": self._encoder.state_dict(),
            "state_head": self._state_head.state_dict(),
            "reward_head": self._reward_head.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "total_updates": self._total_updates,
        }

    def load_state_dict_all(self, state: dict):
        """Restore from checkpoint."""
        self._encoder.load_state_dict(state["encoder"])
        self._state_head.load_state_dict(state["state_head"])
        self._reward_head.load_state_dict(state["reward_head"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._total_updates = state.get("total_updates", 0)
