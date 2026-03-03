"""
PPO Head — Lightweight actor-critic for the ThrongletCell.

Self-contained, single-process PPO with:
- Shared backbone + separate policy/value heads
- GAE advantage estimation
- Clipped surrogate objective
- Entropy bonus for exploration

This is intentionally minimal (~250 lines). The full RL Zoo PPO has
multi-process actors, which we don't need for a single cell.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ActorCritic(nn.Module):
    """Combined actor-critic network."""

    def __init__(self, input_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        logits = self.policy_head(shared)
        value = self.value_head(shared).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


class RolloutBuffer:
    """Stores transitions for PPO batch updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self):
        return len(self.obs)


class PPOHead:
    """
    Lightweight PPO actor-critic.

    Designed for single-cell training — no multi-process, no fancy scheduling.
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        rollout_length: int = 128,
    ):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length

        # Networks
        self.network = ActorCritic(input_dim, n_actions, hidden)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Stats
        self._update_count = 0
        self._total_loss = 0.0
        self._total_policy_loss = 0.0
        self._total_value_loss = 0.0

    def select_action(
        self, features: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Select action given features.

        Returns: (action, log_prob, value)
        """
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
            action, log_prob, _, value = self.network.get_action_and_value(x)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def store_transition(
        self, obs, action, log_prob, reward, done, value
    ):
        """Store a transition in the rollout buffer."""
        self.buffer.add(obs, action, log_prob, reward, done, value)

    def should_update(self) -> bool:
        """Check if we have enough data for an update."""
        return len(self.buffer) >= self.rollout_length

    def update(self, last_value: float = 0.0) -> dict:
        """
        Run PPO update on collected rollout.

        Args:
            last_value: Bootstrap value for the last state (0 if done).

        Returns:
            Loss statistics dict.
        """
        if len(self.buffer) == 0:
            return {}

        # Convert buffer to tensors
        obs = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)

        # Compute GAE advantages
        advantages, returns = self._compute_gae(
            rewards, values, dones, last_value
        )
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        n = len(obs)
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.update_epochs):
            # Shuffle indices
            indices = torch.randperm(n)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                # Get current policy/value for batch
                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(obs[idx], actions[idx])
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                pg_loss1 = -advantages[idx] * ratio
                pg_loss2 = -advantages[idx] * torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = F.mse_loss(new_values, returns[idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    pg_loss
                    + self.value_coef * v_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        # Clear buffer
        self.buffer.clear()
        self._update_count += 1

        stats = {
            "policy_loss": round(total_pg_loss / max(1, n_updates), 4),
            "value_loss": round(total_v_loss / max(1, n_updates), 4),
            "entropy": round(total_entropy / max(1, n_updates), 4),
            "n_updates": n_updates,
        }
        self._total_loss += total_pg_loss + total_v_loss
        self._total_policy_loss += total_pg_loss
        self._total_value_loss += total_v_loss
        return stats

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + values
        return advantages, returns

    def get_value(self, features: np.ndarray) -> float:
        """Get value estimate for a state."""
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
            _, value = self.network(x)
        return float(value.item())

    def stats(self) -> dict:
        """PPO statistics."""
        return {
            "update_count": self._update_count,
            "avg_loss": round(
                self._total_loss / max(1, self._update_count), 4
            ),
            "buffer_size": len(self.buffer),
        }
