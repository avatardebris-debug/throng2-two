"""
mario_torch_agent.py -- PyTorch PPO-ICM agent for Mario ASCII.

GPU-accelerated version of mario_icm_agent.py. Key differences:
  - Batched PPO updates (whole rollout at once, not per-sample)
  - Proper autograd for ICM (no manual backprop)
  - CUDA support (auto-detects GPU)
  - ~50-100x faster than numpy on GPU

Same step()/learn()/reset() interface as MarioICMAgent.

Requirements:
  pip install torch

Usage:
  agent = MarioTorchAgent(obs_dim=378, n_actions=6)
  action = agent.step(obs)           # obs is numpy array
  agent.learn(reward, done, next_obs) # learn with ICM
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ═══════════════════════════════════════════════════════════════
# NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """
    Actor-Critic policy network with CNN+CoordConv spatial encoder.

    Splits the 378-dim observation into:
      - Spatial: 320 values → reshape to (1, 16, 20) grid + 2 CoordConv channels
      - Non-spatial: 58 values (position, physics, enemies, status)

    Spatial stream: Conv2d → ReLU → MaxPool → Conv2d → ReLU → Flatten
    Both streams concatenated → Dense → Actor + Critic heads
    """

    GRID_SIZE = 320    # 16 × 20 viewport tiles
    GRID_H = 16
    GRID_W = 20

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden1: int = 128, hidden2: int = 64,
                 conv_channels: int = 16):
        super().__init__()

        non_spatial_dim = obs_dim - self.GRID_SIZE  # 58

        # Pre-compute CoordConv coordinate grids (registered as buffers)
        y_coords = torch.linspace(0, 1, self.GRID_H).unsqueeze(1).expand(self.GRID_H, self.GRID_W)
        x_coords = torch.linspace(0, 1, self.GRID_W).unsqueeze(0).expand(self.GRID_H, self.GRID_W)
        self.register_buffer('x_coords', x_coords.unsqueeze(0))  # (1, H, W)
        self.register_buffer('y_coords', y_coords.unsqueeze(0))  # (1, H, W)

        # Spatial encoder: 3 input channels (tile_type + x_coord + y_coord)
        self.conv1 = nn.Conv2d(3, conv_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # (16, 8, 10)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        # After pool1: (ch*2, 8, 10) → flatten = ch*2 * 8 * 10

        conv_flat_dim = conv_channels * 2 * 8 * 10  # 2560 with 16 channels

        # Combined MLP: spatial features + non-spatial features
        combined_dim = conv_flat_dim + non_spatial_dim
        self.fc1 = nn.Linear(combined_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.actor = nn.Linear(hidden2, n_actions)
        self.critic = nn.Linear(hidden2, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        # Split spatial vs non-spatial
        spatial_flat = x[:, :self.GRID_SIZE]
        non_spatial = x[:, self.GRID_SIZE:]

        # Reshape grid: (batch, 320) → (batch, 1, 16, 20)
        grid = spatial_flat.reshape(batch_size, 1, self.GRID_H, self.GRID_W)

        # Add CoordConv channels: (batch, 3, 16, 20)
        x_ch = self.x_coords.expand(batch_size, -1, -1).unsqueeze(1)  # (B, 1, H, W)
        y_ch = self.y_coords.expand(batch_size, -1, -1).unsqueeze(1)  # (B, 1, H, W)
        grid = torch.cat([grid, x_ch, y_ch], dim=1)  # (B, 3, 16, 20)

        # CNN spatial encoder
        h = F.relu(self.conv1(grid))     # (B, 16, 16, 20)
        h = self.pool1(h)                 # (B, 16, 8, 10)
        h = F.relu(self.conv2(h))         # (B, 32, 8, 10)
        h = h.reshape(batch_size, -1)     # (B, 2560)

        # Concat with non-spatial features
        h = torch.cat([h, non_spatial], dim=-1)  # (B, 2560+58)

        # Policy MLP
        h = torch.tanh(self.fc1(h))
        h = torch.tanh(self.fc2(h))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


class ICMNetwork(nn.Module):
    """
    Intrinsic Curiosity Module — feature encoder + forward + inverse models.

    All in one module so autograd handles everything.
    """

    def __init__(self, obs_dim: int, n_actions: int,
                 feature_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.n_actions = n_actions
        self.feature_dim = feature_dim

        # Feature encoder: obs → features
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

        # Forward model: (features + action_onehot) → predicted next features
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Inverse model: (features_t + features_tp1) → action logits
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs_t, action, obs_tp1):
        """
        Args:
            obs_t: (batch, obs_dim)
            action: (batch,) int64
            obs_tp1: (batch, obs_dim)

        Returns:
            forward_loss: scalar — prediction error (= intrinsic reward signal)
            inverse_loss: scalar — action prediction error (regularizer)
            intrinsic_reward: (batch,) — per-sample curiosity reward
        """
        # Encode both states
        feat_t = self.encoder(obs_t)
        feat_tp1 = self.encoder(obs_tp1)

        # Forward model: predict next features
        action_oh = F.one_hot(action, self.n_actions).float()
        fwd_input = torch.cat([feat_t, action_oh], dim=-1)
        pred_feat_tp1 = self.forward_model(fwd_input)

        # Forward loss = MSE per sample
        fwd_error = 0.5 * (pred_feat_tp1 - feat_tp1.detach()).pow(2).mean(dim=-1)
        forward_loss = fwd_error.mean()

        # Inverse model: predict action from feature pairs
        inv_input = torch.cat([feat_t, feat_tp1], dim=-1)
        inv_logits = self.inverse_model(inv_input)
        inverse_loss = F.cross_entropy(inv_logits, action)

        # Intrinsic reward = forward prediction error (detached)
        intrinsic_reward = fwd_error.detach()

        return forward_loss, inverse_loss, intrinsic_reward


# ═══════════════════════════════════════════════════════════════
# PPO-ICM AGENT
# ═══════════════════════════════════════════════════════════════

class MarioTorchAgent:
    """
    PyTorch PPO-ICM agent with GPU support.

    Drop-in replacement for MarioICMAgent with same interface.
    """

    def __init__(
        self,
        obs_dim: int = 378,
        n_actions: int = 6,
        hidden1: int = 128,
        hidden2: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        rollout_length: int = 128,
        # ICM params
        icm_feature_dim: int = 32,
        icm_hidden_dim: int = 64,
        icm_lr: float = 1e-3,
        intrinsic_lambda: float = 0.3,
        device: Optional[str] = None,
    ):
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.intrinsic_lambda = intrinsic_lambda

        # Networks
        self.policy = PolicyNetwork(obs_dim, n_actions, hidden1, hidden2).to(self.device)
        self.icm = ICMNetwork(obs_dim, n_actions, icm_feature_dim, icm_hidden_dim).to(self.device)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=icm_lr)

        # Intrinsic reward normalization
        self._int_reward_mean = 0.0
        self._int_reward_var = 1.0
        self._int_reward_count = 0

        # Rollout buffer (stored as lists, converted to tensors at update time)
        self._obs_buf = []
        self._act_buf = []
        self._logp_buf = []
        self._val_buf = []
        self._rew_buf = []
        self._done_buf = []
        self._next_obs_buf = []

        self._last_obs = None
        self._last_act = None
        self._last_logp = None
        self._last_val = None

        # Stats
        self.total_intrinsic_reward = 0.0
        self.total_extrinsic_reward = 0.0

        print(f"  [TorchAgent] Device: {self.device}, "
              f"Policy params: {sum(p.numel() for p in self.policy.parameters()):,}, "
              f"ICM params: {sum(p.numel() for p in self.icm.parameters()):,}")

    @torch.no_grad()
    def step(self, obs: np.ndarray) -> int:
        """Choose action given observation (numpy in, int out)."""
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits, value = self.policy(obs_t)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self._last_obs = obs
        self._last_act = action.item()
        self._last_logp = log_prob.item()
        self._last_val = value.item()

        return self._last_act

    def learn(self, reward: float, done: bool, next_obs: np.ndarray):
        """Store transition and update when rollout is full."""
        if self._last_obs is None:
            return

        self._obs_buf.append(self._last_obs)
        self._act_buf.append(self._last_act)
        self._logp_buf.append(self._last_logp)
        self._val_buf.append(self._last_val)
        self._rew_buf.append(reward)
        self._done_buf.append(done)
        self._next_obs_buf.append(next_obs)

        self.total_extrinsic_reward += reward

        if len(self._obs_buf) >= self.rollout_length or done:
            self._update()
            self._clear_buffer()

    # Alias for compatibility with MarioICMAgent
    def learn_with_next_obs(self, reward: float, done: bool, next_obs: np.ndarray):
        self.learn(reward, done, next_obs)

    def reset(self):
        """Reset episode state."""
        self._last_obs = None
        self._last_act = None
        self._last_logp = None
        self._last_val = None
        self.total_intrinsic_reward = 0.0
        self.total_extrinsic_reward = 0.0

    def _clear_buffer(self):
        self._obs_buf.clear()
        self._act_buf.clear()
        self._logp_buf.clear()
        self._val_buf.clear()
        self._rew_buf.clear()
        self._done_buf.clear()
        self._next_obs_buf.clear()

    def _update(self):
        """Full PPO + ICM update on the rollout buffer."""
        n = len(self._obs_buf)
        if n < 2:
            return

        # Convert to tensors (one GPU transfer)
        obs = torch.from_numpy(np.array(self._obs_buf)).float().to(self.device)
        next_obs = torch.from_numpy(np.array(self._next_obs_buf)).float().to(self.device)
        actions = torch.tensor(self._act_buf, dtype=torch.long, device=self.device)
        old_logps = torch.tensor(self._logp_buf, dtype=torch.float32, device=self.device)
        ext_rewards = torch.tensor(self._rew_buf, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self._done_buf, dtype=torch.float32, device=self.device)
        old_vals = torch.tensor(self._val_buf, dtype=torch.float32, device=self.device)

        # ── ICM: compute intrinsic rewards ────────────────
        with torch.no_grad():
            _, _, intrinsic_rewards = self.icm(obs, actions, next_obs)

        # Normalize intrinsic rewards
        self._int_reward_count += n
        batch_mean = intrinsic_rewards.mean().item()
        batch_var = intrinsic_rewards.var().item()
        self._int_reward_mean = (0.99 * self._int_reward_mean + 0.01 * batch_mean)
        self._int_reward_var = (0.99 * self._int_reward_var + 0.01 * batch_var)
        std = max(self._int_reward_var ** 0.5, 1e-8)
        norm_int_rewards = ((intrinsic_rewards - self._int_reward_mean) / std).clamp(-5, 5)

        self.total_intrinsic_reward += intrinsic_rewards.sum().item()

        # Combined rewards
        rewards = ext_rewards + self.intrinsic_lambda * norm_int_rewards

        # ── GAE advantage estimation ──────────────────────
        with torch.no_grad():
            _, next_val = self.policy(next_obs[-1:])
            next_val = next_val.item() * (1 - dones[-1].item())

            advantages = torch.zeros(n, device=self.device)
            gae = 0.0
            for t in reversed(range(n)):
                if t == n - 1:
                    nv = next_val
                else:
                    nv = old_vals[t + 1].item()
                mask = 1.0 - dones[t].item()
                delta = rewards[t].item() + self.gamma * nv * mask - old_vals[t].item()
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[t] = gae

            returns = advantages + old_vals
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO epochs (batched!) ─────────────────────────
        for _ in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                mb_obs = obs[idx]
                mb_next_obs = next_obs[idx]
                mb_acts = actions[idx]
                mb_old_logps = old_logps[idx]
                mb_returns = returns[idx]
                mb_advs = advantages[idx]

                # ── Policy update ─────────────────────────
                logits, values = self.policy(mb_obs)
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                # Clipped ratio
                ratio = (new_logps - mb_old_logps).exp()
                surr1 = ratio * mb_advs
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns)

                # Total policy loss
                total_policy_loss = (policy_loss
                                     + self.value_coef * value_loss
                                     - self.entropy_coef * entropy)

                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()

                # ── ICM update ────────────────────────────
                fwd_loss, inv_loss, _ = self.icm(mb_obs, mb_acts, mb_next_obs)
                icm_loss = fwd_loss + inv_loss

                self.icm_optimizer.zero_grad()
                icm_loss.backward()
                nn.utils.clip_grad_norm_(self.icm.parameters(), 0.5)
                self.icm_optimizer.step()

    # ═══════════════════════════════════════════════════════════
    # SAVE / LOAD (compatible with numpy agent checkpoint format)
    # ═══════════════════════════════════════════════════════════

    def save(self, path: str):
        """Save model to .pt file."""
        torch.save({
            "policy": self.policy.state_dict(),
            "icm": self.icm.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "icm_optimizer": self.icm_optimizer.state_dict(),
            "config": {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "gamma": self.gamma,
                "intrinsic_lambda": self.intrinsic_lambda,
            },
        }, path)
        print(f"  [TorchAgent] Saved to {path}")

    def _expand_state_dict(self, state_dict: dict, model: torch.nn.Module,
                           name: str) -> dict:
        """
        Copy checkpoint weights into model, expanding any mismatched layers.

        For each parameter where checkpoint shape != model shape:
          - Copy old values into the matching subregion
          - Leave new rows/cols at the model's current (random) init values
        """
        model_params = dict(model.named_parameters())
        result = {}
        for key, ckpt_val in state_dict.items():
            if key not in model_params:
                result[key] = ckpt_val
                continue
            model_val = model_params[key]
            if ckpt_val.shape == model_val.shape:
                result[key] = ckpt_val
            else:
                # Expand: start from current (random) model weights
                new_tensor = model_val.data.clone()
                # Copy old values into the leading subregion
                slices = tuple(slice(0, s) for s in ckpt_val.shape)
                new_tensor[slices] = ckpt_val
                result[key] = new_tensor
                print(f"  [TorchAgent] Expanded {name}.{key}: "
                      f"{list(ckpt_val.shape)} -> {list(model_val.shape)}")
        return result

    def load(self, path: str):
        """Load model from .pt file, handling action count mismatches."""
        data = torch.load(path, map_location=self.device, weights_only=False)

        # Skip CoordConv buffers (recreated correctly in __init__)
        policy_state = {k: v for k, v in data["policy"].items()
                        if k not in ("x_coords", "y_coords")}

        policy_state = self._expand_state_dict(policy_state, self.policy, "policy")
        self.policy.load_state_dict(policy_state, strict=False)

        icm_state = self._expand_state_dict(data["icm"], self.icm, "icm")
        self.icm.load_state_dict(icm_state, strict=False)

        if "policy_optimizer" in data:
            self.policy_optimizer.load_state_dict(data["policy_optimizer"])
        if "icm_optimizer" in data:
            self.icm_optimizer.load_state_dict(data["icm_optimizer"])
        print(f"  [TorchAgent] Loaded from {path}")
