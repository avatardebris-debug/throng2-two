"""
mario_icm_agent.py -- PPO + Intrinsic Curiosity Module (ICM) for Mario ASCII.

Pure-numpy implementation (no PyTorch). Based on:
  "Curiosity-driven Exploration by Self-supervised Prediction"
  (Pathak et al., 2017, https://arxiv.org/abs/1705.05363)

ICM gives the agent intrinsic reward for visiting "surprising" states,
which solves the "never discovers jumping" problem. Three sub-networks:

  1. Feature encoder:  obs(378)    → features(32)
  2. Forward model:    (feat, act) → pred_next_feat    [prediction error = curiosity]
  3. Inverse model:    (feat, feat')  → pred_action     [regularizer]

The intrinsic reward is the forward model prediction error (MSE).
Combined with extrinsic reward: r_total = r_ext + λ * r_intrinsic

ThrongletCell-compatible interface: step(obs) → action, learn(reward, done).
Drop-in replacement for MarioRLAgent with curiosity bonus.

Usage:
    agent = MarioICMAgent(obs_dim=378, n_actions=6)
    action = agent.step(obs)
    agent.learn(reward, done)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.games.mario.backends.numpy_common import (
    AdamParam,
    init_weight,
    relu,
    relu_grad,
    softmax,
)

_init_weight = init_weight
_relu = relu
_relu_grad = relu_grad
_softmax = softmax


# ═══════════════════════════════════════════════════════════════
# ICM MODULE (Intrinsic Curiosity Module)
# ═══════════════════════════════════════════════════════════════

class ICMModule:
    """
    Intrinsic Curiosity Module — pure numpy.

    Components:
      - Feature encoder: obs → features (compresses state)
      - Forward model: (features_t, action_one_hot) → predicted features_t+1
      - Inverse model: (features_t, features_t+1) → predicted action logits

    Intrinsic reward = MSE(predicted_features_t+1, actual_features_t+1)
    """

    def __init__(
        self,
        obs_dim: int = 378,
        n_actions: int = 6,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        lr: float = 1e-3,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Feature encoder: obs → hidden → features
        self.enc_w1 = _init_weight(obs_dim, hidden_dim)
        self.enc_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.enc_w2 = _init_weight(hidden_dim, feature_dim)
        self.enc_b2 = np.zeros(feature_dim, dtype=np.float32)

        # Forward model: (features + action_one_hot) → hidden → pred_features
        fwd_in = feature_dim + n_actions
        self.fwd_w1 = _init_weight(fwd_in, hidden_dim)
        self.fwd_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.fwd_w2 = _init_weight(hidden_dim, feature_dim)
        self.fwd_b2 = np.zeros(feature_dim, dtype=np.float32)

        # Inverse model: (features_t + features_t+1) → hidden → action_logits
        inv_in = feature_dim * 2
        self.inv_w1 = _init_weight(inv_in, hidden_dim)
        self.inv_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.inv_w2 = _init_weight(hidden_dim, n_actions)
        self.inv_b2 = np.zeros(n_actions, dtype=np.float32)

        # Adam states
        self._adam = {}
        for name in ['enc_w1', 'enc_b1', 'enc_w2', 'enc_b2',
                      'fwd_w1', 'fwd_b1', 'fwd_w2', 'fwd_b2',
                      'inv_w1', 'inv_b1', 'inv_w2', 'inv_b2']:
            self._adam[name] = AdamParam(getattr(self, name).shape)

        # Running statistics for reward normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode observation to feature space."""
        h = _relu(obs @ self.enc_w1 + self.enc_b1)
        return _relu(h @ self.enc_w2 + self.enc_b2)

    def compute_intrinsic_reward(
        self,
        obs_t: np.ndarray,
        action: int,
        obs_tp1: np.ndarray,
    ) -> float:
        """
        Compute curiosity reward = forward model prediction error.

        Also updates the ICM networks (encoder, forward, inverse).

        Returns:
            Normalized intrinsic reward (float)
        """
        # ── Forward pass ──────────────────────────────────
        # Encode both states
        enc_h1_t = _relu(obs_t @ self.enc_w1 + self.enc_b1)
        feat_t = _relu(enc_h1_t @ self.enc_w2 + self.enc_b2)

        enc_h1_tp1 = _relu(obs_tp1 @ self.enc_w1 + self.enc_b1)
        feat_tp1 = _relu(enc_h1_tp1 @ self.enc_w2 + self.enc_b2)

        # Forward model: predict next features
        action_oh = np.zeros(self.n_actions, dtype=np.float32)
        action_oh[action] = 1.0
        fwd_input = np.concatenate([feat_t, action_oh])
        fwd_h = _relu(fwd_input @ self.fwd_w1 + self.fwd_b1)
        pred_feat_tp1 = fwd_h @ self.fwd_w2 + self.fwd_b2

        # Inverse model: predict action from feature pairs
        inv_input = np.concatenate([feat_t, feat_tp1])
        inv_h = _relu(inv_input @ self.inv_w1 + self.inv_b1)
        inv_logits = inv_h @ self.inv_w2 + self.inv_b2

        # ── Compute losses ────────────────────────────────
        # Forward loss (MSE)
        fwd_error = pred_feat_tp1 - feat_tp1
        fwd_loss = 0.5 * np.mean(fwd_error ** 2)

        # Inverse loss (cross-entropy)
        inv_probs = _softmax(inv_logits)
        inv_loss = -np.log(inv_probs[action] + 1e-8)

        # Intrinsic reward = forward prediction error
        raw_reward = fwd_loss

        # ── Backward pass (simplified) ────────────────────
        # Forward model gradients
        d_pred = fwd_error / self.feature_dim  # d(MSE)/d(pred)
        d_fwd_w2 = np.outer(fwd_h, d_pred)
        d_fwd_b2 = d_pred
        d_fwd_h = d_pred @ self.fwd_w2.T * _relu_grad(fwd_h)
        d_fwd_w1 = np.outer(fwd_input, d_fwd_h)
        d_fwd_b1 = d_fwd_h

        # Inverse model gradients
        d_inv_logits = inv_probs.copy()
        d_inv_logits[action] -= 1.0  # softmax cross-entropy gradient
        d_inv_w2 = np.outer(inv_h, d_inv_logits)
        d_inv_b2 = d_inv_logits
        d_inv_h = d_inv_logits @ self.inv_w2.T * _relu_grad(inv_h)
        d_inv_w1 = np.outer(inv_input, d_inv_h)
        d_inv_b1 = d_inv_h

        # Encoder gradients (from both forward and inverse)
        # From forward model: gradient w.r.t. feat_t via fwd_input
        # fwd_w1 shape: (feat_dim + n_actions, hidden_dim)
        # We need gradient for feat_t portion only (first feat_dim rows)
        d_fwd_feat_t = d_fwd_h @ self.fwd_w1[:self.feature_dim].T

        # From inverse model: gradient w.r.t. feat_t via inv_input
        # inv_w1 shape: (feat_dim * 2, hidden_dim)
        # feat_t is first feat_dim elements of inv_input
        d_inv_feat_t = d_inv_h @ self.inv_w1[:self.feature_dim].T

        d_feat_t = d_fwd_feat_t + d_inv_feat_t
        d_feat_t *= _relu_grad(feat_t)
        d_enc_w2_t = np.outer(enc_h1_t, d_feat_t)
        d_enc_b2_t = d_feat_t
        d_enc_h1_t = d_feat_t @ self.enc_w2.T * _relu_grad(enc_h1_t)
        d_enc_w1_t = np.outer(obs_t, d_enc_h1_t)
        d_enc_b1_t = d_enc_h1_t

        # ── Update weights ────────────────────────────────
        self.fwd_w2 = self._adam['fwd_w2'].update(self.fwd_w2, d_fwd_w2, self.lr)
        self.fwd_b2 = self._adam['fwd_b2'].update(self.fwd_b2, d_fwd_b2, self.lr)
        self.fwd_w1 = self._adam['fwd_w1'].update(self.fwd_w1, d_fwd_w1, self.lr)
        self.fwd_b1 = self._adam['fwd_b1'].update(self.fwd_b1, d_fwd_b1, self.lr)

        self.inv_w2 = self._adam['inv_w2'].update(self.inv_w2, d_inv_w2, self.lr)
        self.inv_b2 = self._adam['inv_b2'].update(self.inv_b2, d_inv_b2, self.lr)
        self.inv_w1 = self._adam['inv_w1'].update(self.inv_w1, d_inv_w1, self.lr)
        self.inv_b1 = self._adam['inv_b1'].update(self.inv_b1, d_inv_b1, self.lr)

        self.enc_w2 = self._adam['enc_w2'].update(self.enc_w2, d_enc_w2_t, self.lr)
        self.enc_b2 = self._adam['enc_b2'].update(self.enc_b2, d_enc_b2_t, self.lr)
        self.enc_w1 = self._adam['enc_w1'].update(self.enc_w1, d_enc_w1_t, self.lr)
        self.enc_b1 = self._adam['enc_b1'].update(self.enc_b1, d_enc_b1_t, self.lr)

        # ── Normalize intrinsic reward ────────────────────
        self._reward_count += 1
        delta = raw_reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_var += delta * (raw_reward - self._reward_mean)

        std = np.sqrt(self._reward_var / max(1, self._reward_count)) + 1e-8
        normalized = (raw_reward - self._reward_mean) / std
        return float(np.clip(normalized, -5.0, 5.0))


# ═══════════════════════════════════════════════════════════════
# PPO-ICM AGENT
# ═══════════════════════════════════════════════════════════════

from .backends.numpy_ppo import MarioRLAgent


class MarioICMAgent:
    """
    PPO + ICM for Mario ASCII — composes MarioRLAgent + ICMModule.

    Use learn_with_next_obs() so curiosity can run; learn() is extrinsic-only PPO.
    """

    def __init__(
        self,
        obs_dim: int = 378,
        n_actions: int = 6,
        hidden1: int = 128,
        hidden2: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        rollout_length: int = 128,
        icm_feature_dim: int = 32,
        icm_hidden_dim: int = 64,
        icm_lr: float = 1e-3,
        intrinsic_lambda: float = 0.5,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.intrinsic_lambda = intrinsic_lambda

        self.ppo = MarioRLAgent(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden1=hidden1,
            hidden2=hidden2,
            lr=lr,
            gamma=gamma,
            rollout_length=rollout_length,
        )
        self.icm = ICMModule(
            obs_dim=obs_dim,
            n_actions=n_actions,
            feature_dim=icm_feature_dim,
            hidden_dim=icm_hidden_dim,
            lr=icm_lr,
        )

        self._last_raw_obs: Optional[np.ndarray] = None
        self.total_intrinsic_reward = 0.0
        self.total_extrinsic_reward = 0.0

    def step(self, obs: np.ndarray) -> int:
        self._last_raw_obs = np.asarray(obs, dtype=np.float32)
        return self.ppo.step(obs)

    def learn(self, reward: float, done: bool):
        """Extrinsic reward only (no ICM without next_obs)."""
        if self.ppo._last_obs is None:
            return None
        self.total_extrinsic_reward += reward
        return self.ppo.learn(reward, done)

    def learn_with_next_obs(self, reward: float, done: bool, next_obs: np.ndarray):
        """PPO update with r_total = r_ext + λ * r_intrinsic."""
        if self.ppo._last_obs is None or self._last_raw_obs is None:
            return None

        next_obs = np.asarray(next_obs, dtype=np.float32)
        intrinsic_r = self.icm.compute_intrinsic_reward(
            self._last_raw_obs, int(self.ppo._last_action), next_obs
        )
        self.total_intrinsic_reward += intrinsic_r
        self.total_extrinsic_reward += reward
        combined = reward + self.intrinsic_lambda * intrinsic_r
        return self.ppo.learn(combined, done)

    def reset(self):
        self.ppo.reset()
        self._last_raw_obs = None
        self.total_intrinsic_reward = 0.0
        self.total_extrinsic_reward = 0.0

    @property
    def neuron_count(self):
        return 0

    def stats(self) -> dict:
        out = self.ppo.stats()
        out["total_intrinsic_reward"] = round(self.total_intrinsic_reward, 4)
        out["total_extrinsic_reward"] = round(self.total_extrinsic_reward, 4)
        return out
