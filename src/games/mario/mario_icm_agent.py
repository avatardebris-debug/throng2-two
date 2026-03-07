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
import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# NUMPY MLP UTILITIES
# ═══════════════════════════════════════════════════════════════

def _init_weight(rows: int, cols: int) -> np.ndarray:
    """Xavier uniform initialization."""
    limit = np.sqrt(6.0 / (rows + cols))
    return np.random.uniform(-limit, limit, (rows, cols)).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-8)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# ADAM OPTIMIZER (per-parameter)
# ═══════════════════════════════════════════════════════════════

class AdamParam:
    """Adam optimizer state for a single parameter."""
    __slots__ = ['m', 'v', 't']

    def __init__(self, shape):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def update(self, param: np.ndarray, grad: np.ndarray,
               lr: float = 3e-4, beta1: float = 0.9,
               beta2: float = 0.999, eps: float = 1e-8) -> np.ndarray:
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * grad ** 2
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)


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

class MarioICMAgent:
    """
    PPO + ICM agent for Mario ASCII. Pure numpy, no PyTorch.

    Extends MarioRLAgent with intrinsic curiosity rewards.
    Drop-in replacement: same step()/learn()/reset() interface.

    The agent receives two reward signals:
      r_total = r_extrinsic + intrinsic_lambda * r_curiosity

    Where r_curiosity = forward model prediction error from ICM.
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
        # ICM params
        icm_feature_dim: int = 32,
        icm_hidden_dim: int = 64,
        icm_lr: float = 1e-3,
        intrinsic_lambda: float = 0.5,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.intrinsic_lambda = intrinsic_lambda

        # ── PPO policy network ────────────────────────────
        self.w1 = _init_weight(obs_dim, hidden1)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.w2 = _init_weight(hidden1, hidden2)
        self.b2 = np.zeros(hidden2, dtype=np.float32)
        self.w_pi = _init_weight(hidden2, n_actions)
        self.b_pi = np.zeros(n_actions, dtype=np.float32)
        self.w_v = _init_weight(hidden2, 1)
        self.b_v = np.zeros(1, dtype=np.float32)

        # Adam for policy
        self._adam_policy = {}
        for name in ['w1', 'b1', 'w2', 'b2', 'w_pi', 'b_pi', 'w_v', 'b_v']:
            self._adam_policy[name] = AdamParam(getattr(self, name).shape)
        self._lr = lr

        # ── ICM module ────────────────────────────────────
        self.icm = ICMModule(
            obs_dim=obs_dim,
            n_actions=n_actions,
            feature_dim=icm_feature_dim,
            hidden_dim=icm_hidden_dim,
            lr=icm_lr,
        )

        # ── Rollout buffer ────────────────────────────────
        self._obs_buf = []
        self._act_buf = []
        self._rew_buf = []      # extrinsic
        self._int_rew_buf = []  # intrinsic
        self._logp_buf = []
        self._val_buf = []
        self._done_buf = []
        self._next_obs_buf = []

        self._last_obs = None
        self._last_act = None
        self._last_logp = None
        self._last_val = None
        self._step_count = 0

        # Stats
        self.total_intrinsic_reward = 0.0
        self.total_extrinsic_reward = 0.0

    def _forward(self, obs: np.ndarray):
        """Forward pass through policy network."""
        h1 = _tanh(obs @ self.w1 + self.b1)
        h2 = _tanh(h1 @ self.w2 + self.b2)
        logits = h2 @ self.w_pi + self.b_pi
        value = (h2 @ self.w_v + self.b_v)[0]
        return logits, value, h1, h2

    def step(self, obs: np.ndarray) -> int:
        """Choose action given observation."""
        logits, value, _, _ = self._forward(obs)
        probs = _softmax(logits)
        action = np.random.choice(self.n_actions, p=probs)
        log_prob = np.log(probs[action] + 1e-8)

        self._last_obs = obs
        self._last_act = action
        self._last_logp = log_prob
        self._last_val = value

        return action

    def learn(self, reward: float, done: bool):
        """Store transition and potentially do PPO update."""
        if self._last_obs is None:
            return

        self._obs_buf.append(self._last_obs)
        self._act_buf.append(self._last_act)
        self._rew_buf.append(reward)
        self._logp_buf.append(self._last_logp)
        self._val_buf.append(self._last_val)
        self._done_buf.append(done)
        self._step_count += 1

        self.total_extrinsic_reward += reward

        if len(self._obs_buf) >= self.rollout_length or done:
            self._ppo_update()
            self._obs_buf.clear()
            self._act_buf.clear()
            self._rew_buf.clear()
            self._int_rew_buf.clear()
            self._logp_buf.clear()
            self._val_buf.clear()
            self._done_buf.clear()
            self._next_obs_buf.clear()

    def learn_with_next_obs(self, reward: float, done: bool, next_obs: np.ndarray):
        """
        Learn with ICM: stores transition with next_obs for curiosity computation.
        Call this instead of learn() when you have the next observation.
        """
        if self._last_obs is None:
            return

        # Compute intrinsic reward
        intrinsic_r = self.icm.compute_intrinsic_reward(
            self._last_obs, self._last_act, next_obs
        )
        self.total_intrinsic_reward += intrinsic_r

        self._obs_buf.append(self._last_obs)
        self._act_buf.append(self._last_act)
        # Combined reward
        combined_reward = reward + self.intrinsic_lambda * intrinsic_r
        self._rew_buf.append(combined_reward)
        self._int_rew_buf.append(intrinsic_r)
        self._logp_buf.append(self._last_logp)
        self._val_buf.append(self._last_val)
        self._done_buf.append(done)
        self._next_obs_buf.append(next_obs)
        self._step_count += 1

        self.total_extrinsic_reward += reward

        if len(self._obs_buf) >= self.rollout_length or done:
            self._ppo_update()
            self._obs_buf.clear()
            self._act_buf.clear()
            self._rew_buf.clear()
            self._int_rew_buf.clear()
            self._logp_buf.clear()
            self._val_buf.clear()
            self._done_buf.clear()
            self._next_obs_buf.clear()

    def reset(self):
        """Reset episode state."""
        self._last_obs = None
        self._last_act = None
        self._last_logp = None
        self._last_val = None
        self.total_intrinsic_reward = 0.0
        self.total_extrinsic_reward = 0.0

    def _ppo_update(self):
        """PPO update with GAE."""
        n = len(self._obs_buf)
        if n < 2:
            return

        obs = np.array(self._obs_buf)
        acts = np.array(self._act_buf)
        rews = np.array(self._rew_buf)
        old_logps = np.array(self._logp_buf)
        vals = np.array(self._val_buf)
        dones = np.array(self._done_buf)

        # Bootstrap value
        if dones[-1]:
            next_val = 0.0
        else:
            _, next_val, _, _ = self._forward(obs[-1])

        # GAE
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        lam = 0.95
        for t in reversed(range(n)):
            if t == n - 1:
                next_v = next_val
            else:
                next_v = vals[t + 1]
            mask = 1.0 - float(dones[t])
            delta = rews[t] + self.gamma * next_v * mask - vals[t]
            gae = delta + self.gamma * lam * mask * gae
            advantages[t] = gae

        returns = advantages + vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        clip_eps = 0.2
        for _ in range(4):
            indices = np.random.permutation(n)
            for start in range(0, n, 32):
                end = min(start + 32, n)
                idx = indices[start:end]
                mb_obs = obs[idx]
                mb_acts = acts[idx]
                mb_old_logps = old_logps[idx]
                mb_returns = returns[idx]
                mb_advs = advantages[idx]

                # Forward
                for i in range(len(idx)):
                    logits, value, h1, h2 = self._forward(mb_obs[i])
                    probs = _softmax(logits)
                    a = mb_acts[i]
                    log_prob = np.log(probs[a] + 1e-8)

                    # Ratio
                    ratio = np.exp(log_prob - mb_old_logps[i])
                    adv = mb_advs[i]

                    # Clipped surrogate
                    surr1 = ratio * adv
                    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                    policy_loss = -min(surr1, surr2)

                    # Value loss
                    value_loss = 0.5 * (value - mb_returns[i]) ** 2

                    # Entropy bonus
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    entropy_loss = -0.01 * entropy

                    # Total loss gradient (simplified single-sample)
                    # Policy gradient
                    d_logits = probs.copy()
                    d_logits[a] -= 1.0
                    scale = -adv * min(1.0, max(-1.0, ratio))
                    d_logits *= scale

                    # Value gradient
                    d_value = value - mb_returns[i]

                    # Backprop through network
                    d_w_pi = np.outer(h2, d_logits)
                    d_b_pi = d_logits
                    d_w_v = h2.reshape(-1, 1) * d_value
                    d_b_v = np.array([d_value], dtype=np.float32)

                    d_h2 = d_logits @ self.w_pi.T + d_value * self.w_v.squeeze()
                    d_h2 *= (1 - h2 ** 2)  # tanh grad
                    d_w2 = np.outer(h1, d_h2)
                    d_b2 = d_h2

                    d_h1 = d_h2 @ self.w2.T
                    d_h1 *= (1 - h1 ** 2)
                    d_w1 = np.outer(mb_obs[i], d_h1)
                    d_b1 = d_h1

                    # Update
                    lr = self._lr
                    self.w_pi = self._adam_policy['w_pi'].update(self.w_pi, d_w_pi, lr)
                    self.b_pi = self._adam_policy['b_pi'].update(self.b_pi, d_b_pi, lr)
                    self.w_v = self._adam_policy['w_v'].update(self.w_v, d_w_v, lr)
                    self.b_v = self._adam_policy['b_v'].update(self.b_v, d_b_v, lr)
                    self.w2 = self._adam_policy['w2'].update(self.w2, d_w2, lr)
                    self.b2 = self._adam_policy['b2'].update(self.b2, d_b2, lr)
                    self.w1 = self._adam_policy['w1'].update(self.w1, d_w1, lr)
                    self.b1 = self._adam_policy['b1'].update(self.b1, d_b1, lr)
