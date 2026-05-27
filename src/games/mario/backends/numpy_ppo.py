"""
mario_agent.py -- Lightweight pure-numpy RL agent for Mario ASCII.

Compatible with Throng pattern but no PyTorch dependency.
Implements a policy-gradient agent with:
  - MLP policy network (obs -> action probabilities)
  - Value head (obs -> value estimate)
  - GAE advantage estimation
  - Rollout buffer
  - Adam optimizer (pure numpy)

Designed to match ThrongletCell interface:
  agent.step(obs) -> action
  agent.learn(reward, done) -> stats
  agent.reset()

When PyTorch is available, swap this for ThrongletCell seamlessly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .numpy_common import AdamParam, dtanh, softmax, tanh

# Local aliases (historical names in this module)
_tanh = tanh
_dtanh = dtanh
_softmax = softmax


# ── MLP Network ──────────────────────────────────────────────

class NumpyMLP:
    """
    Simple 2-layer MLP with shared backbone + separate heads.

    Architecture (matches ThrongletCell ActorCritic):
      obs -> 128(tanh) -> 64(tanh) -> policy_logits(n_actions) + value(1)
    """

    def __init__(self, input_dim: int, n_actions: int, hidden1: int = 128, hidden2: int = 64, lr: float = 3e-4):
        self.input_dim = input_dim
        self.n_actions = n_actions

        # Shared backbone
        self.W1 = np.random.randn(hidden1, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = np.random.randn(hidden2, hidden1).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)

        # Policy head
        self.Wp = np.random.randn(n_actions, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.bp = np.zeros(n_actions, dtype=np.float32)

        # Value head
        self.Wv = np.random.randn(1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.bv = np.zeros(1, dtype=np.float32)

        # Adam optimizers
        self._adam = {}
        for name in ["W1", "b1", "W2", "b2", "Wp", "bp", "Wv", "bv"]:
            self._adam[name] = AdamParam(getattr(self, name).shape, lr=lr)

    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, float, dict]:
        """
        Single-sample forward pass.
        Returns: (action_probs, value, cache for backprop)
        """
        h1_pre = self.W1 @ obs + self.b1
        h1 = _tanh(h1_pre)
        h2_pre = self.W2 @ h1 + self.b2
        h2 = _tanh(h2_pre)

        logits = self.Wp @ h2 + self.bp
        probs = _softmax(logits)
        value = float((self.Wv @ h2 + self.bv)[0])

        cache = {"obs": obs, "h1_pre": h1_pre, "h1": h1,
                 "h2_pre": h2_pre, "h2": h2, "logits": logits, "probs": probs}
        return probs, value, cache

    def forward_batch(self, obs_batch: np.ndarray):
        """
        Batched forward pass: (N, obs_dim) → (N, n_actions) probs, (N,) values.
        Uses BLAS matmul for ~100x speedup over looping.
        """
        # obs_batch: (N, input_dim)
        h1_pre = obs_batch @ self.W1.T + self.b1   # (N, 128)
        h1 = _tanh(h1_pre)
        h2_pre = h1 @ self.W2.T + self.b2           # (N, 64)
        h2 = _tanh(h2_pre)

        logits = h2 @ self.Wp.T + self.bp            # (N, n_actions)
        # Per-row softmax
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / (e.sum(axis=-1, keepdims=True) + 1e-10)

        values = (h2 @ self.Wv.T + self.bv).ravel()  # (N,)

        cache = {"obs": obs_batch, "h1_pre": h1_pre, "h1": h1,
                 "h2_pre": h2_pre, "h2": h2, "logits": logits, "probs": probs}
        return probs, values, cache

    def backward_batch(self, d_logits: np.ndarray, d_values: np.ndarray, cache: dict):
        """
        Batched backward pass: accumulate gradients across minibatch.
        d_logits: (N, n_actions)   d_values: (N,)
        """
        obs = cache["obs"]      # (N, input_dim)
        h1 = cache["h1"]        # (N, 128)
        h2 = cache["h2"]        # (N, 64)
        h1_pre = cache["h1_pre"]
        h2_pre = cache["h2_pre"]
        N = obs.shape[0]

        # Policy head grads
        dWp = d_logits.T @ h2 / N        # (n_actions, 64)
        dbp = d_logits.mean(axis=0)       # (n_actions,)

        # Value head grads
        dv = d_values.reshape(N, 1)       # (N, 1)
        dWv = dv.T @ h2 / N              # (1, 64)
        dbv = dv.mean(axis=0)             # (1,)

        # Combined dh2 from both heads
        dh2_policy = d_logits @ self.Wp   # (N, 64)
        dh2_value  = dv @ self.Wv         # (N, 64)
        dh2 = (dh2_policy + dh2_value) * _dtanh(h2_pre)  # (N, 64)

        dW2 = dh2.T @ h1 / N             # (64, 128)
        db2 = dh2.mean(axis=0)            # (64,)

        dh1 = (dh2 @ self.W2) * _dtanh(h1_pre)  # (N, 128)
        dW1 = dh1.T @ obs / N            # (128, input_dim)
        db1 = dh1.mean(axis=0)            # (128,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "Wp": dWp, "bp": dbp, "Wv": dWv, "bv": dbv}

    def backward_policy(self, d_logits: np.ndarray, cache: dict) -> dict:
        """Single-sample backprop (legacy, used by step-learn)."""
        dWp = np.outer(d_logits, cache["h2"])
        dbp = d_logits
        dh2 = self.Wp.T @ d_logits
        dh2 *= _dtanh(cache["h2_pre"])
        dW2 = np.outer(dh2, cache["h1"])
        db2 = dh2
        dh1 = self.W2.T @ dh2
        dh1 *= _dtanh(cache["h1_pre"])
        dW1 = np.outer(dh1, cache["obs"])
        db1 = dh1
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "Wp": dWp, "bp": dbp}

    def backward_value(self, d_value: float, cache: dict) -> dict:
        """Single-sample value backprop (legacy)."""
        dv = np.array([d_value], dtype=np.float32)
        dWv = np.outer(dv, cache["h2"])
        dbv = dv
        dh2 = self.Wv.T @ dv
        dh2 = dh2.flatten() * _dtanh(cache["h2_pre"])
        dW2 = np.outer(dh2, cache["h1"])
        db2 = dh2
        dh1 = self.W2.T @ dh2
        dh1 *= _dtanh(cache["h1_pre"])
        dW1 = np.outer(dh1, cache["obs"])
        db1 = dh1
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "Wv": dWv, "bv": dbv}

    def update(self, grads: dict):
        """Apply gradient update via Adam."""
        for name, grad in grads.items():
            param = getattr(self, name)
            self._adam[name].step(param, grad)


# ── RL Agent ─────────────────────────────────────────────────

class MarioRLAgent:
    """
    Lightweight policy-gradient agent for Mario ASCII.

    ThrongletCell-compatible interface:
      agent.step(obs) -> action
      agent.learn(reward, done) -> stats
      agent.reset()
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden1: int = 128,
        hidden2: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rollout_length: int = 128,
        update_epochs: int = 4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_length = rollout_length
        self.update_epochs = update_epochs
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Network
        self.net = NumpyMLP(obs_dim, n_actions, hidden1, hidden2, lr)

        # Observation normalization
        self._obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self._obs_var = np.ones(obs_dim, dtype=np.float32)
        self._obs_count = 0

        # Rollout buffer
        self._buf_obs: List[np.ndarray] = []
        self._buf_actions: List[int] = []
        self._buf_log_probs: List[float] = []
        self._buf_values: List[float] = []
        self._buf_rewards: List[float] = []
        self._buf_dones: List[bool] = []
        self._buf_caches: List[dict] = []

        # State tracking
        self._last_obs = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None
        self._last_cache = None

        # Stats
        self._total_steps = 0
        self._total_episodes = 0
        self._update_count = 0
        self._episode_reward = 0.0
        self._reward_history: List[float] = []

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count
        std = np.sqrt(np.maximum(self._obs_var, 1e-8))
        return ((obs - self._obs_mean) / std).astype(np.float32)

    def step(self, obs: np.ndarray) -> int:
        """Select action given observation."""
        norm_obs = self._normalize_obs(obs)
        probs, value, cache = self.net.forward(norm_obs)

        # Sample action
        action = int(np.random.choice(self.n_actions, p=probs))
        log_prob = float(np.log(probs[action] + 1e-10))

        self._last_obs = norm_obs
        self._last_action = action
        self._last_log_prob = log_prob
        self._last_value = value
        self._last_cache = cache
        self._total_steps += 1

        return action

    def learn_with_next_obs(
        self, reward: float, done: bool, next_obs: np.ndarray
    ) -> Optional[dict]:
        """ICM-free path: next_obs ignored; use learn(reward, done)."""
        return self.learn(reward, done)

    def learn(self, reward: float, done: bool) -> Optional[dict]:
        """Process reward and optionally update policy."""
        if self._last_obs is None:
            return None

        # Store transition
        self._buf_obs.append(self._last_obs)
        self._buf_actions.append(self._last_action)
        self._buf_log_probs.append(self._last_log_prob)
        self._buf_values.append(self._last_value)
        self._buf_rewards.append(reward)
        self._buf_dones.append(done)
        self._buf_caches.append(self._last_cache)

        self._episode_reward += reward

        if done:
            self._total_episodes += 1
            self._reward_history.append(self._episode_reward)
            self._episode_reward = 0.0

        # Update when buffer is full
        if len(self._buf_obs) >= self.rollout_length:
            last_val = 0.0
            if not done and self._last_obs is not None:
                _, last_val, _ = self.net.forward(self._last_obs)
            stats = self._update(last_val)
            self._clear_buffer()
            return stats

        return None

    def reset(self):
        """Reset between episodes."""
        self._last_obs = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None
        self._last_cache = None

    def _update(self, last_value: float) -> dict:
        """Batched PPO update — all matmuls over the minibatch at once."""
        n = len(self._buf_obs)
        if n == 0:
            return {}

        # Stack buffers into arrays (one allocation)
        obs_all = np.array(self._buf_obs, dtype=np.float32)        # (N, obs_dim)
        actions_all = np.array(self._buf_actions, dtype=np.int64)  # (N,)
        rewards = np.array(self._buf_rewards, dtype=np.float32)
        values = np.array(self._buf_values, dtype=np.float32)
        dones = np.array(self._buf_dones, dtype=np.float32)
        old_log_probs = np.array(self._buf_log_probs, dtype=np.float32)

        # GAE (sequential — can't batch this)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
        returns = advantages + values

        # Normalize advantages
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-batch PPO updates (BATCHED — one matmul per minibatch)
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        for epoch in range(self.update_epochs):
            indices = np.random.permutation(n)
            batch_size = min(128, n)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]
                mb = len(idx)

                mb_obs = obs_all[idx]            # (mb, obs_dim)
                mb_acts = actions_all[idx]       # (mb,)
                mb_old_lp = old_log_probs[idx]   # (mb,)
                mb_adv = advantages[idx]         # (mb,)
                mb_ret = returns[idx]            # (mb,)

                # Batched forward
                probs, vals, cache = self.net.forward_batch(mb_obs)

                # New log probs (gather action probs)
                new_lp = np.log(probs[np.arange(mb), mb_acts] + 1e-10)  # (mb,)

                # PPO clipped ratios
                ratio = np.exp(new_lp - mb_old_lp)                     # (mb,)
                clipped = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                surr1 = ratio * mb_adv
                surr2 = clipped * mb_adv
                pg_loss = -np.minimum(surr1, surr2).mean()

                # Value loss
                v_loss = ((vals - mb_ret) ** 2).mean()

                # Entropy
                entropy = -(probs * np.log(probs + 1e-10)).sum(axis=-1).mean()

                # Build policy gradient: d_logits (mb, n_actions)
                # For unclipped samples, gradient is -adv*ratio at the action index
                use_unclipped = (surr1 <= surr2).astype(np.float32)  # (mb,)
                d_logits = np.zeros((mb, self.n_actions), dtype=np.float32)
                d_logits[np.arange(mb), mb_acts] = -mb_adv * ratio * use_unclipped

                # Entropy gradient
                d_logits += self.entropy_coef * (np.log(probs + 1e-10) + 1)

                # Value gradient
                d_values = (self.value_coef * 2 * (vals - mb_ret)).astype(np.float32)

                # Batched backward (one BLAS call for entire minibatch)
                grads = self.net.backward_batch(d_logits, d_values, cache)
                self.net.update(grads)

                total_pg_loss += pg_loss
                total_v_loss += v_loss
                total_entropy += entropy
                n_batches += 1

        self._update_count += 1
        return {
            "policy_loss": round(float(total_pg_loss / max(1, n_batches)), 4),
            "value_loss": round(float(total_v_loss / max(1, n_batches)), 4),
            "entropy": round(float(total_entropy / max(1, n_batches)), 4),
            "n_updates": n_batches,
        }

    def _clear_buffer(self):
        self._buf_obs.clear()
        self._buf_actions.clear()
        self._buf_log_probs.clear()
        self._buf_values.clear()
        self._buf_rewards.clear()
        self._buf_dones.clear()
        self._buf_caches.clear()

    @property
    def neuron_count(self):
        return 0  # No SNN

    def stats(self) -> dict:
        avg_100 = float(np.mean(self._reward_history[-100:])) if self._reward_history else 0.0
        return {
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "update_count": self._update_count,
            "avg_reward_last_100": round(avg_100, 2),
            "reward_history_len": len(self._reward_history),
        }
