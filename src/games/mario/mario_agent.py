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


# ── Activations ──────────────────────────────────────────────

def _tanh(x):
    return np.tanh(np.clip(x, -10, 10))

def _dtanh(x):
    t = _tanh(x)
    return 1.0 - t ** 2

def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / (e.sum() + 1e-10)


# ── Adam Optimizer ───────────────────────────────────────────

class AdamParam:
    """Adam state for a single parameter array."""
    def __init__(self, shape, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, param, grad):
        grad = np.clip(grad, -1.0, 1.0)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param


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
        Forward pass.
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

    def backward_policy(self, d_logits: np.ndarray, cache: dict) -> dict:
        """Backprop through policy head + shared backbone."""
        # Policy head
        dWp = np.outer(d_logits, cache["h2"])
        dbp = d_logits

        # Shared backbone (from policy path)
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
        """Backprop through value head + shared backbone."""
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
        """PPO-style update on collected rollout."""
        n = len(self._buf_obs)
        if n == 0:
            return {}

        rewards = np.array(self._buf_rewards, dtype=np.float32)
        values = np.array(self._buf_values, dtype=np.float32)
        dones = np.array(self._buf_dones, dtype=np.float32)
        old_log_probs = np.array(self._buf_log_probs, dtype=np.float32)

        # GAE
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

        # Mini-batch PPO updates
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.update_epochs):
            indices = np.random.permutation(n)
            batch_size = min(64, n)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                for i in idx:
                    obs = self._buf_obs[i]
                    action = self._buf_actions[i]
                    old_lp = old_log_probs[i]
                    adv = advantages[i]
                    ret = returns[i]

                    # Forward pass
                    probs, value, cache = self.net.forward(obs)
                    new_lp = np.log(probs[action] + 1e-10)

                    # PPO clipped objective
                    ratio = np.exp(new_lp - old_lp)
                    clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    pg_loss = -min(ratio * adv, clipped_ratio * adv)

                    # Value loss
                    v_loss = (value - ret) ** 2

                    # Entropy
                    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

                    # Policy gradient
                    d_logits = np.zeros(self.n_actions, dtype=np.float32)
                    if ratio * adv <= clipped_ratio * adv:
                        # Use unclipped gradient
                        d_logits[action] = -adv * ratio
                    else:
                        pass  # Clipped — no gradient

                    # Add entropy gradient
                    d_logits += self.entropy_coef * (np.log(probs + 1e-10) + 1)

                    # Convert logit gradient through softmax
                    # d_loss/d_logit_i = sum_j (d_loss/d_prob_j * d_prob_j/d_logit_i)
                    # Simplified: just use policy gradient as approximate logit grad
                    p_grads = self.net.backward_policy(d_logits, cache)
                    v_grads = self.net.backward_value(
                        self.value_coef * 2 * (value - ret), cache
                    )

                    # Combine gradients
                    combined = {}
                    for k in p_grads:
                        combined[k] = p_grads[k] + v_grads.get(k, 0)
                    for k in v_grads:
                        if k not in combined:
                            combined[k] = v_grads[k]

                    self.net.update(combined)

                    total_pg_loss += pg_loss
                    total_v_loss += v_loss
                    total_entropy += entropy
                    n_updates += 1

        self._update_count += 1
        return {
            "policy_loss": round(total_pg_loss / max(1, n_updates), 4),
            "value_loss": round(total_v_loss / max(1, n_updates), 4),
            "entropy": round(total_entropy / max(1, n_updates), 4),
            "n_updates": n_updates,
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
