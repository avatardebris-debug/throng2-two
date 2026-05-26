"""
Single-game world model — residual MLP dynamics for ThrongletCell.

Adapted from throng5's WorldModel pattern:
  - Residual MLP: predicts (next_state = state + delta, reward)
  - Replay buffer for experience storage
  - dream_all_actions() for hypothesis evaluation
"""
from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..surprise_classifier import SurpriseResult
from .core import WorldModelCore
from .rollout import greedy_action_values


class SingleGameWorldModel(WorldModelCore):
    """
    Learned dynamics model for one action space.

    Predicts (next_state, reward, done_probability) given (state, action).
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
        super().__init__(
            feature_dim=feature_dim,
            n_actions=n_actions,
            batch_size=batch_size,
            min_transitions=min_transitions,
        )
        self._replay: deque = deque(maxlen=buffer_size)

        input_dim = feature_dim + n_actions

        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        ).to(self.device)

        self._state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, feature_dim),
        ).to(self.device)

        self._reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        ).to(self.device)

        self._done_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        ).to(self.device)

        all_params = (
            list(self._encoder.parameters())
            + list(self._state_head.parameters())
            + list(self._reward_head.parameters())
            + list(self._done_head.parameters())
        )
        self._optimizer = optim.Adam(all_params, lr=lr)

    def replay_size(self) -> int:
        return len(self._replay)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool = False,
    ):
        self._replay.append(
            (
                np.asarray(state, dtype=np.float32),
                action,
                np.asarray(next_state, dtype=np.float32),
                float(reward),
                float(done),
            )
        )

    def _network_params(self):
        return (
            list(self._encoder.parameters())
            + list(self._state_head.parameters())
            + list(self._reward_head.parameters())
            + list(self._done_head.parameters())
        )

    def _encode_batch(self, batch: list):
        batch_len = len(batch)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)
        rewards = torch.FloatTensor([b[3] for b in batch]).unsqueeze(1).to(self.device)
        actions_oh = torch.zeros(batch_len, self.n_actions).to(self.device)
        for i, b in enumerate(batch):
            actions_oh[i, int(b[1])] = 1.0
        dones = torch.FloatTensor(
            [b[4] if len(b) > 4 else 0.0 for b in batch]
        ).unsqueeze(1).to(self.device)
        return states, actions_oh, next_states, rewards, dones

    def _loss_on_batch(
        self,
        batch: list,
        *,
        train_done: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Union[float, None]]]:
        states, actions_oh, next_states, rewards, dones = self._encode_batch(batch)
        x = torch.cat([states, actions_oh], dim=1)
        encoded = self._encoder(x)
        delta_pred = self._state_head(encoded)
        reward_pred = self._reward_head(encoded)

        delta_target = next_states - states
        state_loss = F.mse_loss(delta_pred, delta_target)
        reward_loss = F.smooth_l1_loss(reward_pred, rewards)

        done_loss = None
        if train_done:
            done_pred = self._done_head(encoded)
            done_loss = F.binary_cross_entropy_with_logits(done_pred, dones)
            total_loss = state_loss + reward_loss * 0.1 + done_loss * 0.5
        else:
            total_loss = state_loss + reward_loss * 0.1

        return total_loss, {
            "state_loss": state_loss,
            "reward_loss": reward_loss,
            "done_loss": done_loss,
        }

    def train_step(self) -> Dict[str, float]:
        if len(self._replay) < self.batch_size:
            return {"wm_loss": 0.0, "wm_buffer": len(self._replay)}

        indices = np.random.choice(len(self._replay), self.batch_size, replace=False)
        batch = [self._replay[i] for i in indices]

        total_loss, parts = self._loss_on_batch(batch, train_done=True)

        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._network_params(), 5.0)
        self._optimizer.step()

        self._total_updates += 1
        loss_val = total_loss.item()
        self._losses.append(loss_val)

        return {
            "wm_loss": round(loss_val, 4),
            "wm_state_loss": round(parts["state_loss"].item(), 4),
            "wm_reward_loss": round(parts["reward_loss"].item(), 4),
            "wm_done_loss": round(parts["done_loss"].item(), 4),
            "wm_updates": self._total_updates,
        }

    def save_weights(self, path: str):
        state = {
            "encoder": self._encoder.state_dict(),
            "state_head": self._state_head.state_dict(),
            "reward_head": self._reward_head.state_dict(),
            "done_head": self._done_head.state_dict(),
            "updates": self._total_updates,
        }
        torch.save(state, path)

    def load_weights(self, path: str):
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location=self.device)
        self._encoder.load_state_dict(state["encoder"])
        self._state_head.load_state_dict(state["state_head"])
        self._reward_head.load_state_dict(state["reward_head"])
        self._done_head.load_state_dict(state["done_head"])
        self._total_updates = state.get("updates", 0)

    def predict(
        self, features: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float, float]:
        with torch.inference_mode():
            state_t = (
                torch.as_tensor(features, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            action_oh = torch.zeros(1, self.n_actions).to(self.device)
            action_oh[0, action] = 1.0

            x = torch.cat([state_t, action_oh], dim=1)
            encoded = self._encoder(x)
            delta = self._state_head(encoded)
            reward = self._reward_head(encoded)
            done_logit = self._done_head(encoded)

            next_features = (state_t + delta).squeeze(0).cpu().numpy()
            pred_reward = reward.item()
            done_prob = torch.sigmoid(done_logit).item()

        return next_features, pred_reward, done_prob

    def predict_next_reward(
        self, features: np.ndarray, action: int
    ) -> Tuple[np.ndarray, float]:
        next_features, pred_reward, _ = self.predict(features, action)
        return next_features, pred_reward

    def predict_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = states.shape[0]
        with torch.inference_mode():
            states_t = torch.as_tensor(states, dtype=torch.float32).to(self.device)
            actions_oh = torch.zeros(n, self.n_actions, device=self.device)
            actions_oh[torch.arange(n), actions.astype(np.int64)] = 1.0

            x = torch.cat([states_t, actions_oh], dim=1)
            encoded = self._encoder(x)
            deltas = self._state_head(encoded)
            rewards = self._reward_head(encoded).squeeze(1)
            done_logits = self._done_head(encoded).squeeze(1)

            next_states = (states_t + deltas).cpu().numpy()
            rew_arr = rewards.cpu().numpy()
            done_probs = torch.sigmoid(done_logits).cpu().numpy()

        return next_states, rew_arr, done_probs

    def dream_all_actions(
        self,
        state: np.ndarray,
        depth: int = 1,
        game_id: int = 0,
    ) -> np.ndarray:
        del game_id
        if not self.is_ready:
            return np.zeros(self.n_actions, dtype=np.float32)
        return greedy_action_values(
            self.predict_next_reward,
            state,
            self.n_actions,
            depth=depth,
        )

    def measure_surprise(
        self,
        state: np.ndarray,
        action: int,
        actual_next: np.ndarray,
        entity_tag: Optional[str] = None,
    ) -> SurpriseResult:
        predicted_next, _ = self.predict_next_reward(state, action)
        return self._surprise_clf.classify(
            predicted_next=predicted_next,
            actual_next=actual_next,
            prev_state=state,
            entity_tag=entity_tag,
        )

    def sim_reset(self, initial_state: np.ndarray) -> np.ndarray:
        self._current_sim_state = np.asarray(initial_state, dtype=np.float32).copy()
        return self._current_sim_state

    def step(
        self,
        action: int,
        real_next: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        if self._current_sim_state is None:
            raise RuntimeError("Call sim_reset(initial_state) before step().")

        next_state, reward = self.predict_next_reward(self._current_sim_state, action)
        self._current_sim_state = next_state

        surprise = 0.0
        if real_next is not None:
            real_next = np.asarray(real_next, dtype=np.float32)
            err = float(np.mean(np.abs(next_state - real_next)))
            self._sim2real_errors.append(err)
            surprise = err
            self._current_sim_state = real_next.copy()

        return next_state, reward, surprise

    def information_gain(
        self,
        state: np.ndarray,
        action: int,
        n_samples: int = 5,
    ) -> float:
        if not self.is_ready:
            return 1.0

        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_oh = torch.zeros(1, self.n_actions).to(self.device)
        action_oh[0, action] = 1.0

        predictions = []
        with torch.inference_mode():
            for _ in range(n_samples):
                noise = torch.randn_like(state_t) * 0.01
                x = torch.cat([state_t + noise, action_oh], dim=1)
                encoded = self._encoder(x)
                delta = self._state_head(encoded)
                predictions.append(delta.squeeze(0).cpu().numpy())

        pred_array = np.array(predictions)
        return float(np.mean(np.var(pred_array, axis=0)))

    def information_gains_all_actions(
        self,
        state: np.ndarray,
        n_samples: int = 5,
    ) -> np.ndarray:
        return np.array(
            [
                self.information_gain(state, a, n_samples=n_samples)
                for a in range(self.n_actions)
            ]
        )

    def fast_retrain(
        self,
        transitions: List[Tuple],
        lr_multiplier: float = 5.0,
        n_steps: int = 50,
    ) -> Dict[str, float]:
        if not transitions:
            return {"fast_retrain_loss": 0.0, "n_steps": 0}

        original_lr = self._optimizer.param_groups[0]["lr"]
        for pg in self._optimizer.param_groups:
            pg["lr"] = original_lr * lr_multiplier

        losses = []
        for _ in range(n_steps):
            idx = np.random.choice(
                len(transitions),
                min(len(transitions), self.batch_size),
                replace=True,
            )
            batch = [transitions[i] for i in idx]
            total_loss, _ = self._loss_on_batch(batch, train_done=False)

            self._optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self._encoder.parameters())
                + list(self._state_head.parameters())
                + list(self._reward_head.parameters()),
                5.0,
            )
            self._optimizer.step()
            losses.append(total_loss.item())

        for pg in self._optimizer.param_groups:
            pg["lr"] = original_lr

        avg_loss = float(np.mean(losses))
        self._losses.append(avg_loss)
        return {"fast_retrain_loss": round(avg_loss, 4), "n_steps": n_steps}

    def state_dict_all(self) -> dict:
        return {
            "encoder": self._encoder.state_dict(),
            "state_head": self._state_head.state_dict(),
            "reward_head": self._reward_head.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "total_updates": self._total_updates,
        }

    def load_state_dict_all(self, state: dict):
        self._encoder.load_state_dict(state["encoder"])
        self._state_head.load_state_dict(state["state_head"])
        self._reward_head.load_state_dict(state["reward_head"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._total_updates = state.get("total_updates", 0)


# Backward-compatible name used across the repo
CellWorldModel = SingleGameWorldModel
