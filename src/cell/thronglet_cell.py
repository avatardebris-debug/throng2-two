"""
ThrongletCell — The Minimal Fractal Unit.

Orchestrates:
  - StateEncoder (obs → compressed features)
  - SNNFeatureExtractor (compressed → SNN activity features)
  - PPOHead (combined features → action)

The cell IS the Gymnasium agent interface:
  obs → cell.step(obs) → action
  reward, done → cell.learn(reward, done)

Design principles:
  - SNN is a RECEIVER, not an actor (no reward signal to SNN)
  - Encoder trains end-to-end with PPO (gradients flow through)
  - SNN features are APPENDED to encoder features (augmentation)
  - Cell can be saved/loaded for expert storage
  - Cell tracks its own neuron count for future growth/pruning
"""

import os
import torch
import numpy as np
from typing import Optional, Tuple

from .encoder import StateEncoder
from .snn_features import SNNFeatureExtractor
from .ppo_head import PPOHead
from .world_model import CellWorldModel
from .dreamer import CellDreamer
from .growth_controller import GrowthController


class ThrongletCell:
    """
    Minimal fractal unit: Encoder + SNN + PPO.

    Usage:
        cell = ThrongletCell(obs_dim=4, n_actions=2)
        obs = env.reset()
        for step in range(max_steps):
            action = cell.step(obs)
            obs, reward, done, info = env.step(action)
            stats = cell.learn(reward, done)
            if done:
                cell.reset()
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        snn_neurons: int = 64,
        compressed_dim: int = 16,
        ppo_hidden: int = 64,
        ppo_lr: float = 3e-4,
        ppo_rollout_length: int = 128,
        use_snn: bool = True,
        use_dreamer: bool = True,
        dream_interval: int = 10,
        dream_depth: int = 3,
        use_growth: bool = True,
        max_neurons: int = 512,
    ):
        """
        Args:
            obs_dim: Raw observation dimension.
            n_actions: Number of discrete actions.
            snn_neurons: Number of SNN neurons.
            compressed_dim: Encoder output dimension.
            ppo_hidden: PPO hidden layer size.
            ppo_lr: PPO learning rate.
            ppo_rollout_length: Steps between PPO updates.
            use_snn: If False, skip SNN features (for ablation).
            use_dreamer: If False, skip WorldModel+Dreamer (for ablation).
            dream_interval: Dream every N steps.
            dream_depth: Steps to dream ahead.
            use_growth: If False, skip growth/pruning controller.
            max_neurons: Maximum SNN neurons (computational budget).
        """
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.use_snn = use_snn
        self.compressed_dim = compressed_dim

        # Observation normalization (running mean/std)
        self._obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self._obs_var = np.ones(obs_dim, dtype=np.float32)
        self._obs_count = 0

        # 1. Encoder: obs → compressed
        self.encoder = StateEncoder(obs_dim, compressed_dim)

        # 2. SNN: compressed → SNN features
        if use_snn:
            self.snn = SNNFeatureExtractor(
                n_neurons=snn_neurons,
                input_dim=compressed_dim,
            )
            # PPO input = raw_obs + encoder_features + snn_features
            feature_dim = obs_dim + compressed_dim + self.snn.feature_dim
        else:
            self.snn = None
            # PPO input = raw_obs + encoder_features
            feature_dim = obs_dim + compressed_dim

        # 3. PPO: combined features → action
        self.ppo = PPOHead(
            input_dim=feature_dim,
            n_actions=n_actions,
            hidden=ppo_hidden,
            lr=ppo_lr,
            rollout_length=ppo_rollout_length,
        )

        # The encoder is part of PPO's optimization graph
        self.ppo.optimizer.add_param_group(
            {"params": self.encoder.parameters(), "lr": ppo_lr}
        )

        # 4. WorldModel + Dreamer (optional)
        self.use_dreamer = use_dreamer
        if use_dreamer:
            self.world_model = CellWorldModel(
                feature_dim=feature_dim,
                n_actions=n_actions,
                hidden_size=128,
            )
            self.dreamer = CellDreamer(
                n_actions=n_actions,
                dream_interval=dream_interval,
                dream_depth=dream_depth,
            )
            self._wm_train_interval = 4  # Train WM every 4 steps
        else:
            self.world_model = None
            self.dreamer = None
            self._wm_train_interval = 0

        # 5. Growth/Pruning Controller (optional)
        self.use_growth = use_growth and use_snn  # Growth only works with SNN
        if self.use_growth:
            self.growth_controller = GrowthController(
                min_neurons=32,
                max_neurons=max_neurons,
                growth_batch=16,
                prune_batch=8,
                plateau_window=50,
                check_interval=10,
            )
        else:
            self.growth_controller = None

        # Previous features (for WM transition storage)
        self._prev_features = None

        # State tracking
        self._last_features = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._total_episodes = 0
        self._total_steps = 0
        self._reward_history = []

    @property
    def neuron_count(self) -> int:
        """Current number of SNN neurons."""
        return self.snn.n_neurons if self.snn else 0

    def step(self, obs: np.ndarray) -> int:
        """
        Process observation and select action.

        This is the core loop:
        1. Encode raw observation
        2. Run SNN to extract temporal features
        3. Concatenate encoder + SNN features
        4. PPO selects action

        Args:
            obs: Raw observation from environment.

        Returns:
            action: Selected discrete action.
        """
        # 0. Normalize observations (running statistics)
        self._update_obs_stats(obs)
        norm_obs = self._normalize_obs(obs)

        # 1. Encode
        compressed = self.encoder.encode(norm_obs)

        # 2. Build feature vector: ALWAYS include raw obs
        if self.snn is not None:
            snn_feat = self.snn.step(compressed)
            features = np.concatenate([norm_obs, compressed, snn_feat])
        else:
            features = np.concatenate([norm_obs, compressed])

        # 3. PPO action selection
        action, log_prob, value = self.ppo.select_action(features)

        # 4. Dream-based action blending (optional)
        if self.dreamer is not None and self.world_model is not None:
            dream_values = self.dreamer.dream(features, self.world_model)
            action = self.dreamer.blend_action(
                ppo_action=action,
                ppo_log_prob=log_prob,
                dream_values=dream_values,
                wm_confidence=self.world_model.confidence,
            )

        # Store for learning
        self._last_features = features
        self._last_action = action
        self._last_log_prob = log_prob
        self._last_value = value

        self._episode_steps += 1
        self._total_steps += 1
        return action

    def learn(self, reward: float, done: bool) -> Optional[dict]:
        """
        Process reward and potentially update policy.

        Args:
            reward: Reward from environment.
            done: Whether episode ended.

        Returns:
            Update stats dict if PPO updated, else None.
        """
        if self._last_features is None:
            return None

        # Store transition
        self.ppo.store_transition(
            obs=self._last_features,
            action=self._last_action,
            log_prob=self._last_log_prob,
            reward=reward,
            done=done,
            value=self._last_value,
        )

        self._episode_reward += reward

        # Store transition in WorldModel
        if self.world_model is not None and self._prev_features is not None:
            self.world_model.store_transition(
                state=self._prev_features,
                action=self._last_action,
                next_state=self._last_features,
                reward=reward,
            )

        # Train WorldModel periodically
        wm_stats = None
        if (self.world_model is not None
                and self._wm_train_interval > 0
                and self._total_steps % self._wm_train_interval == 0):
            wm_stats = self.world_model.train_step()

        # Save current features for next WM transition
        self._prev_features = self._last_features.copy()

        # Track episode completion
        if done:
            self._total_episodes += 1
            self._reward_history.append(self._episode_reward)

            # Growth/Pruning check
            if self.growth_controller is not None and self.snn is not None:
                self.growth_controller.on_episode_end(self._episode_reward)
                decision = self.growth_controller.decide(
                    current_neurons=self.snn.n_neurons,
                    snn_activity=self.snn.activity.reshape(1, -1),
                    wm_loss=(
                        self.world_model.confidence
                        if self.world_model else 0.0
                    ),
                )
                if decision == "grow":
                    added = self.growth_controller.grow_snn(self.snn)
                    if added > 0:
                        self._rebuild_ppo_input()
                elif decision == "prune":
                    removed = self.growth_controller.prune_snn(self.snn)
                    if removed > 0:
                        self._rebuild_ppo_input()

            self._episode_reward = 0.0
            self._episode_steps = 0

        # PPO update when buffer is full
        update_stats = None
        if self.ppo.should_update():
            # Bootstrap value for non-terminal states
            if not done and self._last_features is not None:
                last_value = self.ppo.get_value(self._last_features)
            else:
                last_value = 0.0
            update_stats = self.ppo.update(last_value)

        return update_stats

    def reset(self):
        """Reset between episodes."""
        if self.snn is not None:
            self.snn.reset()
        if self.dreamer is not None:
            self.dreamer.reset()
        self._last_features = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None
        self._prev_features = None

    def save(self, path: str):
        """
        Save cell checkpoint (for expert storage).

        Saves:
        - Encoder weights
        - PPO network + optimizer
        - SNN state (input_weights, connectivity)
        - Metadata (obs_dim, n_actions, neuron_count, performance)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "ppo_network_state": self.ppo.network.state_dict(),
            "ppo_optimizer_state": self.ppo.optimizer.state_dict(),
            "metadata": {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "neuron_count": self.neuron_count,
                "use_snn": self.use_snn,
                "total_episodes": self._total_episodes,
                "total_steps": self._total_steps,
                "reward_history": self._reward_history[-100:],  # Last 100
                "avg_reward_last_100": (
                    float(np.mean(self._reward_history[-100:]))
                    if self._reward_history else 0.0
                ),
            },
        }

        if self.snn is not None:
            checkpoint["snn_state"] = {
                "input_weights": self.snn.input_weights,
                "neuron_frequencies": self.snn.neuron_frequencies,
                # Note: SNN connectivity (weights) is structural, not learned
                # For now we re-init it on load. In v2, we'll save it too.
            }

        torch.save(checkpoint, path)

        # Save WorldModel separately (it has its own state)
        if self.world_model is not None:
            wm_path = path.replace('.pt', '_wm.pt')
            torch.save(self.world_model.state_dict_all(), wm_path)

    def load(self, path: str):
        """Load cell checkpoint."""
        checkpoint = torch.load(path, weights_only=False)

        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.ppo.network.load_state_dict(checkpoint["ppo_network_state"])
        self.ppo.optimizer.load_state_dict(checkpoint["ppo_optimizer_state"])

        meta = checkpoint["metadata"]
        self._total_episodes = meta.get("total_episodes", 0)
        self._total_steps = meta.get("total_steps", 0)
        self._reward_history = meta.get("reward_history", [])

        if self.snn is not None and "snn_state" in checkpoint:
            snn_state = checkpoint["snn_state"]
            self.snn.input_weights = snn_state["input_weights"]
            self.snn.neuron_frequencies = snn_state["neuron_frequencies"]

        # Load WorldModel
        if self.world_model is not None:
            wm_path = path.replace('.pt', '_wm.pt')
            try:
                wm_state = torch.load(wm_path, weights_only=False)
                self.world_model.load_state_dict_all(wm_state)
            except FileNotFoundError:
                pass  # No WM checkpoint — fresh model

    def _rebuild_ppo_input(self):
        """Rebuild PPO input layer after neuron count change."""
        if self.snn is not None:
            new_dim = self.obs_dim + self.compressed_dim + self.snn.feature_dim
        else:
            new_dim = self.obs_dim + self.compressed_dim

        old_dim = self.ppo.input_dim
        if new_dim != old_dim:
            # SNN feature_dim is fixed (n_regions*2+1), so this only triggers
            # if we change n_regions. For now, neuron count changes don't
            # change feature_dim, so this is a safety check.
            pass  # feature_dim stays the same after grow/prune

    def _update_obs_stats(self, obs: np.ndarray):
        """Update running observation statistics for normalization."""
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        std = np.sqrt(np.maximum(self._obs_var, 1e-8))
        return ((obs - self._obs_mean) / std).astype(np.float32)

    def stats(self) -> dict:
        """Complete cell statistics."""
        avg_100 = (
            float(np.mean(self._reward_history[-100:]))
            if self._reward_history else 0.0
        )
        result = {
            "total_episodes": int(self._total_episodes),
            "total_steps": int(self._total_steps),
            "avg_reward_last_100": round(float(avg_100), 2),
            "neuron_count": int(self.neuron_count),
            "encoder": self.encoder.stats(),
            "ppo": self.ppo.stats(),
        }
        if self.snn is not None:
            result["snn"] = self.snn.stats()
        if self.world_model is not None:
            result["world_model"] = self.world_model.stats()
        if self.dreamer is not None:
            result["dreamer"] = self.dreamer.stats()
        if self.growth_controller is not None:
            result["growth"] = self.growth_controller.stats()
        return result
