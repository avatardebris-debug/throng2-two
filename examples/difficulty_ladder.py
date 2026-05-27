# -*- coding: utf-8 -*-
"""
difficulty_ladder.py -- Transfer learning benchmark for Throng2.

Three conditions on a difficulty ladder:
  scratch             -- train from random init
  transfer            -- fit encoder PCA on source obs, then train target
  transfer+contrastive -- fit encoder contrastively on source obs, then train target

Levels (pure numpy / gymnasium, no GPU):
  Level 1: CartPole -> MountainCar
  Level 2: CartPole+MountainCar -> LunarLander

Usage:
    python examples/difficulty_ladder.py --quick
    python examples/difficulty_ladder.py --episodes 100 --level 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.encoder.universal_encoder import EncoderRegistry, register_game, EncoderConfig

# ---------------------------------------------------------------
# GYM RUNNER
# ---------------------------------------------------------------

class GymRunner:
    """Thin wrapper around a gymnasium environment."""

    def __init__(self, env_id: str, seed: int = 0):
        import gymnasium as gym
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.n_actions = self.env.action_space.n
        self._seed = seed

    def reset(self):
        obs, _ = self.env.reset(seed=self._seed)
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: int):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return np.asarray(obs, dtype=np.float32), float(reward), done

    def close(self):
        self.env.close()


# ---------------------------------------------------------------
# SIMPLE Q-AGENT (linear, no torch)
# ---------------------------------------------------------------

class SimpleQAgent:
    def __init__(self, obs_dim, n_actions, lr=5e-3, gamma=0.99,
                 eps_start=0.5, eps_end=0.05, eps_decay=300):
        self.n_actions = n_actions
        self.lr, self.gamma = lr, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.W = np.zeros((obs_dim, n_actions), dtype=np.float32)
        self.b = np.zeros(n_actions, dtype=np.float32)
        self._step = 0

    @property
    def epsilon(self):
        frac = min(1.0, self._step / max(1, self.eps_decay))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = obs @ self.W + self.b
        return int(np.argmax(q))

    def learn(self, obs, action, reward, next_obs, done):
        self._step += 1
        q_next = 0.0 if done else float(np.max(next_obs @ self.W + self.b))
        target = reward + self.gamma * q_next
        q_pred = float(obs @ self.W[:, action] + self.b[action])
        err = target - q_pred
        self.W[:, action] += self.lr * err * obs
        self.b[action] += self.lr * err


# ---------------------------------------------------------------
# GAME CONFIG
# ---------------------------------------------------------------

GAME_ENVS = {
    "cartpole":    ("CartPole-v1",     2),
    "mountaincar": ("MountainCar-v0",  3),
    "lunarlander": ("LunarLander-v3",  4),   # v3 = stable discrete action space
}

LEVELS = {
    1: {"source": ["cartpole"],                    "target": "mountaincar"},
    2: {"source": ["cartpole", "mountaincar"],      "target": "lunarlander"},
}

REWARD_THRESHOLDS = {
    "mountaincar": -80.0,   # shaped rewards, so threshold is higher than raw -110
    "lunarlander": 100.0,
}


# ---------------------------------------------------------------
# EPISODE RUNNER
# ---------------------------------------------------------------

def run_game_episodes(
    game_name: str,
    n_episodes: int,
    enc: Optional[EncoderRegistry] = None,
    verbose: bool = False,
) -> List[float]:
    """Run N episodes, return per-episode (shaped) rewards."""
    env_id, _ = GAME_ENVS.get(game_name, (None, None))
    if env_id is None:
        return [0.0] * n_episodes

    try:
        runner = GymRunner(env_id)
    except Exception as e:
        if verbose:
            print(f"  Could not create {env_id}: {e}")
        return [0.0] * n_episodes

    enc_games = set(enc._encoders.keys()) if enc is not None else set()
    use_enc = enc is not None and game_name in enc_games
    agent_obs_dim = enc.out_dim if use_enc else len(runner.reset())

    agent = SimpleQAgent(agent_obs_dim, runner.n_actions)
    rewards = []

    for ep in range(n_episodes):
        try:
            obs = runner.reset()
        except Exception:
            rewards.append(0.0)
            continue

        if use_enc:
            try:
                z = enc.encode(game_name, obs.flatten())
            except Exception:
                z = obs.flatten()[:agent_obs_dim]
        else:
            z = obs.flatten()[:agent_obs_dim]

        ep_reward = 0.0
        for _ in range(500):
            action = agent.act(z)
            try:
                next_obs, reward, done = runner.step(action)
            except Exception:
                break

            # Reward shaping for MountainCar: velocity + progress bonus
            if game_name == "mountaincar" and len(next_obs) >= 2:
                pos, vel = float(next_obs[0]), float(next_obs[1])
                reward = reward + 0.5 * (pos + 0.5) + 2.0 * abs(vel)

            next_arr = next_obs.flatten()
            if use_enc:
                try:
                    z_next = enc.encode(game_name, next_arr)
                except Exception:
                    z_next = next_arr[:agent_obs_dim]
            else:
                z_next = next_arr[:agent_obs_dim]

            agent.learn(z, action, reward, z_next, done)
            ep_reward += reward
            z = z_next
            if done:
                break

        rewards.append(ep_reward)
        if verbose and (ep + 1) % 5 == 0:
            print(f"    ep {ep+1}: reward={ep_reward:.1f}")

    try:
        runner.close()
    except Exception:
        pass

    return rewards


# ---------------------------------------------------------------
# CONDITION RUNNER
# ---------------------------------------------------------------

def run_condition(
    level: int,
    mode: str,
    n_source_eps: int = 30,
    n_target_eps: int = 50,
    z_dim: int = 16,
    verbose: bool = True,
) -> dict:
    cfg = LEVELS[level]
    source_games = cfg["source"]
    target_game  = cfg["target"]
    t0 = time.time()

    # Register games
    all_games = list(source_games) + [target_game]
    for i, gname in enumerate(all_games):
        try:
            register_game(EncoderConfig(
                game_name=gname, game_id=i, obs_type="flat", obs_dim=8
            ))
        except Exception:
            pass

    enc = EncoderRegistry(z_dim=z_dim, games=all_games)
    result = {
        "level": level, "mode": mode,
        "source_games": source_games, "target_game": target_game,
        "source_rewards": {}, "target_rewards": [],
        "episodes_to_threshold": None,
    }

    if verbose:
        print(f"\n  [{mode.upper()}] Level {level}: {source_games} -> {target_game}")

    # Phase 1: train on source games
    obs_by_game: Dict[str, list] = {}
    for gname in source_games:
        if verbose:
            print(f"  Training source: {gname} ({n_source_eps} eps)...")
        rewards = run_game_episodes(gname, n_source_eps, enc=enc, verbose=False)
        result["source_rewards"][gname] = round(float(np.mean(rewards[-10:])), 2)

        if mode in ("transfer", "transfer+contrastive"):
            env_id, _ = GAME_ENVS.get(gname, (None, None))
            if env_id:
                try:
                    r = GymRunner(env_id)
                    obs_list = [r.reset()]
                    for _ in range(199):
                        obs_list.append(r.step(np.random.randint(r.n_actions))[0])
                    obs_by_game[gname] = obs_list
                    r.close()
                except Exception:
                    pass

    # Phase 2: fit encoders
    if mode == "transfer" and obs_by_game:
        enc.fit_all(obs_by_game)
    elif mode == "transfer+contrastive" and obs_by_game:
        enc.fit_contrastive_all(obs_by_game, n_epochs=10, verbose=False)

    # Phase 3: train on target
    if verbose:
        print(f"  Training target: {target_game} ({n_target_eps} eps)...")
    target_rewards = run_game_episodes(target_game, n_target_eps, enc=enc, verbose=verbose)
    result["target_rewards"] = [round(r, 2) for r in target_rewards]

    threshold = REWARD_THRESHOLDS.get(target_game, 0.0)
    window = 5
    for i in range(window, len(target_rewards) + 1):
        if np.mean(target_rewards[i-window:i]) >= threshold:
            result["episodes_to_threshold"] = i
            break

    result["target_final_mean"] = round(float(np.mean(target_rewards[-10:])), 2)
    result["elapsed"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------

def print_summary(all_results: List[dict]):
    print("\n" + "=" * 72)
    print(f"{'Level':>5}  {'Mode':>20}  {'Target':>16}  {'Final Reward':>13}  {'Eps->Thresh':>11}")
    print("-" * 72)
    for r in all_results:
        eps_str = str(r["episodes_to_threshold"]) if r["episodes_to_threshold"] else "never"
        print(f"{r['level']:>5}  {r['mode']:>20}  {r['target_game']:>16}  "
              f"{r['target_final_mean']:>13.2f}  {eps_str:>11}")
    print("=" * 72)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Throng2 difficulty ladder benchmark")
    parser.add_argument("--quick",      action="store_true")
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--source-eps", type=int, default=30)
    parser.add_argument("--level",      type=int, default=0, help="0=all levels")
    parser.add_argument("--z-dim",      type=int, default=16)
    parser.add_argument("--save",       type=str, default="results/difficulty_ladder.json")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.episodes   = 20
        args.source_eps = 15

    levels = [args.level] if args.level > 0 else [1, 2]
    modes  = ["scratch", "transfer", "transfer+contrastive"]

    all_results = []
    for level in levels:
        for mode in modes:
            try:
                r = run_condition(
                    level=level, mode=mode,
                    n_source_eps=args.source_eps,
                    n_target_eps=args.episodes,
                    z_dim=args.z_dim,
                    verbose=args.verbose,
                )
                all_results.append(r)
            except Exception as e:
                print(f"  SKIP level={level} mode={mode}: {e}")

    print_summary(all_results)

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()
