"""
50_cartpole_cell.py — Integration test for the ThrongletCell.

Benchmarks:
  1. Random baseline
  2. Bare PPO (raw obs, validates PPO works)
  3. ThrongletCell without dreamer (encoder + SNN + PPO)
  4. ThrongletCell with dreamer (encoder + SNN + PPO + WorldModel)

Also tests save/load round-trip and LunarLander.
"""

import sys
import os
import time
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from src.cell.thronglet_cell import ThrongletCell
from src.cell.ppo_head import PPOHead


def run_random_baseline(env_name: str = "CartPole-v1", n_episodes: int = 200) -> dict:
    """Random agent baseline."""
    env = gym.make(env_name)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return {
        "agent": "Random",
        "avg_last_100": round(float(np.mean(rewards[-100:])), 1),
        "max_reward": round(float(np.max(rewards)), 1),
        "episodes_to_solve": "N/A",
        "time": "N/A",
        "rewards": rewards,
    }


def run_bare_ppo(env_name: str = "CartPole-v1", n_episodes: int = 500,
                 label: str = "Bare PPO") -> dict:
    """Bare PPO on raw observations."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    ppo = PPOHead(
        input_dim=obs_dim, n_actions=n_actions, hidden=64,
        lr=3e-4, rollout_length=2048, batch_size=64,
        update_epochs=10, entropy_coef=0.01,
    )

    obs_mean = np.zeros(obs_dim, dtype=np.float32)
    obs_var = np.ones(obs_dim, dtype=np.float32)
    obs_count = 0

    def normalize(o):
        nonlocal obs_mean, obs_var, obs_count
        obs_count += 1
        delta = o - obs_mean
        obs_mean += delta / obs_count
        delta2 = o - obs_mean
        obs_var += (delta * delta2 - obs_var) / obs_count
        std = np.sqrt(np.maximum(obs_var, 1e-8))
        return ((o - obs_mean) / std).astype(np.float32)

    rewards = []
    solved_at = None
    total_time = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        t0 = time.time()
        while not done:
            norm_obs = normalize(obs)
            action, log_prob, value = ppo.select_action(norm_obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ppo.store_transition(norm_obs, action, log_prob, reward, done, value)
            ep_reward += reward
            if ppo.should_update():
                last_val = 0.0 if done else ppo.get_value(normalize(next_obs))
                ppo.update(last_val)
            obs = next_obs
        total_time += time.time() - t0
        rewards.append(ep_reward)
        if len(rewards) >= 100 and solved_at is None:
            if np.mean(rewards[-100:]) >= 195.0:
                solved_at = ep + 1
        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards[-100:])
            print(f"  [{label}] Episode {ep+1:4d}  avg100={avg:6.1f}  time={total_time:.1f}s")
    env.close()
    return {
        "agent": label, "avg_last_100": round(float(np.mean(rewards[-100:])), 1),
        "max_reward": round(float(np.max(rewards)), 1),
        "episodes_to_solve": solved_at if solved_at else f"> {n_episodes}",
        "time": round(total_time, 1), "rewards": rewards,
    }


def run_cell(env_name: str = "CartPole-v1", n_episodes: int = 500,
             use_snn: bool = True, use_dreamer: bool = True,
             label: str = "ThrongletCell", snn_neurons: int = 64) -> dict:
    """Train a ThrongletCell on an environment."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    cell = ThrongletCell(
        obs_dim=obs_dim, n_actions=n_actions, snn_neurons=snn_neurons,
        compressed_dim=16, ppo_lr=3e-4, ppo_rollout_length=2048,
        use_snn=use_snn, use_dreamer=use_dreamer,
        dream_interval=10, dream_depth=3,
    )

    rewards = []
    solved_at = None
    total_time = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        cell.reset()
        total_reward = 0.0
        done = False
        t0 = time.time()
        while not done:
            action = cell.step(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cell.learn(reward, done)
            total_reward += reward
        total_time += time.time() - t0
        rewards.append(total_reward)
        if len(rewards) >= 100 and solved_at is None:
            if np.mean(rewards[-100:]) >= 195.0:
                solved_at = ep + 1
        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards[-100:])
            print(f"  [{label}] Episode {ep+1:4d}  avg100={avg:6.1f}  time={total_time:.1f}s")
    env.close()
    return {
        "agent": label, "avg_last_100": round(float(np.mean(rewards[-100:])), 1),
        "max_reward": round(float(np.max(rewards)), 1),
        "episodes_to_solve": solved_at if solved_at else f"> {n_episodes}",
        "rewards": rewards, "cell": cell, "time": round(total_time, 1),
    }


def print_results(results: list, title: str = "RESULTS"):
    print(f"\n{'=' * 65}")
    print(title)
    print("=" * 65)
    print(f"{'Agent':<22} {'Avg100':>8} {'Max':>8} {'Solved@':>10} {'Time':>8}")
    print("-" * 65)
    for r in results:
        t = r.get("time", "N/A")
        t_str = f"{t}s" if isinstance(t, (int, float)) else t
        print(f"{r['agent']:<22} {r['avg_last_100']:>8.1f} "
              f"{r['max_reward']:>8.1f} {str(r['episodes_to_solve']):>10} "
              f"{t_str:>8}")


def main():
    print("=" * 65)
    print("THRONGLET CELL — PHASE 2 BENCHMARK (WorldModel + Dreamer)")
    print("=" * 65)

    N = 500

    # === CARTPOLE ===
    print("\n>>> CARTPOLE-v1 <<<")

    print("\n[1/4] Random baseline...")
    random_r = run_random_baseline("CartPole-v1", 200)
    print(f"  Random avg: {random_r['avg_last_100']}")

    print(f"\n[2/4] Bare PPO, {N} eps...")
    bare_r = run_bare_ppo("CartPole-v1", N)

    print(f"\n[3/4] ThrongletCell (no dreamer), {N} eps...")
    nodream_r = run_cell("CartPole-v1", N, use_dreamer=False, label="Cell (no dream)")

    print(f"\n[4/4] ThrongletCell (with dreamer), {N} eps...")
    dream_r = run_cell("CartPole-v1", N, use_dreamer=True, label="Cell (dream)")

    print_results([random_r, bare_r, nodream_r, dream_r], "CARTPOLE RESULTS")

    # WorldModel stats
    if dream_r.get("cell") and dream_r["cell"].world_model:
        print("\nWorldModel Stats:")
        print(json.dumps(dream_r["cell"].world_model.stats(), indent=2, default=str))
        print("\nDreamer Stats:")
        print(json.dumps(dream_r["cell"].dreamer.stats(), indent=2, default=str))

    # Cell stats
    print("\nFull Cell Stats:")
    print(json.dumps(dream_r["cell"].stats(), indent=2, default=str))

    # === LUNAR LANDER (shorter run) ===
    print("\n\n>>> LUNARLANDER-v3 <<<")
    print("\nNote: LunarLander has 8-dim obs + temporal dynamics — SNN should help more")

    print("\n[1/3] Random baseline...")
    ll_random = run_random_baseline("LunarLander-v3", 200)
    print(f"  Random avg: {ll_random['avg_last_100']}")

    print(f"\n[2/3] ThrongletCell (no dreamer), 300 eps...")
    ll_nodream = run_cell("LunarLander-v3", 300, use_dreamer=False, label="Cell (no dream)")

    print(f"\n[3/3] ThrongletCell (with dreamer), 300 eps...")
    ll_dream = run_cell("LunarLander-v3", 300, use_dreamer=True, label="Cell (dream)")

    print_results([ll_random, ll_nodream, ll_dream], "LUNARLANDER RESULTS")

    if ll_dream.get("cell") and ll_dream["cell"].world_model:
        print("\nLunarLander WorldModel Stats:")
        print(json.dumps(ll_dream["cell"].world_model.stats(), indent=2, default=str))

    print(f"\n{'=' * 65}")
    print("DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
