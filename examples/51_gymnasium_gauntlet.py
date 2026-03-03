"""
51_gymnasium_gauntlet.py — Multi-environment benchmark for the ThrongletCell.

Tests the cell across 6 classic Gymnasium environments:
  1. CartPole-v1 (4-dim obs, 2 actions, balance)
  2. LunarLander-v3 (8-dim obs, 4 actions, landing)
  3. MountainCar-v0 (2-dim obs, 3 actions, momentum)
  4. Acrobot-v1 (6-dim obs, 3 actions, swing-up)
  5. FrozenLake-v1 (discrete obs, 4 actions, navigation)
  6. Blackjack-v1 (3-dim obs, 2 actions, card game)

Designed to run on cloud compute.
Run: python examples/51_gymnasium_gauntlet.py

Results are saved to examples/gauntlet_results.json
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from src.cell.thronglet_cell import ThrongletCell
from src.cell.ppo_head import PPOHead


# ── Environment Configs ─────────────────────────────────────────────

ENVS = [
    {
        "name": "CartPole-v1",
        "episodes": 1000,
        "solve_threshold": 195.0,
        "obs_type": "continuous",
    },
    {
        "name": "LunarLander-v3",
        "episodes": 1000,
        "solve_threshold": 200.0,
        "obs_type": "continuous",
    },
    {
        "name": "MountainCar-v0",
        "episodes": 1000,
        "solve_threshold": -110.0,
        "obs_type": "continuous",
    },
    {
        "name": "Acrobot-v1",
        "episodes": 1000,
        "solve_threshold": -100.0,
        "obs_type": "continuous",
    },
    {
        "name": "FrozenLake-v1",
        "episodes": 2000,
        "solve_threshold": 0.70,
        "obs_type": "discrete",
        "env_kwargs": {"is_slippery": False},
    },
    {
        "name": "Blackjack-v1",
        "episodes": 5000,
        "solve_threshold": 0.0,
        "obs_type": "tuple",
    },
]


# ── Helpers ──────────────────────────────────────────────────────────

def get_obs_dim(env):
    """Get observation dimension, handling discrete and tuple obs spaces."""
    space = env.observation_space
    if hasattr(space, "shape") and space.shape:
        return space.shape[0]
    elif hasattr(space, "n"):
        return space.n  # Discrete — use one-hot
    elif hasattr(space, "spaces"):
        # Tuple space (like Blackjack)
        return sum(
            s.n if hasattr(s, "n") else s.shape[0]
            for s in space.spaces
        )
    return 4  # fallback


def encode_obs(obs, env, obs_type):
    """Convert observation to float array."""
    if obs_type == "discrete":
        # One-hot encode
        n = env.observation_space.n
        vec = np.zeros(n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec
    elif obs_type == "tuple":
        # Flatten tuple elements
        parts = []
        for i, val in enumerate(obs):
            if isinstance(val, (int, np.integer, bool, np.bool_)):
                parts.append(float(val))
            else:
                parts.append(float(val))
        return np.array(parts, dtype=np.float32)
    else:
        return np.asarray(obs, dtype=np.float32)


# ── Random Baseline ─────────────────────────────────────────────────

def run_random(env_name, n_episodes, env_kwargs=None):
    """Random agent baseline."""
    kwargs = env_kwargs or {}
    env = gym.make(env_name, **kwargs)
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
    return rewards


# ── Bare PPO ─────────────────────────────────────────────────────────

def run_bare_ppo(env_name, n_episodes, obs_type="continuous", env_kwargs=None):
    """Bare PPO on raw observations."""
    kwargs = env_kwargs or {}
    env = gym.make(env_name, **kwargs)
    obs_dim = get_obs_dim(env)
    n_actions = env.action_space.n

    ppo = PPOHead(
        input_dim=obs_dim, n_actions=n_actions, hidden=64,
        lr=3e-4, rollout_length=2048, batch_size=64,
        update_epochs=10,
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
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs = encode_obs(obs, env, obs_type)
        done = False
        ep_reward = 0.0
        while not done:
            norm_obs = normalize(obs)
            action, log_prob, value = ppo.select_action(norm_obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = encode_obs(next_obs, env, obs_type)
            done = terminated or truncated
            ppo.store_transition(norm_obs, action, log_prob, reward, done, value)
            ep_reward += reward
            if ppo.should_update():
                last_val = 0.0 if done else ppo.get_value(normalize(next_obs))
                ppo.update(last_val)
            obs = next_obs
        rewards.append(ep_reward)
    env.close()
    return rewards


# ── ThrongletCell ────────────────────────────────────────────────────

def run_cell(env_name, n_episodes, obs_type="continuous", use_dreamer=True,
             env_kwargs=None):
    """ThrongletCell agent."""
    kwargs = env_kwargs or {}
    env = gym.make(env_name, **kwargs)
    obs_dim = get_obs_dim(env)
    n_actions = env.action_space.n

    cell = ThrongletCell(
        obs_dim=obs_dim, n_actions=n_actions, snn_neurons=64,
        compressed_dim=16, ppo_lr=3e-4, ppo_rollout_length=2048,
        use_snn=True, use_dreamer=use_dreamer, use_growth=True,
        max_neurons=256,
    )

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs = encode_obs(obs, env, obs_type)
        cell.reset()
        total = 0.0
        done = False
        while not done:
            action = cell.step(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = encode_obs(next_obs, env, obs_type)
            done = terminated or truncated
            cell.learn(reward, done)
            total += reward
            obs = next_obs
        rewards.append(total)
    env.close()
    return rewards, cell


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("THRONGLET CELL — GYMNASIUM GAUNTLET")
    print("=" * 70)
    print()

    all_results = {}

    out_path = os.path.join(os.path.dirname(__file__), "gauntlet_results.json")

    for i, cfg in enumerate(ENVS):
        env_name = cfg["name"]
        n_eps = cfg["episodes"]
        threshold = cfg["solve_threshold"]
        obs_type = cfg.get("obs_type", "continuous")
        env_kwargs = cfg.get("env_kwargs", {})

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(ENVS)}] {env_name}")
        print(f"  Episodes: {n_eps}, Solve threshold: {threshold}")
        print(f"{'='*70}")

        try:
            # Quick test that env can be created
            test_env = gym.make(env_name, **env_kwargs)
            test_env.close()
        except Exception as e:
            print(f"  SKIPPED: {e}")
            all_results[env_name] = {"skipped": True, "reason": str(e)}
            # Save partial results after each env
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            continue

        try:
            # 1. Random baseline
            print("  Running Random baseline...")
            t0 = time.time()
            rand_rewards = run_random(env_name, min(n_eps, 500), env_kwargs)
            rand_time = time.time() - t0
            rand_avg = float(np.mean(rand_rewards[-100:]))
            print(f"    Random avg100: {rand_avg:.1f} ({rand_time:.1f}s)")

            # 2. Bare PPO
            print("  Running Bare PPO...")
            t0 = time.time()
            ppo_rewards = run_bare_ppo(env_name, n_eps, obs_type, env_kwargs)
            ppo_time = time.time() - t0
            ppo_avg = float(np.mean(ppo_rewards[-100:]))
            ppo_solved = None
            for ep in range(100, len(ppo_rewards)):
                if np.mean(ppo_rewards[ep-100:ep]) >= threshold:
                    ppo_solved = ep
                    break
            print(f"    PPO avg100: {ppo_avg:.1f} ({ppo_time:.1f}s) "
                  f"solved={'ep '+str(ppo_solved) if ppo_solved else 'no'}")

            # 3. ThrongletCell (with dreamer)
            print("  Running ThrongletCell (dreamer)...")
            t0 = time.time()
            cell_rewards, cell = run_cell(env_name, n_eps, obs_type, True, env_kwargs)
            cell_time = time.time() - t0
            cell_avg = float(np.mean(cell_rewards[-100:]))
            cell_solved = None
            for ep in range(100, len(cell_rewards)):
                if np.mean(cell_rewards[ep-100:ep]) >= threshold:
                    cell_solved = ep
                    break
            print(f"    Cell avg100: {cell_avg:.1f} ({cell_time:.1f}s) "
                  f"solved={'ep '+str(cell_solved) if cell_solved else 'no'} "
                  f"neurons={cell.neuron_count}")

            # Store results
            all_results[env_name] = {
                "threshold": threshold,
                "random": {"avg100": round(rand_avg, 2)},
                "bare_ppo": {
                    "avg100": round(ppo_avg, 2),
                    "solved_at": ppo_solved,
                    "time_s": round(ppo_time, 1),
                },
                "cell": {
                    "avg100": round(cell_avg, 2),
                    "solved_at": cell_solved,
                    "time_s": round(cell_time, 1),
                    "final_neurons": cell.neuron_count,
                    "growth_events": cell.growth_controller.stats()["grow_events"]
                    if cell.growth_controller else 0,
                    "prune_events": cell.growth_controller.stats()["prune_events"]
                    if cell.growth_controller else 0,
                    "wm_confidence": cell.world_model.confidence
                    if cell.world_model else 0,
                },
            }
        except Exception as e:
            print(f"  ERROR during {env_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[env_name] = {"error": True, "reason": str(e)}

        # Save partial results after each env
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  (partial results saved)")

    # ── Summary Table ────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("GAUNTLET RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Environment':<20} {'Random':>8} {'PPO':>8} {'Cell':>8} "
          f"{'PPO✓':>6} {'Cell✓':>6} {'Neurons':>8}")
    print("-" * 70)

    ppo_wins = 0
    cell_wins = 0
    completed = 0
    for name, r in all_results.items():
        if r.get("skipped") or r.get("error"):
            reason = r.get("reason", "unknown")[:40]
            print(f"{name:<20} {'SKIPPED':>8} {reason}")
            continue
        completed += 1
        ppo_solved = "✓" if r["bare_ppo"]["solved_at"] else "✗"
        cell_solved = "✓" if r["cell"]["solved_at"] else "✗"
        if r["bare_ppo"]["avg100"] > r["random"]["avg100"] + 5:
            ppo_wins += 1
        if r["cell"]["avg100"] > r["random"]["avg100"] + 5:
            cell_wins += 1
        print(f"{name:<20} {r['random']['avg100']:>8.1f} "
              f"{r['bare_ppo']['avg100']:>8.1f} "
              f"{r['cell']['avg100']:>8.1f} "
              f"{ppo_solved:>6} {cell_solved:>6} "
              f"{r['cell']['final_neurons']:>8}")

    print(f"\nCompleted: {completed}/{len(ENVS)} environments")
    print(f"PPO beats random in {ppo_wins}/{completed} envs")
    print(f"Cell beats random in {cell_wins}/{completed} envs")

    # Final save
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
