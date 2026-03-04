"""
52_snn_ab_test.py — Fast A/B comparison: SNN v1 (frozen) vs SNN v2 (resonant)

Target runtime: < 3 minutes total (200 episodes × 2 agents × CartPole)

Measures:
  - avg reward last 50 episodes
  - time per episode
  - SNN prediction error (v2 only)
  - step time breakdown

Run: python examples/52_snn_ab_test.py
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from src.cell.thronglet_cell import ThrongletCell

N_EPISODES = 300
REPORT_EVERY = 50


def run_cell(label, use_v2_snn, n_episodes=N_EPISODES):
    env = gym.make("CartPole-v1")
    obs_dim, n_actions = 4, 2

    cell = ThrongletCell(
        obs_dim=obs_dim,
        n_actions=n_actions,
        snn_neurons=64,
        compressed_dim=16,
        ppo_lr=3e-4,
        ppo_rollout_length=256,   # short: PPO updates every ~10 eps
        use_snn=True,
        use_dreamer=False,        # off — isolates SNN signal
        use_growth=False,         # off — isolates SNN signal
        use_v2_snn=use_v2_snn,
    )

    rewards = []
    step_times = []
    t_total = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        cell.reset()
        total, done = 0.0, False

        while not done:
            t0 = time.perf_counter()
            action = cell.step(obs)
            step_times.append(time.perf_counter() - t0)

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            done = terminated or truncated
            cell.learn(reward, done)
            total += reward

        rewards.append(total)

        if (ep + 1) % REPORT_EVERY == 0:
            avg = np.mean(rewards[-REPORT_EVERY:])
            elapsed = time.time() - t_total
            print(f"  [{label}] ep {ep+1:4d}  avg{REPORT_EVERY}={avg:6.1f}  "
                  f"elapsed={elapsed:.1f}s  step={np.mean(step_times)*1000:.2f}ms")
            step_times.clear()

    env.close()
    total_time = time.time() - t_total

    # SNN stats
    snn_stats = cell.snn.stats() if cell.snn else {}
    avg_pred_err = snn_stats.get("avg_pred_error", "n/a")
    pred_updates = snn_stats.get("pred_updates", "n/a")

    return {
        "label": label,
        "avg_last50": round(float(np.mean(rewards[-50:])), 1),
        "avg_last100": round(float(np.mean(rewards[-100:])), 1) if n_episodes >= 100 else "n/a",
        "max_reward": round(float(max(rewards)), 1),
        "time_s": round(total_time, 1),
        "pred_error": avg_pred_err,
        "pred_updates": pred_updates,
        "rewards": rewards,
    }


def main():
    print("=" * 60)
    print("SNN A/B TEST — CartPole-v1 (300 episodes)")
    print("=" * 60)
    print("  dreamer=OFF, growth=OFF, rollout=256 (PPO updates every ~10 eps)")
    print()

    # --- v1: frozen SNN (current) ---
    print("[v1] Frozen SNN (Fibonacci spiral, CSR, no learning)")
    r1 = run_cell("v1_frozen", use_v2_snn=False)

    print()
    # --- v2: resonant SNN (new) ---
    print("[v2] Resonant SNN (freq bands, dense numpy, prediction error)")
    r2 = run_cell("v2_resonant", use_v2_snn=True)

    # --- Summary ---
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'':30s} {'v1 frozen':>12} {'v2 resonant':>12}")
    print("-" * 60)
    print(f"{'Avg last 50 episodes':30s} {r1['avg_last50']:>12.1f} {r2['avg_last50']:>12.1f}")
    print(f"{'Avg last 100 episodes':30s} {str(r1['avg_last100']):>12} {str(r2['avg_last100']):>12}")
    print(f"{'Max episode reward':30s} {r1['max_reward']:>12.1f} {r2['max_reward']:>12.1f}")
    print(f"{'Total time (s)':30s} {r1['time_s']:>12.1f} {r2['time_s']:>12.1f}")
    print(f"{'SNN pred error':30s} {'n/a':>12} {str(r2['pred_error']):>12}")
    print(f"{'SNN pred updates':30s} {'n/a':>12} {str(r2['pred_updates']):>12}")

    delta = r2["avg_last50"] - r1["avg_last50"]
    faster = r1["time_s"] - r2["time_s"]
    print()
    if delta > 0:
        print(f"  v2 wins: +{delta:.1f} reward improvement")
    elif delta < 0:
        print(f"  v1 wins: v2 is {abs(delta):.1f} lower")
    else:
        print("  Tied on reward")

    if faster > 0:
        print(f"  v2 is {faster:.1f}s faster ({faster/r1['time_s']*100:.0f}% speedup)")
    else:
        print(f"  v1 is {abs(faster):.1f}s faster")

    # Quick learning curve (every 25 eps)
    print()
    print("Learning curve (avg per 25-ep window):")
    print(f"  {'Episode':>8}  {'v1':>8}  {'v2':>8}  {'winner':>8}")
    for i in range(0, N_EPISODES, 25):
        end = min(i + 25, N_EPISODES)
        a1 = np.mean(r1["rewards"][i:end])
        a2 = np.mean(r2["rewards"][i:end])
        winner = "v2" if a2 > a1 + 1 else ("v1" if a1 > a2 + 1 else "tie")
        print(f"  {end:>8d}  {a1:>8.1f}  {a2:>8.1f}  {winner:>8}")


if __name__ == "__main__":
    main()
