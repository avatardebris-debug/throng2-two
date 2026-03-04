"""
52_snn_ab_test.py — Proper 3-way ablation: isolating SNN changes one at a time

Variables being tested:
  v1   = CSR sparse + Fibonacci spiral + NO learning   (current)
  v1b  = Dense numpy + Freq bands     + NO learning   (structure change only)
  v2   = Dense numpy + Freq bands     + YES learning  (structure + learning)

Isolates:
  v1  vs v1b  → effect of dense matrix + freq band structure
  v1b vs v2   → effect of prediction error learning alone

Target runtime: ~5 minutes total (300 eps × 3 agents)
Run: python examples/52_snn_ab_test.py
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from src.cell.thronglet_cell import ThrongletCell

N_EPISODES = 300
REPORT_EVERY = 100


def run_cell(label, use_v2_snn, frozen_v2=False, n_episodes=N_EPISODES):
    """
    frozen_v2=True: use ResonantSNN with learning_rate=0 (structure only, no learning)
    """
    env = gym.make("CartPole-v1")
    obs_dim, n_actions = 4, 2

    cell = ThrongletCell(
        obs_dim=obs_dim,
        n_actions=n_actions,
        snn_neurons=64,
        compressed_dim=16,
        ppo_lr=3e-4,
        ppo_rollout_length=256,
        use_snn=True,
        use_dreamer=False,
        use_growth=False,
        use_v2_snn=use_v2_snn,
    )

    # Override learning rate for v1b (freeze prediction weights)
    if frozen_v2 and use_v2_snn:
        cell.snn.learning_rate = 0.0

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
            ms = np.mean(step_times) * 1000
            print(f"  [{label}] ep {ep+1:4d}  avg{REPORT_EVERY}={avg:6.1f}  "
                  f"elapsed={elapsed:.1f}s  step={ms:.2f}ms")
            step_times.clear()

    env.close()
    total_time = time.time() - t_total
    snn_stats = cell.snn.stats() if cell.snn else {}

    return {
        "label": label,
        "avg_last50":  round(float(np.mean(rewards[-50:])), 1),
        "avg_last100": round(float(np.mean(rewards[-100:])), 1),
        "max_reward":  round(float(max(rewards)), 1),
        "time_s":      round(total_time, 1),
        "pred_error":  snn_stats.get("avg_pred_error", "n/a"),
        "pred_updates": snn_stats.get("pred_updates", "n/a"),
        "rewards":     rewards,
    }


def main():
    print("=" * 65)
    print("SNN 3-WAY ABLATION — CartPole-v1 (300 eps, rollout=256)")
    print("=" * 65)
    print("  v1  = CSR + Fibonacci    + no learning  (current)")
    print("  v1b = Dense + Freq bands + no learning  (structure only)")
    print("  v2  = Dense + Freq bands + prediction error  (all changes)")
    print()

    print("[v1]  Frozen SNN — CSR sparse, Fibonacci spiral")
    r1  = run_cell("v1_frozen",   use_v2_snn=False)

    print()
    print("[v1b] Frozen resonant — dense+bands, NO learning")
    r1b = run_cell("v1b_struct",  use_v2_snn=True,  frozen_v2=True)

    print()
    print("[v2]  Resonant + learning — dense+bands + prediction error")
    r2  = run_cell("v2_learning", use_v2_snn=True,  frozen_v2=False)

    # --- Summary ---
    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    fmt = f"{{:32s}} {{:>10}} {{:>10}} {{:>10}}"
    print(fmt.format("", "v1", "v1b", "v2"))
    print("-" * 65)
    def row(label, k):
        return fmt.format(label, str(r1[k]), str(r1b[k]), str(r2[k]))
    print(row("Avg last 50 episodes",  "avg_last50"))
    print(row("Avg last 100 episodes", "avg_last100"))
    print(row("Max reward",            "max_reward"))
    print(row("Total time (s)",        "time_s"))
    print(row("SNN pred error",        "pred_error"))
    print(row("SNN pred updates",      "pred_updates"))

    print()
    print("Variable isolation:")
    struct_delta = r1b["avg_last100"] - r1["avg_last100"]
    learn_delta  = r2["avg_last100"]  - r1b["avg_last100"]
    total_delta  = r2["avg_last100"]  - r1["avg_last100"]
    print(f"  Structure effect  (v1 → v1b):  {struct_delta:+.1f} avg100")
    print(f"  Learning effect   (v1b → v2):  {learn_delta:+.1f} avg100")
    print(f"  Total effect      (v1 → v2):   {total_delta:+.1f} avg100")

    # Learning curve
    print()
    print("Learning curve (25-ep windows):")
    print(f"  {'Episode':>8}  {'v1':>8}  {'v1b':>8}  {'v2':>8}")
    for i in range(0, N_EPISODES, 25):
        end = min(i + 25, N_EPISODES)
        a1  = np.mean(r1["rewards"][i:end])
        a1b = np.mean(r1b["rewards"][i:end])
        a2  = np.mean(r2["rewards"][i:end])
        best = max(a1, a1b, a2)
        def mark(v): return f"{v:8.1f}*" if v == best and best > min(a1,a1b,a2)+2 else f"{v:8.1f} "
        print(f"  {end:>8d}  {mark(a1)} {mark(a1b)} {mark(a2)}")


if __name__ == "__main__":
    main()
