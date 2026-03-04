"""
53_snn_ablation_v2.py — 4-way ablation with proper variable isolation

Variants:
  ppo_only    = No SNN at all, pure PPO baseline
  v1_frozen   = CSR + Fibonacci, frozen weights (prior winner)
  v2_frozen   = Dense + Freq bands, frozen weights (structure only)
  v2_episodic = Dense + Freq bands, learns between episodes only

Isolates:
  ppo_only vs v1_frozen   → does SNN help at all?
  v1_frozen vs v2_frozen  → CSR/Fibonacci vs Dense/FreqBands (structure)
  v2_frozen vs v2_episodic→ does episodic prediction learning help?

Target runtime: ~7 minutes (300 eps × 4 agents)
Run: python examples/53_snn_ablation_v2.py
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from src.cell.thronglet_cell import ThrongletCell

N_EPISODES = 300
REPORT_EVERY = 100


def run_cell(label, use_snn, use_v2_snn=False, learn_mode="frozen", n_episodes=N_EPISODES):
    env = gym.make("CartPole-v1")
    obs_dim, n_actions = 4, 2

    cell = ThrongletCell(
        obs_dim=obs_dim,
        n_actions=n_actions,
        snn_neurons=64,
        compressed_dim=16,
        ppo_lr=3e-4,
        ppo_rollout_length=256,
        use_snn=use_snn,
        use_dreamer=False,
        use_growth=False,
        use_v2_snn=use_v2_snn,
    )

    # Set learn mode for v2 SNN
    if use_v2_snn and use_snn:
        cell.snn.learn_mode = learn_mode
        if learn_mode == "frozen":
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
            print(f"  [{label:12s}] ep {ep+1:4d}  avg{REPORT_EVERY}={avg:6.1f}  "
                  f"elapsed={elapsed:.1f}s  step={ms:.2f}ms")
            step_times.clear()

    env.close()
    total_time = time.time() - t_total
    snn_stats = cell.snn.stats() if cell.snn else {}

    return {
        "label": label,
        "avg_last50":   round(float(np.mean(rewards[-50:])), 1),
        "avg_last100":  round(float(np.mean(rewards[-100:])), 1),
        "max_reward":   round(float(max(rewards)), 1),
        "time_s":       round(total_time, 1),
        "pred_error":   snn_stats.get("avg_pred_error", "n/a"),
        "pred_updates": snn_stats.get("pred_updates", "n/a"),
        "rewards":      rewards,
    }


def main():
    print("=" * 70)
    print("SNN 4-WAY ABLATION — CartPole-v1 (300 eps, rollout=256)")
    print("=" * 70)
    print("  ppo_only    = No SNN, pure PPO baseline")
    print("  v1_frozen   = CSR + Fibonacci, frozen (prior winner)")
    print("  v2_frozen   = Dense + Freq bands, frozen")
    print("  v2_episodic = Dense + Freq bands, learns between episodes")
    print()

    configs = [
        ("ppo_only",    dict(use_snn=False)),
        ("v1_frozen",   dict(use_snn=True, use_v2_snn=False)),
        ("v2_frozen",   dict(use_snn=True, use_v2_snn=True, learn_mode="frozen")),
        ("v2_episodic", dict(use_snn=True, use_v2_snn=True, learn_mode="episodic")),
    ]

    results = {}
    for label, kwargs in configs:
        desc = {
            "ppo_only":    "No SNN — pure PPO baseline",
            "v1_frozen":   "CSR + Fibonacci, frozen (prior winner)",
            "v2_frozen":   "Dense + Freq bands, frozen",
            "v2_episodic": "Dense + Freq bands, episodic prediction learning",
        }[label]
        print(f"[{label}]  {desc}")
        results[label] = run_cell(label, **kwargs)
        print()

    r = results

    # --- Summary ---
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    labels = ["ppo_only", "v1_frozen", "v2_frozen", "v2_episodic"]
    header = f"{'':28s}" + "".join(f"{l:>12s}" for l in labels)
    print(header)
    print("-" * 70)

    for key, title in [
        ("avg_last50",  "Avg last 50 episodes"),
        ("avg_last100", "Avg last 100 episodes"),
        ("max_reward",  "Max reward"),
        ("time_s",      "Total time (s)"),
        ("pred_error",  "SNN pred error"),
        ("pred_updates","SNN pred updates"),
    ]:
        row = f"{title:28s}" + "".join(f"{str(r[l][key]):>12s}" for l in labels)
        print(row)

    # Variable isolation
    print()
    print("Variable isolation (avg last 100):")
    d1 = r["v1_frozen"]["avg_last100"]  - r["ppo_only"]["avg_last100"]
    d2 = r["v2_frozen"]["avg_last100"]  - r["v1_frozen"]["avg_last100"]
    d3 = r["v2_episodic"]["avg_last100"] - r["v2_frozen"]["avg_last100"]
    print(f"  SNN value         (ppo → v1_frozen):   {d1:+.1f}")
    print(f"  Structure effect  (v1 → v2_frozen):    {d2:+.1f}")
    print(f"  Episodic learning (v2_frozen → v2_ep): {d3:+.1f}")

    # Learning curve
    print()
    print("Learning curve (50-ep windows):")
    print(f"  {'Episode':>8}" + "".join(f"  {l:>12s}" for l in labels))
    for i in range(0, N_EPISODES, 50):
        end = min(i + 50, N_EPISODES)
        vals = [np.mean(r[l]["rewards"][i:end]) for l in labels]
        best = max(vals)
        def fmt(v): return f"{v:12.1f}*" if v == best and best > min(vals)+3 else f"{v:12.1f} "
        print(f"  {end:>8d}" + "".join(fmt(v) for v in vals))

    # Winner
    print()
    avg100s = {l: r[l]["avg_last100"] for l in labels}
    winner = max(avg100s, key=avg100s.get)
    print(f"  Winner: {winner} ({avg100s[winner]:.1f} avg100)")


if __name__ == "__main__":
    main()
