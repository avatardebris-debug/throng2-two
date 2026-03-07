"""
run_mario_training.py -- Train ThrongletCell on Mario ASCII with curriculum.

Connects the existing throng2 RL pipeline to the Mario ASCII simulator:
  ThrongletCell (Encoder + SNN + PPO + WorldModel + Dreamer)
    ↕ MarioAdapter (gymnasium interface)
    ↕ MarioCurriculum (tiered level generation + GAN)

Usage:
    python examples/run_mario_training.py
    python examples/run_mario_training.py --episodes 500 --no-snn
    python examples/run_mario_training.py --use-v2-snn --episodes 1000
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.games.mario.mario_simulator import MarioSimulator, Action
from src.games.mario.mario_adapter import MarioAdapter
from src.games.mario.mario_curriculum import MarioCurriculum
from src.cell.thronglet_cell import ThrongletCell


def parse_args():
    p = argparse.ArgumentParser(description="Mario ASCII RL Training")
    p.add_argument("--episodes", type=int, default=200, help="Total training episodes")
    p.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    p.add_argument("--start-tier", type=int, default=1, help="Starting difficulty tier")
    p.add_argument("--advance-threshold", type=float, default=0.7, help="Win rate to advance tier")
    p.add_argument("--window-size", type=int, default=30, help="Window for advancement check")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-snn", action="store_true", help="Disable SNN features")
    p.add_argument("--no-dreamer", action="store_true", help="Disable WorldModel dreaming")
    p.add_argument("--use-v2-snn", action="store_true", help="Use ResonantSNN v2")
    p.add_argument("--snn-neurons", type=int, default=64, help="SNN neuron count")
    p.add_argument("--ppo-hidden", type=int, default=64, help="PPO hidden layer size")
    p.add_argument("--save-path", type=str, default=None, help="Save checkpoint path")
    p.add_argument("--log-interval", type=int, default=10, help="Episodes between log prints")
    p.add_argument("--gan-interval", type=int, default=50, help="Episodes between GAN training")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("    MARIO ASCII -- THRONGLET CELL TRAINING")
    print("=" * 70)

    # ── Setup ────────────────────────────────────────────────
    adapter = MarioAdapter()
    curriculum = MarioCurriculum(
        start_tier=args.start_tier,
        advance_threshold=args.advance_threshold,
        window_size=args.window_size,
        seed=args.seed,
    )

    # ThrongletCell dimensions:
    #   obs_dim = MarioAdapter obs_dim (378)
    #   n_actions = 6 (NOOP, LEFT, RIGHT, JUMP, JUMP_LEFT, JUMP_RIGHT)
    obs_dim = adapter.obs_dim
    n_actions = adapter.n_actions

    cell = ThrongletCell(
        obs_dim=obs_dim,
        n_actions=n_actions,
        snn_neurons=args.snn_neurons,
        compressed_dim=16,
        ppo_hidden=args.ppo_hidden,
        ppo_lr=3e-4,
        ppo_rollout_length=128,
        use_snn=not args.no_snn,
        use_dreamer=not args.no_dreamer,
        use_v2_snn=args.use_v2_snn,
    )

    print(f"  Obs dim: {obs_dim}")
    print(f"  Actions: {n_actions}")
    print(f"  SNN: {'OFF' if args.no_snn else f'ON ({args.snn_neurons} neurons)'}")
    print(f"  Dreamer: {'OFF' if args.no_dreamer else 'ON'}")
    print(f"  Device: {cell.ppo.device}")
    print(f"  Start tier: {args.start_tier}")
    print(f"  Episodes: {args.episodes}")
    print()

    # ── Training loop ────────────────────────────────────────
    best_avg_reward = -float("inf")
    episode_rewards = []
    episode_wins = []
    tier_log = []
    t_start = time.perf_counter()

    for ep in range(args.episodes):
        # Get level from curriculum
        level = curriculum.next_level()
        obs = adapter.reset(level)
        cell.reset()

        ep_reward = 0.0
        ep_steps = 0

        for step in range(args.max_steps):
            # ThrongletCell selects action
            action = cell.step(obs)

            # Environment step
            obs, reward, done, info = adapter.step(action)
            ep_reward += reward
            ep_steps += 1

            # Learn from transition
            update_stats = cell.learn(reward, done)

            if done:
                break

        # Record results
        won = level.won
        progress = level.max_x_reached / max(1, level.width)
        episode_rewards.append(ep_reward)
        episode_wins.append(int(won))
        tier_log.append(curriculum.tier)

        # Feed curriculum
        curriculum.record_result(
            won=won, progress=progress,
            steps=ep_steps, level=level,
        )

        # Check tier advancement
        if curriculum.should_advance():
            old_tier = curriculum.tier
            new_tier = curriculum.advance()
            print(f"  >>> TIER {old_tier} -> {new_tier} "
                  f"(ep {ep}, win_rate={np.mean(episode_wins[-args.window_size:]):.2f})")

        # GAN training
        if ep > 0 and ep % args.gan_interval == 0:
            gan_result = curriculum.train_gan()

        # Logging
        if ep % args.log_interval == 0 or ep == args.episodes - 1:
            recent_rewards = episode_rewards[-args.log_interval:]
            recent_wins = episode_wins[-args.log_interval:]
            avg_r = float(np.mean(recent_rewards))
            win_rate = float(np.mean(recent_wins))
            elapsed = time.perf_counter() - t_start

            cell_stats = cell.stats()
            ppo_loss = cell_stats.get("ppo", {}).get("avg_loss", 0)

            print(f"  Ep {ep:4d} | tier={curriculum.tier} "
                  f"| avg_r={avg_r:+7.2f} | win={win_rate:.0%} "
                  f"| ppo_loss={ppo_loss:.4f} "
                  f"| neurons={cell.neuron_count} "
                  f"| {elapsed:.0f}s")

            if avg_r > best_avg_reward:
                best_avg_reward = avg_r

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    total_steps = cell._total_steps

    print()
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Episodes: {args.episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Wall time: {elapsed:.1f}s ({total_steps/elapsed:,.0f} steps/sec)")
    print(f"  Final tier: {curriculum.tier}")
    print(f"  Best avg reward: {best_avg_reward:+.2f}")
    print(f"  Final win rate (last {args.window_size}): "
          f"{np.mean(episode_wins[-args.window_size:]):.0%}")

    # Per-tier breakdown
    print()
    cstats = curriculum.report()
    for tier, ts in cstats.get("tier_stats", {}).items():
        print(f"  Tier {tier}: {ts['episodes']} eps, "
              f"win_rate={ts['win_rate']:.0%}, "
              f"avg_progress={ts['avg_progress']:.2f}")

    # Cell stats
    print()
    cs = cell.stats()
    print(f"  Cell: {cs['total_episodes']} eps, {cs['total_steps']} steps")
    print(f"    Encoder: {cs['encoder']}")
    print(f"    PPO: {cs['ppo']}")
    if "snn" in cs:
        print(f"    SNN: {cs['snn']}")
    if "world_model" in cs:
        print(f"    WorldModel: {cs['world_model']}")

    # Save checkpoint
    if args.save_path:
        cell.save(args.save_path)
        print(f"\n  Checkpoint saved to {args.save_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
