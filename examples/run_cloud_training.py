"""
run_cloud_training.py -- Full Mario RL training pipeline for cloud GPU.

Two-phase training:
  Phase 1: Fast ASCII training (PPO-ICM on ASCII simulator, ~14k steps/sec)
  Phase 2: Real game validation (gym-super-mario-bros at ~60fps)

The agent trains on ASCII, then proves it can play the real game unchanged.

Requirements (cloud):
  pip install torch gym-super-mario-bros nes-py

Usage:
  # Phase 1 only (no real game needed):
  python run_cloud_training.py --ascii-only --episodes 500

  # Full pipeline:
  python run_cloud_training.py --episodes 500

  # Real game only (load pretrained weights):
  python run_cloud_training.py --real-only --load checkpoint.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.games.mario.mario_adapter import MarioAdapter
from src.games.mario.mario_curriculum import MarioCurriculum
from src.games.mario.mario_icm_agent import MarioICMAgent


def save_agent(agent: MarioICMAgent, path: str):
    """Save agent weights to npz file."""
    np.savez(path,
        # Policy network
        w1=agent.w1, b1=agent.b1,
        w2=agent.w2, b2=agent.b2,
        w_pi=agent.w_pi, b_pi=agent.b_pi,
        w_v=agent.w_v, b_v=agent.b_v,
        # ICM encoder
        icm_enc_w1=agent.icm.enc_w1, icm_enc_b1=agent.icm.enc_b1,
        icm_enc_w2=agent.icm.enc_w2, icm_enc_b2=agent.icm.enc_b2,
        # ICM forward model
        icm_fwd_w1=agent.icm.fwd_w1, icm_fwd_b1=agent.icm.fwd_b1,
        icm_fwd_w2=agent.icm.fwd_w2, icm_fwd_b2=agent.icm.fwd_b2,
        # ICM inverse model
        icm_inv_w1=agent.icm.inv_w1, icm_inv_b1=agent.icm.inv_b1,
        icm_inv_w2=agent.icm.inv_w2, icm_inv_b2=agent.icm.inv_b2,
    )
    print(f"  Saved agent to {path}")


def load_agent(agent: MarioICMAgent, path: str):
    """Load agent weights from npz file."""
    data = np.load(path)
    # Policy
    agent.w1 = data['w1']; agent.b1 = data['b1']
    agent.w2 = data['w2']; agent.b2 = data['b2']
    agent.w_pi = data['w_pi']; agent.b_pi = data['b_pi']
    agent.w_v = data['w_v']; agent.b_v = data['b_v']
    # ICM
    agent.icm.enc_w1 = data['icm_enc_w1']; agent.icm.enc_b1 = data['icm_enc_b1']
    agent.icm.enc_w2 = data['icm_enc_w2']; agent.icm.enc_b2 = data['icm_enc_b2']
    agent.icm.fwd_w1 = data['icm_fwd_w1']; agent.icm.fwd_b1 = data['icm_fwd_b1']
    agent.icm.fwd_w2 = data['icm_fwd_w2']; agent.icm.fwd_b2 = data['icm_fwd_b2']
    agent.icm.inv_w1 = data['icm_inv_w1']; agent.icm.inv_b1 = data['icm_inv_b1']
    agent.icm.inv_w2 = data['icm_inv_w2']; agent.icm.inv_b2 = data['icm_inv_b2']
    print(f"  Loaded agent from {path}")


# ═══════════════════════════════════════════════════════════════
# PHASE 1: ASCII TRAINING
# ═══════════════════════════════════════════════════════════════

def train_ascii(agent: MarioICMAgent, episodes: int = 500,
                max_steps: int = 400, log_interval: int = 20,
                save_path: str = "mario_agent.npz"):
    """
    Fast training on ASCII simulator with ICM curiosity.
    ~14,000 steps/sec on CPU, much faster on GPU.
    """
    print("=" * 60)
    print("  PHASE 1: ASCII SIMULATOR TRAINING (PPO-ICM)")
    print(f"  Episodes: {episodes}, Max steps: {max_steps}")
    print("=" * 60)

    adapter = MarioAdapter()
    curriculum = MarioCurriculum(
        start_tier=1,
        advance_threshold=0.7,
        window_size=20,
        seed=42,
    )

    rewards = []
    wins = []
    t0 = time.perf_counter()
    total_steps = 0

    for ep in range(episodes):
        level = curriculum.next_level()
        obs = adapter.reset(level)
        agent.reset()
        ep_reward = 0.0

        for step in range(max_steps):
            action = agent.step(obs)
            next_obs, reward, done, info = adapter.step(action)
            ep_reward += reward
            total_steps += 1

            agent.learn_with_next_obs(reward, done, next_obs)
            obs = next_obs

            if done:
                break

        won = level.won
        progress = level.max_x_reached / max(1, level.width)
        rewards.append(ep_reward)
        wins.append(int(won))

        curriculum.record_result(won=won, progress=progress,
                                 steps=step+1, level=level)

        if curriculum.should_advance():
            old = curriculum.tier
            new = curriculum.advance()
            print(f"  >>> ADVANCED tier {old} -> {new}")

        if ep % log_interval == 0 or ep == episodes - 1:
            recent_r = rewards[-log_interval:]
            recent_w = wins[-log_interval:]
            elapsed = time.perf_counter() - t0
            sps = total_steps / max(0.01, elapsed)
            print(f"  Ep {ep:4d} | tier={curriculum.tier} "
                  f"| r={np.mean(recent_r):+6.2f} "
                  f"| win={np.mean(recent_w):.0%} "
                  f"| {sps:.0f} sps "
                  f"| {elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    print()
    print(f"  ASCII training: {episodes} ep, {total_steps} steps in {elapsed:.1f}s")
    print(f"  Final win rate (last 20): {np.mean(wins[-20:]):.0%}")
    print(f"  Final avg reward (last 20): {np.mean(rewards[-20:]):+.2f}")
    print(f"  Max tier: {max(curriculum.tier for _ in [1])}")

    save_agent(agent, save_path)
    return rewards, wins


# ═══════════════════════════════════════════════════════════════
# PHASE 2: REAL GAME VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_real_game(agent: MarioICMAgent, episodes: int = 20,
                       max_steps: int = 2000, render: bool = False):
    """
    Test the ASCII-trained agent on the real NES game.

    The agent sees pixel frames converted to ASCII — same observation
    format it trained on. This proves sim-to-real transfer.
    """
    from src.games.mario.mario_real_adapter import MarioRealAdapter

    print()
    print("=" * 60)
    print("  PHASE 2: REAL GAME VALIDATION (gym-super-mario-bros)")
    print(f"  Episodes: {episodes}, Max steps: {max_steps}")
    print("=" * 60)

    render_mode = "human" if render else None
    adapter = MarioRealAdapter(render_mode=render_mode)

    rewards = []
    distances = []

    for ep in range(episodes):
        obs = adapter.reset()
        agent.reset()
        ep_reward = 0.0

        for step in range(max_steps):
            action = agent.step(obs)
            obs, reward, done, info = adapter.step(action)
            ep_reward += reward

            if done:
                break

        rewards.append(ep_reward)
        x_pos = info.get("x_pos", 0)
        distances.append(x_pos)

        print(f"  Ep {ep:3d} | reward={ep_reward:+7.1f} "
              f"| x_pos={x_pos:5d} "
              f"| steps={step+1} "
              f"| {'WIN' if info.get('flag_get', False) else 'died'}")

    adapter.close()

    print()
    print(f"  Real game: {episodes} episodes")
    print(f"  Avg reward: {np.mean(rewards):+.1f}")
    print(f"  Avg distance: {np.mean(distances):.0f}")
    print(f"  Max distance: {max(distances)}")
    print(f"  Wins: {sum(1 for r in rewards if r > 3000)}/{episodes}")
    print("=" * 60)

    return rewards, distances


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mario RL Cloud Training")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Training episodes for ASCII phase")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Max steps per episode")
    parser.add_argument("--ascii-only", action="store_true",
                        help="Skip real game validation")
    parser.add_argument("--real-only", action="store_true",
                        help="Skip ASCII training, load weights")
    parser.add_argument("--load", type=str, default=None,
                        help="Load pretrained weights")
    parser.add_argument("--save", type=str, default="mario_agent.npz",
                        help="Save path for trained weights")
    parser.add_argument("--render", action="store_true",
                        help="Render real game (requires display)")
    parser.add_argument("--val-episodes", type=int, default=20,
                        help="Real game validation episodes")
    # Agent hyperparams
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--icm-lambda", type=float, default=0.3,
                        help="Intrinsic reward scaling")
    args = parser.parse_args()

    print("=" * 60)
    print("  MARIO ASCII RL -- CLOUD TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Mode: {'ASCII only' if args.ascii_only else 'Real only' if args.real_only else 'Full pipeline'}")
    print(f"  Episodes: {args.episodes}")
    print(f"  ICM lambda: {args.icm_lambda}")
    print()

    # Create agent
    agent = MarioICMAgent(
        obs_dim=378,
        n_actions=6,
        hidden1=128,
        hidden2=64,
        lr=args.lr,
        gamma=0.99,
        rollout_length=128,
        icm_feature_dim=32,
        icm_hidden_dim=64,
        icm_lr=1e-3,
        intrinsic_lambda=args.icm_lambda,
    )

    if args.load:
        load_agent(agent, args.load)

    # Phase 1: ASCII training
    if not args.real_only:
        train_ascii(agent, episodes=args.episodes,
                    max_steps=args.max_steps,
                    save_path=args.save)

    # Phase 2: Real game validation
    if not args.ascii_only:
        try:
            validate_real_game(agent, episodes=args.val_episodes,
                               render=args.render)
        except ImportError as e:
            print(f"\n  ⚠ Real game skipped: {e}")
            print("  Install: pip install gym-super-mario-bros nes-py")

    print()
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
