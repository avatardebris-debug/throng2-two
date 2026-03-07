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
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.games.mario.mario_adapter import MarioAdapter
from src.games.mario.mario_curriculum import MarioCurriculum
from src.games.mario.mario_icm_agent import MarioICMAgent
from src.games.mario.mario_ghost import GhostRacer
from src.games.mario.mario_selfplay import DualRacer

# Auto-detect PyTorch for GPU acceleration
try:
    import torch
    from src.games.mario.mario_torch_agent import MarioTorchAgent
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def save_agent(agent, path: str):
    """Save agent weights (auto-detects torch vs numpy agent)."""
    if isinstance(agent, MarioICMAgent):
        np.savez(path,
            w1=agent.w1, b1=agent.b1,
            w2=agent.w2, b2=agent.b2,
            w_pi=agent.w_pi, b_pi=agent.b_pi,
            w_v=agent.w_v, b_v=agent.b_v,
            icm_enc_w1=agent.icm.enc_w1, icm_enc_b1=agent.icm.enc_b1,
            icm_enc_w2=agent.icm.enc_w2, icm_enc_b2=agent.icm.enc_b2,
            icm_fwd_w1=agent.icm.fwd_w1, icm_fwd_b1=agent.icm.fwd_b1,
            icm_fwd_w2=agent.icm.fwd_w2, icm_fwd_b2=agent.icm.fwd_b2,
            icm_inv_w1=agent.icm.inv_w1, icm_inv_b1=agent.icm.inv_b1,
            icm_inv_w2=agent.icm.inv_w2, icm_inv_b2=agent.icm.inv_b2,
        )
    else:
        # Torch agent
        pt_path = path.replace('.npz', '.pt')
        agent.save(pt_path)
        path = pt_path
    print(f"  Saved agent to {path}")


def load_agent(agent, path: str):
    """Load agent weights (auto-detects format)."""
    if isinstance(agent, MarioICMAgent):
        data = np.load(path)
        agent.w1 = data['w1']; agent.b1 = data['b1']
        agent.w2 = data['w2']; agent.b2 = data['b2']
        agent.w_pi = data['w_pi']; agent.b_pi = data['b_pi']
        agent.w_v = data['w_v']; agent.b_v = data['b_v']
        agent.icm.enc_w1 = data['icm_enc_w1']; agent.icm.enc_b1 = data['icm_enc_b1']
        agent.icm.enc_w2 = data['icm_enc_w2']; agent.icm.enc_b2 = data['icm_enc_b2']
        agent.icm.fwd_w1 = data['icm_fwd_w1']; agent.icm.fwd_b1 = data['icm_fwd_b1']
        agent.icm.fwd_w2 = data['icm_fwd_w2']; agent.icm.fwd_b2 = data['icm_fwd_b2']
        agent.icm.inv_w1 = data['icm_inv_w1']; agent.icm.inv_b1 = data['icm_inv_b1']
        agent.icm.inv_w2 = data['icm_inv_w2']; agent.icm.inv_b2 = data['icm_inv_b2']
    else:
        agent.load(path)
    print(f"  Loaded agent from {path}")


# ═══════════════════════════════════════════════════════════════
# PHASE 1: ASCII TRAINING
# ═══════════════════════════════════════════════════════════════

def train_ascii(agent, episodes: int = 500,
                max_steps: int = 400, log_interval: int = 20,
                save_path: str = "mario_agent.npz",
                checkpoint_interval: int = 100,
                report_path: str = "training_report.json",
                use_ghost: bool = False,
                use_selfplay: bool = False):
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
    tier_history = []
    checkpoints = []
    t0 = time.perf_counter()
    total_steps = 0

    # Ghost racing
    ghost = GhostRacer() if use_ghost else None
    if ghost:
        print("  Ghost racing: ENABLED")

    # Self-play racing
    racer = DualRacer() if use_selfplay else None
    if racer:
        print("  Self-play racing: ENABLED (dual rollouts)")

    for ep in range(episodes):
        level = curriculum.next_level()

        # ── Self-play mode: two rollouts on same level ────────
        if racer:
            import copy
            adapter_b = MarioAdapter()
            level_b = copy.deepcopy(level)
            race_scale = 0.005  # small per-step reward

            # Rollout A — record x positions for comparison
            obs = adapter.reset(level)
            agent.reset()
            rollout_a_reward = 0.0
            a_positions = []
            for step in range(max_steps):
                a_positions.append(level.mario_col)
                action = agent.step(obs)
                next_obs, reward, done, info = adapter.step(action)
                rollout_a_reward += reward
                total_steps += 1
                agent.learn_with_next_obs(reward, done, next_obs)
                obs = next_obs
                if done:
                    break
            a_x = level.max_x_reached
            a_won = level.won
            a_steps = step + 1

            # Rollout B — per-step comparison to A's recorded positions
            obs = adapter_b.reset(level_b)
            agent.reset()
            rollout_b_reward = 0.0
            b_race_total = 0.0
            for step in range(max_steps):
                action = agent.step(obs)
                next_obs, reward, done, info = adapter_b.step(action)

                # Per-step race reward: +0.005 if ahead, -0.005 if behind
                a_x_at_step = a_positions[step] if step < len(a_positions) else a_x
                b_x_at_step = level_b.mario_col
                if b_x_at_step > a_x_at_step:
                    reward += race_scale
                    b_race_total += race_scale
                elif b_x_at_step < a_x_at_step:
                    reward -= race_scale
                    b_race_total -= race_scale

                rollout_b_reward += reward
                total_steps += 1
                agent.learn_with_next_obs(reward, done, next_obs)
                obs = next_obs
                if done:
                    break
            b_x = level_b.max_x_reached
            b_won = level_b.won

            # A gets symmetric correction (mirror of B's race reward)
            rollout_a_reward -= b_race_total

            # Report best of the two runs
            ep_reward = max(rollout_a_reward, rollout_b_reward)
            won = a_won or b_won
            progress = max(a_x, b_x) / max(1, level.width)

        # ── Normal mode: single rollout ───────────────────────
        else:
            obs = adapter.reset(level)
            agent.reset()
            ep_reward = 0.0

            # Start ghost race
            if ghost:
                ghost.start_race(tier=curriculum.tier, level_width=level.width)

            for step in range(max_steps):
                action = agent.step(obs)
                next_obs, reward, done, info = adapter.step(action)

                # Add ghost-shaped reward
                if ghost:
                    ghost_r = ghost.compute_reward(level.mario_col, step)
                    reward += ghost_r

                ep_reward += reward
                total_steps += 1

                agent.learn_with_next_obs(reward, done, next_obs)
                obs = next_obs

                if done:
                    break

            # End ghost race
            if ghost:
                ghost_info = ghost.end_race(
                    won=level.won,
                    final_x=level.max_x_reached,
                    total_steps=step + 1,
                )
                ep_reward += ghost_info.get("bonus", 0.0)

            won = level.won
            progress = level.max_x_reached / max(1, level.width)

        rewards.append(ep_reward)
        wins.append(int(won))
        tier_history.append(curriculum.tier)

        curriculum.record_result(won=won, progress=progress,
                                 steps=step+1, level=level)

        if curriculum.should_advance():
            old = curriculum.tier
            new = curriculum.advance()
            print(f"  >>> ADVANCED tier {old} -> {new}")

        # Periodic checkpoint
        if checkpoint_interval and (ep + 1) % checkpoint_interval == 0:
            ckpt_name = f"mario_agent_ep{ep+1}.npz"
            save_agent(agent, ckpt_name)
            checkpoints.append(ckpt_name)

        if ep % log_interval == 0 or ep == episodes - 1:
            recent_r = rewards[-log_interval:]
            recent_w = wins[-log_interval:]
            elapsed = time.perf_counter() - t0
            sps = total_steps / max(0.01, elapsed)
            ghost_str = ""
            if ghost and ghost.has_ghost(curriculum.tier):
                gs = ghost.ghost_stats(curriculum.tier)
                ghost_str = f" | ghost_beaten={gs['ghost_beaten']}/{gs['races']}"
            if racer:
                ghost_str = f" | races={racer.total_races} margin={racer.avg_margin:.1f}"
            print(f"  Ep {ep:4d} | tier={curriculum.tier} "
                  f"| r={np.mean(recent_r):+6.2f} "
                  f"| win={np.mean(recent_w):.0%} "
                  f"| {sps:.0f} sps "
                  f"| {elapsed:.0f}s{ghost_str}")

    elapsed = time.perf_counter() - t0
    final_win = float(np.mean(wins[-20:]))
    final_reward = float(np.mean(rewards[-20:]))
    max_tier = max(tier_history) if tier_history else 1
    sps = total_steps / max(0.01, elapsed)

    print()
    print(f"  ASCII training: {episodes} ep, {total_steps} steps in {elapsed:.1f}s")
    print(f"  Final win rate (last 20): {final_win:.0%}")
    print(f"  Final avg reward (last 20): {final_reward:+.2f}")
    print(f"  Max tier: {max_tier}")

    # Save final weights
    save_agent(agent, save_path)
    checkpoints.append(save_path)

    # Export training report
    report = {
        "timestamp": datetime.now().isoformat(),
        "episodes": episodes,
        "total_steps": total_steps,
        "training_time_sec": round(elapsed, 1),
        "steps_per_sec": round(sps, 0),
        "final_win_rate": round(final_win, 3),
        "final_avg_reward": round(final_reward, 2),
        "max_tier_reached": max_tier,
        "rewards": [round(r, 2) for r in rewards],
        "wins": wins,
        "tier_history": tier_history,
        "checkpoints": checkpoints,
        "hyperparams": {
            "lr": getattr(agent, '_lr', getattr(agent, 'policy_optimizer', None) and agent.policy_optimizer.param_groups[0]['lr'] if hasattr(agent, 'policy_optimizer') else 0),
            "gamma": agent.gamma,
            "rollout_length": agent.rollout_length,
            "icm_lambda": agent.intrinsic_lambda,
            "backend": "torch" if hasattr(agent, 'policy') else "numpy",
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

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
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N episodes (0=disabled)")
    parser.add_argument("--report", type=str, default="training_report.json",
                        help="Path for JSON training report")
    parser.add_argument("--force-numpy", action="store_true",
                        help="Use numpy agent even if torch is available")
    parser.add_argument("--ghost", action="store_true",
                        help="Enable ghost racing self-play")
    parser.add_argument("--selfplay", action="store_true",
                        help="Enable dual-rollout self-play racing")
    args = parser.parse_args()

    print("=" * 60)
    print("  MARIO ASCII RL -- CLOUD TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Mode: {'ASCII only' if args.ascii_only else 'Real only' if args.real_only else 'Full pipeline'}")
    print(f"  Episodes: {args.episodes}")
    print(f"  ICM lambda: {args.icm_lambda}")
    print()

    # Create agent — auto-detect torch
    use_torch = HAS_TORCH and not args.force_numpy
    if use_torch:
        print(f"  Backend: PyTorch (GPU: {torch.cuda.is_available()})")
        agent = MarioTorchAgent(
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
    else:
        print("  Backend: NumPy (CPU only)")
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
    print()

    if args.load:
        load_agent(agent, args.load)

    # Phase 1: ASCII training
    if not args.real_only:
        rewards, wins = train_ascii(
            agent, episodes=args.episodes,
            max_steps=args.max_steps,
            save_path=args.save,
            checkpoint_interval=args.checkpoint_interval,
            report_path=args.report,
            use_ghost=args.ghost,
            use_selfplay=args.selfplay,
        )

    # Phase 2: Real game validation
    if not args.ascii_only:
        try:
            validate_real_game(agent, episodes=args.val_episodes,
                               render=args.render)
        except ImportError as e:
            print(f"\n  ⚠ Real game skipped: {e}")
            print("  Install: pip install gym-super-mario-bros nes-py")

    # Suggest git push
    print()
    print("  Done! To push results back to GitHub:")
    print("    git add mario_agent*.pt mario_agent*.npz training_report.json")
    print("    git commit -m 'Training results: %(ep)s episodes'" % {"ep": args.episodes})
    print("    git push origin main")
    print("=" * 60)


if __name__ == "__main__":
    main()
