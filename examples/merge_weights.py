"""
merge_weights.py -- Merge multiple Mario agent checkpoints.

Supports three merge strategies:
  1. Average (model soup): average all weight tensors
  2. Best-of: pick the checkpoint with highest eval score
  3. SLERP: spherical interpolation between two checkpoints

Usage:
  # Average two checkpoints:
  python examples/merge_weights.py --inputs agent_a.pt agent_b.pt --output merged.pt

  # Average with custom weights (70% A, 30% B):
  python examples/merge_weights.py --inputs agent_a.pt agent_b.pt --weights 0.7 0.3 --output merged.pt

  # SLERP between two (alpha=0.5 = midpoint):
  python examples/merge_weights.py --inputs agent_a.pt agent_b.pt --strategy slerp --alpha 0.5 --output merged.pt

  # Evaluate all checkpoints and pick best:
  python examples/merge_weights.py --inputs agent_a.pt agent_b.pt agent_c.pt --strategy best --eval-episodes 50 --output best.pt
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def load_checkpoint(path):
    """Load a .pt checkpoint."""
    data = torch.load(path, map_location="cpu")
    print(f"  Loaded: {path}")
    if "config" in data:
        cfg = data["config"]
        print(f"    obs_dim={cfg.get('obs_dim')}, n_actions={cfg.get('n_actions')}, "
              f"gamma={cfg.get('gamma')}")
    return data


def average_merge(checkpoints, weights=None):
    """
    Average weight tensors across checkpoints.

    This is "model soup" -- shown by Google to improve generalization
    when averaging checkpoints from different training runs.
    """
    n = len(checkpoints)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    print(f"  Merging {n} checkpoints with weights: "
          f"{[round(w, 3) for w in weights]}")

    merged = {}
    # Merge policy weights
    ref_policy = checkpoints[0]["policy"]
    merged_policy = {}
    for key in ref_policy:
        merged_policy[key] = sum(
            w * cp["policy"][key].float()
            for w, cp in zip(weights, checkpoints)
        )
    merged["policy"] = merged_policy

    # Merge ICM weights
    if "icm" in checkpoints[0]:
        ref_icm = checkpoints[0]["icm"]
        merged_icm = {}
        for key in ref_icm:
            merged_icm[key] = sum(
                w * cp["icm"][key].float()
                for w, cp in zip(weights, checkpoints)
            )
        merged["icm"] = merged_icm

    # Keep config from first checkpoint
    if "config" in checkpoints[0]:
        merged["config"] = checkpoints[0]["config"]

    return merged


def slerp_merge(cp_a, cp_b, alpha=0.5):
    """
    Spherical linear interpolation between two checkpoints.

    SLERP preserves the magnitude of weight vectors (unlike linear
    interpolation which can shrink them). Better for merging models
    trained on different data distributions.

    alpha=0.0 → 100% A, alpha=1.0 → 100% B, alpha=0.5 → midpoint
    """
    print(f"  SLERP merge: alpha={alpha} (0=A, 1=B)")

    def _slerp_tensor(a, b, t):
        """SLERP between two tensors."""
        a_flat = a.float().flatten()
        b_flat = b.float().flatten()

        # Normalize
        a_norm = torch.nn.functional.normalize(a_flat, dim=0)
        b_norm = torch.nn.functional.normalize(b_flat, dim=0)

        # Angle between vectors
        dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
        omega = torch.acos(dot)

        if omega.abs() < 1e-6:
            # Vectors are nearly identical, just lerp
            return ((1 - t) * a + t * b)

        sin_omega = torch.sin(omega)
        result_flat = (torch.sin((1 - t) * omega) / sin_omega * a_flat +
                       torch.sin(t * omega) / sin_omega * b_flat)
        return result_flat.reshape(a.shape)

    merged = {}

    # SLERP policy
    merged_policy = {}
    for key in cp_a["policy"]:
        merged_policy[key] = _slerp_tensor(
            cp_a["policy"][key], cp_b["policy"][key], alpha
        )
    merged["policy"] = merged_policy

    # SLERP ICM
    if "icm" in cp_a and "icm" in cp_b:
        merged_icm = {}
        for key in cp_a["icm"]:
            merged_icm[key] = _slerp_tensor(
                cp_a["icm"][key], cp_b["icm"][key], alpha
            )
        merged["icm"] = merged_icm

    if "config" in cp_a:
        merged["config"] = cp_a["config"]

    return merged


def evaluate_checkpoint(path, episodes=50, max_steps=400):
    """Quick evaluation: play curriculum levels and measure win rate."""
    from src.games.mario.mario_torch_agent import MarioTorchAgent
    from src.games.mario.mario_adapter import MarioAdapter
    from src.games.mario.mario_curriculum import MarioCurriculum

    agent = MarioTorchAgent(obs_dim=378, n_actions=6)
    agent.load(path)

    adapter = MarioAdapter()
    curriculum = MarioCurriculum(start_tier=1)

    wins = 0
    total_reward = 0.0

    for ep in range(episodes):
        level = curriculum.next_level()
        obs = adapter.reset(level)
        agent.reset()
        ep_reward = 0.0

        for step in range(max_steps):
            action = agent.step(obs)
            next_obs, reward, done, info = adapter.step(action)
            ep_reward += reward
            obs = next_obs
            if done:
                break

        if level.won:
            wins += 1
        total_reward += ep_reward

        progress = level.max_x_reached / max(1, level.width)
        curriculum.record_result(won=level.won, progress=progress,
                                  steps=step + 1, level=level)

    win_rate = wins / episodes
    avg_reward = total_reward / episodes
    max_tier = curriculum.tier

    print(f"    {path}: wr={win_rate:.0%}, r={avg_reward:+.1f}, tier={max_tier}")
    return {"path": path, "win_rate": win_rate, "avg_reward": avg_reward,
            "max_tier": max_tier, "score": win_rate * 100 + max_tier * 10}


def main():
    parser = argparse.ArgumentParser(description="Merge Mario agent checkpoints")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input .pt checkpoint files")
    parser.add_argument("--output", type=str, default="merged_agent.pt",
                        help="Output merged checkpoint")
    parser.add_argument("--strategy", choices=["average", "slerp", "best"],
                        default="average", help="Merge strategy")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Per-checkpoint weights for average merge")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SLERP interpolation alpha (0=first, 1=second)")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Episodes for best-of evaluation")
    args = parser.parse_args()

    print("=" * 60)
    print("  MARIO WEIGHT MERGER")
    print(f"  Strategy: {args.strategy}")
    print(f"  Inputs: {len(args.inputs)} checkpoints")
    print("=" * 60)

    if args.strategy == "best":
        # Evaluate each and pick the best
        print("  Evaluating each checkpoint...")
        results = []
        for path in args.inputs:
            result = evaluate_checkpoint(path, episodes=args.eval_episodes)
            results.append(result)

        best = max(results, key=lambda r: r["score"])
        print(f"\n  Best: {best['path']} (score={best['score']:.1f})")

        # Copy best to output
        import shutil
        shutil.copy2(best["path"], args.output)
        print(f"  Saved to {args.output}")

    elif args.strategy == "slerp":
        if len(args.inputs) != 2:
            print("  ERROR: SLERP requires exactly 2 inputs")
            return
        cp_a = load_checkpoint(args.inputs[0])
        cp_b = load_checkpoint(args.inputs[1])
        merged = slerp_merge(cp_a, cp_b, alpha=args.alpha)
        torch.save(merged, args.output)
        print(f"  Saved to {args.output}")

    else:  # average
        checkpoints = [load_checkpoint(p) for p in args.inputs]
        merged = average_merge(checkpoints, weights=args.weights)
        torch.save(merged, args.output)
        print(f"  Saved to {args.output}")

    print("=" * 60)

    # Quick eval of merged result
    print("  Evaluating merged checkpoint...")
    evaluate_checkpoint(args.output, episodes=50)
    print("=" * 60)


if __name__ == "__main__":
    main()
