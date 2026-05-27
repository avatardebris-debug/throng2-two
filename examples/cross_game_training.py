"""
cross_game_training.py — Joint world model training across Mario, CartPole, MountainCar.

Trains a MultiGameWorldModel that learns physics representations shared across games.
The key insight: gravity, momentum, and contact dynamics appear in all three games;
a shared encoder should capture these, while per-game heads capture specifics.

Training protocol:
  Phase 1 (warm-up):  100 random episodes per game → fill multi-game replay buffer
  Phase 2 (joint):    Interleaved training: collect 1 episode from each game,
                      then do 5 world model update steps on balanced batches
  Phase 3 (transfer): Freeze world model; train a fresh agent on LunarLander;
                      compare episodes-to-solve vs. a baseline with no world model

Metrics tracked:
  - per-game prediction MSE (surprise) over time
  - dream accuracy: how often dream action == best real action in hindsight
  - transfer efficiency: episodes to reach solve threshold on LunarLander

Usage:
    python examples/cross_game_training.py --episodes 300
    python examples/cross_game_training.py --eval-only --games mario cartpole
    python examples/cross_game_training.py --transfer-test --target lunarlander
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.cross_game.training_loop import (
    _GYM_AVAILABLE,
    _TORCH_AVAILABLE,
    run_training,
    run_transfer_test,
)


def main():
    parser = argparse.ArgumentParser(description="Cross-game world model training")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per game")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["mario", "cartpole"],
        help="Games to train on (space-separated)",
    )
    parser.add_argument("--z-dim", type=int, default=32, help="z-vector dimension")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument(
        "--wm-steps", type=int, default=5, help="World model updates per episode"
    )
    parser.add_argument("--log-every", type=int, default=10, help="Log every N episodes")
    parser.add_argument("--save", type=str, default=None, help="Save results JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training; just evaluate"
    )
    parser.add_argument(
        "--transfer-test", action="store_true", help="Run transfer test after training"
    )
    parser.add_argument(
        "--target", type=str, default="lunarlander", help="Transfer test target game"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--elite-n",
        type=int,
        default=3,
        help="Keep top-N replay trajectories per game (default 3)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N episodes (0=off)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint directory (default results/checkpoints)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint directory to resume from"
    )
    parser.add_argument(
        "--dual-mode",
        action="store_true",
        help="Enable surprise-triggered DualModeEncoder during episodes",
    )
    args = parser.parse_args()

    save_path = args.save or f"results/cross_game_z{args.z_dim}_ep{args.episodes}.json"

    results = run_training(
        games=args.games,
        total_episodes=args.episodes,
        wm_train_steps_per_episode=args.wm_steps,
        max_steps_per_episode=args.max_steps,
        z_dim=args.z_dim,
        log_every=args.log_every,
        save_path=save_path,
        seed=args.seed,
        verbose=args.verbose,
        elite_n=args.elite_n,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=args.checkpoint_path,
        resume_from=args.resume,
        use_dual_mode=args.dual_mode,
    )

    if args.transfer_test and _TORCH_AVAILABLE and _GYM_AVAILABLE:
        print("\n═══ Transfer Test ═══")
        world_model = results.get("world_model")
        encoder = results.get("encoder")
        if world_model is None or encoder is None:
            print("[WARN] No trained world model from run; skipping transfer test")
        else:
            transfer = run_transfer_test(
                world_model,
                encoder,
                target_game=args.target,
                verbose=args.verbose,
            )
            results["transfer"] = transfer


if __name__ == "__main__":
    main()
