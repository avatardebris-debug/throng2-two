"""
run_mario_hpo.py — CLI for Bayesian HPO of MarioICMAgent.

Usage:
    python examples/run_mario_hpo.py
    python examples/run_mario_hpo.py --trials 30 --eval-episodes 20 --tier 3
    python examples/run_mario_hpo.py --trials 50 --tier 5 --save results/hpo_t5.json
    python examples/run_mario_hpo.py --load results/hpo_t3.json  # show best config
"""
from __future__ import annotations
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.mario.mario_hpo import run_hpo, load_best_config, MarioParameterSpace

def main():
    parser = argparse.ArgumentParser(description="Bayesian HPO for MarioICMAgent")
    parser.add_argument("--trials", type=int, default=20,
                        help="Total HP evaluation budget")
    parser.add_argument("--initial-random", type=int, default=5,
                        help="Random trials before Bayesian search")
    parser.add_argument("--eval-episodes", type=int, default=15,
                        help="Episodes per config evaluation")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode during eval")
    parser.add_argument("--tier", type=int, default=3,
                        help="Mario level tier (1-7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save HPO results JSON")
    parser.add_argument("--load", type=str, default=None,
                        help="Load + display results from a previous run")
    args = parser.parse_args()

    if args.load:
        config = load_best_config(args.load)
        if config is None:
            print(f"Could not load config from {args.load}")
            sys.exit(1)
        ps = MarioParameterSpace()
        print(f"Best config from {args.load}:")
        print(ps.pretty(config))
        return

    save_path = args.save or f"results/hpo_tier{args.tier}_t{args.trials}.json"

    best_config, best_score, history = run_hpo(
        tier=args.tier,
        n_trials=args.trials,
        n_initial_random=args.initial_random,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        save_path=save_path,
        verbose=True,
    )

    print(f"\n  To use with zone training:")
    print(f"  python examples/run_mario_zone_training.py --tier {args.tier} \\")
    print(f"      # (agent_from_config auto-loads from '{save_path}')")

if __name__ == "__main__":
    main()
