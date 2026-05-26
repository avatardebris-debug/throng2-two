"""
seed_mario_demos.py -- Load FCEUX .fm2 playthroughs into the elite replay buffer.

Usage:
    # Seed from one or more fm2 files
    python examples/seed_mario_demos.py

    # List what is in the current checkpoint's elite buffer
    python examples/seed_mario_demos.py --list

    # Specify a different checkpoint directory
    python examples/seed_mario_demos.py --checkpoint results/checkpoints/ep000300

Each .fm2 file becomes one "human" trajectory in the elite buffer.
The score is estimated as (frame_count * completion_bonus) since the real
NES reward can't be replayed without the emulator -- you can override with
--score to supply a manual score (e.g. 1000 for a full level-1 clear).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.elite_replay import EliteReplayManager, parse_fm2

# Default fm2 locations (auto-discovered)
_DEFAULT_FM2_DIRS = [
    "src/games/mario",
    "recordings",
]

# Default score when none provided:
# A level-1 clear in SMB is worth ~1000 points + time bonus.
_DEFAULT_SCORE = 1200.0


def find_fm2_files(dirs: list) -> list:
    found = []
    for d in dirs:
        found.extend(glob.glob(os.path.join(d, "*.fm2")))
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description="Seed elite replay buffer from FCEUX fm2 files")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/latest",
                        help="Checkpoint directory (default: results/checkpoints/latest)")
    parser.add_argument("--score", type=float, default=None,
                        help="Score to assign all demos (default: auto from frame count)")
    parser.add_argument("--fm2", nargs="*", default=None,
                        help="Specific .fm2 files; auto-discovers from default dirs if omitted")
    parser.add_argument("--list", action="store_true", help="List current elite buffer and exit")
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_dir = args.checkpoint
    if not os.path.isdir(ckpt_dir):
        # Try .txt fallback (Windows symlink substitute)
        txt = ckpt_dir + ".txt"
        if os.path.exists(txt):
            with open(txt) as f:
                ckpt_dir = f.read().strip()

    elite_dir = os.path.join(ckpt_dir, "elite_replay")
    game = "mario"

    # Load existing buffer if present
    try:
        mgr = EliteReplayManager.load(elite_dir)
        print(f"Loaded existing elite buffer: {mgr.stats()}")
    except Exception:
        print("No existing elite buffer found -- creating fresh.")
        mgr = EliteReplayManager(games=[game], n=3)

    if args.list:
        print("\nElite buffer contents:")
        buf = mgr.buffer(game)
        for i, traj in enumerate(buf._elites):
            print(f"  [{i}] label={traj.label:6s}  score={traj.score:8.1f}  "
                  f"frames={len(traj.actions):5d}  ep={traj.episode}")
        return

    # Collect fm2 files
    fm2_files = args.fm2 if args.fm2 else find_fm2_files(_DEFAULT_FM2_DIRS)
    if not fm2_files:
        print("No .fm2 files found. Searched:", _DEFAULT_FM2_DIRS)
        print("Use --fm2 path/to/file.fm2 to specify files explicitly.")
        return

    print(f"\nFound {len(fm2_files)} .fm2 file(s):")
    for path in fm2_files:
        print(f"  {path}")

    print()
    added = 0
    for path in fm2_files:
        try:
            actions, frames = parse_fm2(path)
            # Estimate score: ~1 point per 10 frames + flat bonus for longer runs
            score = args.score if args.score is not None else float(frames) / 10.0 + 500.0
            accepted = mgr.seed_human(game, actions, score)
            status = "ACCEPTED" if accepted else "rejected (below elite threshold)"
            print(f"  {os.path.basename(path):50s}: {frames:5d} frames  "
                  f"score={score:7.1f}  {status}")
            if accepted:
                added += 1
        except Exception as e:
            print(f"  {os.path.basename(path)}: ERROR -- {e}")

    print(f"\n{added}/{len(fm2_files)} demo(s) added to elite buffer.")
    print(f"Elite buffer: {mgr.stats()}")

    # Save back to checkpoint
    os.makedirs(elite_dir, exist_ok=True)
    mgr.save(elite_dir)
    print(f"Saved to: {elite_dir}")


if __name__ == "__main__":
    main()
