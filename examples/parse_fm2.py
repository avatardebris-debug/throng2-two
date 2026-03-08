"""
parse_fm2.py -- Parse FCEUX movie files (.fm2) into action sequences.

Extracts frame-by-frame NES controller inputs from FCEUX movie recordings.
Can be used for:
  1. Imitation learning -- train agent on expert playthrough
  2. Level mapping -- replay in emulator to capture ASCII grids
  3. Action analysis -- study expert strategies

.fm2 format: |frame|RLDUTSBA|RLDUTSBA||
  Controller 1 buttons: R=Right, L=Left, D=Down, U=Up,
                        T=Start, S=Select, B=B-button, A=A-button

Our action mapping (matches simulator Action enum):
  0: NOOP       (no buttons)
  1: LEFT       (L only)
  2: RIGHT      (R only)
  3: JUMP       (A only)
  4: JUMP_LEFT  (L + A)
  5: JUMP_RIGHT (R + A)
  6: RUN_RIGHT  (R + B)
  7: RUN_JUMP   (R + B + A)

Usage:
  python examples/parse_fm2.py path/to/movie.fm2
  python examples/parse_fm2.py path/to/movie.fm2 --replay  # replay in NES emulator
"""

import sys
import os
import argparse
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_fm2(path: str) -> List[dict]:
    """
    Parse .fm2 movie file into list of frame data.

    Returns:
        list of {"frame": int, "buttons": str, "action": int, "raw": str}
    """
    frames = []
    frame_num = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue  # Skip header lines

            parts = line.split("|")
            if len(parts) < 3:
                continue

            # Controller 1 buttons: RLDUTSBA
            buttons = parts[2] if len(parts) > 2 else "........"

            # Map to our 8 actions (matching simulator Action enum)
            r = "R" in buttons and buttons[0] == "R"
            l = "L" in buttons and buttons[1] == "L"
            a = "A" in buttons and buttons[7] == "A"
            b = "B" in buttons and buttons[6] == "B"

            if r and a and b:
                action = 7   # RUN_JUMP (right + B + A)
            elif r and b:
                action = 6   # RUN_RIGHT (right + B)
            elif r and a:
                action = 5   # JUMP_RIGHT (right + A)
            elif l and a:
                action = 4   # JUMP_LEFT
            elif r:
                action = 2   # RIGHT
            elif l:
                action = 1   # LEFT
            elif a:
                action = 3   # JUMP
            else:
                action = 0   # NOOP

            frames.append({
                "frame": frame_num,
                "buttons": buttons,
                "action": action,
                "raw": line,
            })
            frame_num += 1

    return frames


ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "JUMP", "J+LEFT", "J+RIGHT", "RUN_R", "R+B+A"]


def analyze_playthrough(frames: List[dict]):
    """Print analysis of the expert playthrough."""
    print(f"  Total frames: {len(frames)}")
    print(f"  Duration: {len(frames) / 60:.1f}s (at 60fps)")
    print()

    # Action distribution
    action_counts = {}
    for f in frames:
        a = f["action"]
        action_counts[a] = action_counts.get(a, 0) + 1

    print("  Action Distribution:")
    for action, count in sorted(action_counts.items()):
        pct = count / len(frames) * 100
        bar = "#" * int(pct / 2)
        print(f"    {ACTION_NAMES[action]:8s}: {count:5d} ({pct:5.1f}%) {bar}")
    print()

    # Find key moments (action changes)
    transitions = []
    for i in range(1, len(frames)):
        if frames[i]["action"] != frames[i-1]["action"]:
            transitions.append((i, frames[i-1]["action"], frames[i]["action"]))

    print(f"  Action transitions: {len(transitions)}")
    print(f"  Avg hold duration: {len(frames) / max(len(transitions), 1):.1f} frames")

    # First 20 transitions
    print("  First 20 transitions:")
    for i, (frame, old, new) in enumerate(transitions[:20]):
        print(f"    Frame {frame:5d}: {ACTION_NAMES[old]:8s} -> {ACTION_NAMES[new]}")


def extract_action_sequence(frames: List[dict], frame_skip: int = 4) -> List[int]:
    """
    Extract action sequence at our agent's frame skip rate.

    The agent acts every frame_skip frames (default 4 = 15fps).
    We take the most common action in each window.
    """
    actions = []
    for i in range(0, len(frames), frame_skip):
        window = frames[i:i + frame_skip]
        # Most common action in this window
        action_counts = {}
        for f in window:
            a = f["action"]
            action_counts[a] = action_counts.get(a, 0) + 1
        best = max(action_counts, key=action_counts.get)
        actions.append(best)
    return actions


def main():
    parser = argparse.ArgumentParser(description="Parse FCEUX .fm2 movie files")
    parser.add_argument("fm2_path", help="Path to .fm2 file")
    parser.add_argument("--replay", action="store_true",
                        help="Replay in NES emulator (needs gym-super-mario-bros)")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Frame skip for action extraction")
    parser.add_argument("--save-actions", type=str, default=None,
                        help="Save action sequence as .npy")
    args = parser.parse_args()

    print("=" * 60)
    print("  FM2 MOVIE PARSER")
    print(f"  File: {args.fm2_path}")
    print("=" * 60)

    frames = parse_fm2(args.fm2_path)
    analyze_playthrough(frames)

    # Extract at agent rate
    actions = extract_action_sequence(frames, args.frame_skip)
    print(f"\n  Agent actions (at {60//args.frame_skip}fps): {len(actions)} steps")
    print(f"  First 30: {[ACTION_NAMES[a] for a in actions[:30]]}")

    if args.save_actions:
        import numpy as np
        np.save(args.save_actions, np.array(actions, dtype=np.int32))
        print(f"  Saved to {args.save_actions}")

    if args.replay:
        replay_with_ascii(actions)

    print("=" * 60)


def replay_with_ascii(actions: List[int]):
    """Replay the expert actions in NES emulator, showing ASCII conversion."""
    import time
    from src.games.mario.mario_real_adapter import MarioRealAdapter

    print("\n  Replaying in NES emulator with ASCII overlay...")
    adapter = MarioRealAdapter(render_mode="human")
    obs = adapter.reset()

    for step, action in enumerate(actions):
        obs, reward, done, info = adapter.step(action)

        if step % 10 == 0:
            ascii_view = adapter.render()
            print(f"\n  Step {step} | x_pos={info.get('x_pos', 0)} "
                  f"| Action: {ACTION_NAMES[action]}")
            for line in ascii_view.split("\n")[:5]:  # First 5 rows
                print(f"  {line}")

        if done:
            print(f"\n  Episode ended at step {step}: "
                  f"{'WIN' if info.get('flag_get') else 'DIED'}")
            break

        time.sleep(0.016)  # ~60fps

    adapter.close()


if __name__ == "__main__":
    main()
