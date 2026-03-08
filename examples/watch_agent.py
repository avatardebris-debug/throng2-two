"""
watch_agent.py -- Watch the agent play ASCII Mario in your terminal.

Renders each step as ASCII art so you can see exactly what it's doing.

Usage:
  python examples/watch_agent.py                        # random agent
  python examples/watch_agent.py --load mario_agent.pt  # trained agent
  python examples/watch_agent.py --tier 3 --episodes 5  # specific tier
  python examples/watch_agent.py --speed 0.05           # faster playback
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.games.mario.mario_adapter import MarioAdapter
from src.games.mario.mario_curriculum import MarioCurriculum
from src.games.mario.mario_simulator import TILE_CHAR

# Try torch agent first
try:
    import torch
    from src.games.mario.mario_torch_agent import MarioTorchAgent
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.games.mario.mario_icm_agent import MarioICMAgent


ACTION_NAMES = ["NOOP", "RIGHT", "R+JUMP", "LEFT", "JUMP", "RUN_R"]


def render_frame(sim, action, step, ep_reward, ep, won_count, total_eps):
    """Render one frame of the game to terminal."""
    # Clear screen
    print("\033[2J\033[H", end="")

    # Header
    print(f"  Episode {ep+1}/{total_eps} | Step {step:3d} | "
          f"Action: {ACTION_NAMES[action]:6s} | "
          f"Reward: {ep_reward:+.1f} | Wins: {won_count}")
    print("  " + "-" * 42)

    # Render viewport
    ascii_str = sim.render_ascii(viewport=True)
    for line in ascii_str.split("\n"):
        print(f"  {line}")

    print("  " + "-" * 42)
    print(f"  Mario: ({sim.mario_row}, {sim.mario_col}) | "
          f"Progress: {sim.mario_col}/{sim.width} | "
          f"{'ON GROUND' if sim.on_ground else 'IN AIR'} | "
          f"Coins: {sim.coins}")


def main():
    parser = argparse.ArgumentParser(description="Watch agent play ASCII Mario")
    parser.add_argument("--load", type=str, default=None,
                        help="Load trained weights (.pt or .npz)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to watch")
    parser.add_argument("--tier", type=int, default=None,
                        help="Force a specific tier (default: curriculum)")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Max steps per episode")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="Seconds between frames (lower = faster)")
    parser.add_argument("--force-numpy", action="store_true")
    args = parser.parse_args()

    # Create agent
    use_torch = HAS_TORCH and not args.force_numpy
    if use_torch:
        agent = MarioTorchAgent(obs_dim=378, n_actions=6, device="cpu")
    else:
        agent = MarioICMAgent(obs_dim=378, n_actions=6)

    if args.load:
        if args.load.endswith(".pt") and use_torch:
            agent.load(args.load)
        elif hasattr(agent, "load"):
            agent.load(args.load)
        print(f"  Loaded: {args.load}")
    else:
        print("  No weights loaded -- using random agent")

    adapter = MarioAdapter()
    curriculum = MarioCurriculum(
        start_tier=args.tier or 1,
        advance_threshold=0.7
    )

    won_count = 0

    print("\n  Press Ctrl+C to stop\n")
    time.sleep(1)

    try:
        for ep in range(args.episodes):
            if args.tier:
                curriculum.tier = args.tier
            level = curriculum.next_level()
            obs = adapter.reset(level)
            agent.reset()
            ep_reward = 0.0

            for step in range(args.max_steps):
                action = agent.step(obs)

                # Render before stepping
                render_frame(level, action, step, ep_reward, ep, 
                             won_count, args.episodes)
                time.sleep(args.speed)

                next_obs, reward, done, info = adapter.step(action)
                ep_reward += reward
                obs = next_obs

                if done:
                    break

            # Final frame
            result = "WIN!" if level.won else "DIED"
            if level.won:
                won_count += 1

            render_frame(level, action, step, ep_reward, ep,
                         won_count, args.episodes)
            print(f"\n  >>> {result} | Final reward: {ep_reward:+.1f} | "
                  f"Distance: {level.max_x_reached}/{level.width}")
            print(f"  >>> Tier: {curriculum.tier}")

            progress = level.max_x_reached / max(1, level.width)
            curriculum.record_result(won=level.won, progress=progress,
                                      steps=step + 1, level=level)

            time.sleep(1.5)  # Pause between episodes

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    print(f"\n  Summary: {won_count}/{args.episodes} wins "
          f"({won_count/max(args.episodes,1):.0%})")


if __name__ == "__main__":
    main()
