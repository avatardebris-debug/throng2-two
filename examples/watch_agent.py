"""
watch_agent.py -- Watch the agent play Mario in your terminal.

Modes:
  ASCII simulator: renders the ASCII level the agent plays on
  Real NES game:   runs gym-super-mario-bros, shows both the game window
                   AND the ASCII grid the agent "sees" in your terminal

Usage:
  python examples/watch_agent.py                              # random on ASCII sim
  python examples/watch_agent.py --load mario_agent.pt        # trained on ASCII sim
  python examples/watch_agent.py --load mario_agent.pt --real # trained on REAL NES
  python examples/watch_agent.py --tier 5 --speed 0.05        # specific tier, fast
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


ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "JUMP", "J+LEFT", "J+RIGHT", "RUN_R", "R+B+A"]


def render_sim_frame(sim, action, step, ep_reward, ep, won_count, total_eps):
    """Render one frame of ASCII simulator to terminal."""
    print("\033[2J\033[H", end="")

    print(f"  Episode {ep+1}/{total_eps} | Step {step:3d} | "
          f"Action: {ACTION_NAMES[action]:6s} | "
          f"Reward: {ep_reward:+.1f} | Wins: {won_count}")
    print("  " + "-" * 42)

    ascii_str = sim.render_ascii(viewport=True)
    for line in ascii_str.split("\n"):
        print(f"  {line}")

    print("  " + "-" * 42)
    print(f"  Mario: ({sim.mario_row}, {sim.mario_col}) | "
          f"Progress: {sim.mario_col}/{sim.width} | "
          f"{'ON GROUND' if sim.on_ground else 'IN AIR'} | "
          f"Coins: {sim.coins}")


def render_real_frame(adapter, action, step, ep_reward, ep, won_count,
                      total_eps, info):
    """Render the ASCII interpretation of a real NES frame."""
    print("\033[2J\033[H", end="")

    x_pos = info.get("x_pos", 0)
    life = info.get("life", 0)
    coins = info.get("coins", 0)
    world = info.get("world", 1)
    stage = info.get("stage", 1)

    print(f"  REAL NES | Ep {ep+1}/{total_eps} | Step {step:3d} | "
          f"Action: {ACTION_NAMES[action]:6s} | "
          f"Reward: {ep_reward:+.1f} | Wins: {won_count}")
    print(f"  World {world}-{stage} | x_pos: {x_pos} | "
          f"Lives: {life} | Coins: {coins}")
    print("  " + "-" * 50)

    # Show the ASCII grid the agent sees
    ascii_view = adapter.render()
    for line in ascii_view.split("\n"):
        print(f"  {line}")

    print("  " + "-" * 50)
    stats = adapter.stats()
    mario = stats.get("mario_pos", (0, 0))
    enemies = stats.get("enemies", [])
    print(f"  Agent sees: Mario at {mario} | "
          f"Enemies: {len(enemies)} | "
          f"x_pos: {x_pos}")


def create_agent(args):
    """Create and optionally load agent."""
    use_torch = HAS_TORCH and not args.force_numpy
    if use_torch:
        agent = MarioTorchAgent(obs_dim=378, n_actions=8, device="cpu")
    else:
        agent = MarioICMAgent(obs_dim=378, n_actions=8)

    if args.load:
        if args.load.endswith(".pt") and use_torch:
            agent.load(args.load)
        elif hasattr(agent, "load"):
            agent.load(args.load)
        print(f"  Loaded: {args.load}")
    else:
        print("  No weights loaded -- using random agent")

    return agent


def watch_ascii(agent, args):
    """Watch agent play ASCII simulator levels."""
    adapter = MarioAdapter()
    curriculum = MarioCurriculum(
        start_tier=args.tier or 1,
        advance_threshold=0.7
    )
    won_count = 0

    print("\n  MODE: ASCII Simulator")
    print("  Press Ctrl+C to stop\n")
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

                render_sim_frame(level, action, step, ep_reward, ep,
                                 won_count, args.episodes)
                time.sleep(args.speed)

                next_obs, reward, done, info = adapter.step(action)
                ep_reward += reward
                obs = next_obs

                if done:
                    break

            result = "WIN!" if level.won else "DIED"
            if level.won:
                won_count += 1

            render_sim_frame(level, action, step, ep_reward, ep,
                             won_count, args.episodes)
            print(f"\n  >>> {result} | Final reward: {ep_reward:+.1f} | "
                  f"Distance: {level.max_x_reached}/{level.width}")
            print(f"  >>> Tier: {curriculum.tier}")

            progress = level.max_x_reached / max(1, level.width)
            curriculum.record_result(won=level.won, progress=progress,
                                      steps=step + 1, level=level)

            time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    print(f"\n  Summary: {won_count}/{args.episodes} wins "
          f"({won_count/max(args.episodes,1):.0%})")


def watch_real(agent, args):
    """Watch agent play real NES Mario with ASCII overlay."""
    from src.games.mario.mario_real_adapter import MarioRealAdapter

    render_mode = "human" if not args.headless else None
    adapter = MarioRealAdapter(render_mode=render_mode)
    won_count = 0

    print("\n  MODE: Real NES Game (gym-super-mario-bros)")
    print("  The ASCII viewport below is what the agent 'sees'")
    print("  Press Ctrl+C to stop\n")
    time.sleep(1)

    try:
        for ep in range(args.episodes):
            obs = adapter.reset()
            agent.reset()
            ep_reward = 0.0
            info = {}

            for step in range(args.max_steps):
                action = agent.step(obs)

                render_real_frame(adapter, action, step, ep_reward, ep,
                                  won_count, args.episodes, info)
                time.sleep(args.speed)

                obs, reward, done, info = adapter.step(action)
                ep_reward += reward

                if done:
                    break

            won = info.get("flag_get", False)
            if won:
                won_count += 1
            result = "WIN!" if won else "DIED"

            render_real_frame(adapter, action, step, ep_reward, ep,
                             won_count, args.episodes, info)
            print(f"\n  >>> {result} | Final reward: {ep_reward:+.1f} | "
                  f"x_pos: {info.get('x_pos', 0)}")

            time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    adapter.close()

    print(f"\n  Summary: {won_count}/{args.episodes} wins "
          f"({won_count/max(args.episodes,1):.0%})")


def main():
    parser = argparse.ArgumentParser(description="Watch agent play Mario")
    parser.add_argument("--load", type=str, default=None,
                        help="Load trained weights (.pt or .npz)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to watch")
    parser.add_argument("--tier", type=int, default=None,
                        help="Force a specific tier (ASCII mode only)")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Max steps per episode")
    parser.add_argument("--speed", type=float, default=0.1,
                        help="Seconds between frames (lower = faster)")
    parser.add_argument("--force-numpy", action="store_true")
    parser.add_argument("--real", action="store_true",
                        help="Play REAL NES game (needs gym-super-mario-bros)")
    parser.add_argument("--headless", action="store_true",
                        help="Real game without rendering window (cloud)")
    args = parser.parse_args()

    agent = create_agent(args)

    if args.real:
        args.max_steps = args.max_steps if args.max_steps != 400 else 2000
        watch_real(agent, args)
    else:
        watch_ascii(agent, args)


if __name__ == "__main__":
    main()
