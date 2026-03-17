"""
run_mario_zone_training.py — Training script with zone-based Go-Explore curriculum.

Trains MarioICMAgent using the ZoneCurriculum system:
  - Geographic zone progression (6 zones, cumulative)
  - Column-novelty DiscoRL bonus
  - Checkpoint management (spawn from zone boundaries)
  - Death hotspot tracking + targeted drills (--drill flag)
  - Periodic stats logging with zone progress bars

Usage:
    python examples/run_mario_zone_training.py --episodes 1000 --tier 5
    python examples/run_mario_zone_training.py --episodes 500 --tier 3 --log-zones
    python examples/run_mario_zone_training.py --episodes 1000 --tier 5 --drill
"""

from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.games.mario.mario_generator import MarioLevelGenerator
from src.games.mario.mario_adapter import MarioAdapter
from src.games.mario.mario_zone_curriculum import ZoneCurriculum
from src.games.mario.mario_icm_agent import MarioICMAgent
from src.games.mario.mario_difficulty_analyzer import DifficultyAnalyzer, HotspotDrillCurriculum


def main():
    parser = argparse.ArgumentParser(description="Mario Zone Curriculum Training")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--tier", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--log-zones", action="store_true")
    parser.add_argument("--novelty-bonus", type=float, default=0.1)
    parser.add_argument("--drill", action="store_true",
                        help="Run hotspot drills between zone phases")
    parser.add_argument("--drill-interval", type=int, default=200,
                        help="Run drills every N zone episodes")
    parser.add_argument("--drill-max-episodes", type=int, default=100,
                        help="Max drill episodes per hotspot session")
    parser.add_argument("--drill-threshold", type=float, default=0.70,
                        help="Success rate needed to master a drill hotspot")
    args = parser.parse_args()

    print(f"═══ Mario Zone Curriculum Training ═══")
    print(f"  Episodes: {args.episodes}")
    print(f"  Tier: {args.tier}")
    print(f"  Max steps/episode: {args.max_steps}")
    print(f"  Novelty bonus: {args.novelty_bonus}")
    print()

    # ── Setup ──────────────────────────────────────────────────────
    gen = MarioLevelGenerator(seed=args.seed)
    adapter = MarioAdapter()

    curriculum = ZoneCurriculum(
        generator=gen,
        tier=args.tier,
        novelty_bonus=args.novelty_bonus,
        seed=args.seed,
    )

    agent = MarioICMAgent(
        obs_dim=378,
        n_actions=8,
        intrinsic_lambda=0.3,
    )

    # Difficulty analyzer (collects deaths for drill curriculum)
    analyzer = DifficultyAnalyzer()
    drill_curriculum: HotspotDrillCurriculum = None  # built after enough deaths
    drill_episode_count = 0
    last_drill_ep = 0

    # ── Training loop ─────────────────────────────────────────────
    episode_rewards = []
    episode_columns = []
    wins = 0
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        sim, zone_info = curriculum.get_episode()
        obs = adapter.reset(sim)
        agent.reset()

        total_reward = 0.0
        done = False
        last_action = 0

        for step in range(args.max_steps):
            action = agent.step(obs)
            last_action = action
            obs_next, reward, done, info = adapter.step(action)

            # Add column novelty (DiscoRL)
            novelty = curriculum.column_visited(sim.mario_col)
            total_reward_step = reward + novelty

            # Learn with ICM curiosity + DiscoRL novelty
            agent.learn_with_next_obs(reward + novelty, done, obs_next)
            obs = obs_next
            total_reward += total_reward_step

            if done:
                break

        # Report result
        final_col = sim.mario_col
        curriculum.report_result(final_col, sim.won, sim.alive)

        if not sim.alive:
            curriculum.record_death(final_col, sim.mario_row, last_action)
            analyzer.record_death(final_col, sim.mario_row, last_action)

        # ── Hotspot drill phase (if --drill flag set) ──────────────
        if args.drill and ep - last_drill_ep >= args.drill_interval:
            hotspots = analyzer.analyze()
            if hotspots:
                last_drill_ep = ep
                # (Re)build drill curriculum on current level
                dc = HotspotDrillCurriculum(
                    source_sim=curriculum._current_level,
                    analyzer=analyzer,
                    pass_threshold=args.drill_threshold,
                    min_attempts=10,
                )
                if not dc.all_mastered():
                    print(f"  [DRILL] Starting hotspot drills at ep {ep}")
                    print(f"  {analyzer.report()}")
                    for d_ep in range(args.drill_max_episodes):
                        if dc.all_mastered():
                            break
                        psim, dinfo = dc.get_episode()
                        if psim is None:
                            break
                        obs = adapter.reset(psim)
                        agent.reset()
                        done = False
                        for _ in range(args.max_steps):
                            action = agent.step(obs)
                            obs_next, reward, done_step, _ = adapter.step(action)
                            agent.learn_with_next_obs(reward, done_step, obs_next)
                            obs = obs_next
                            drill_episode_count += 1
                            if done_step:
                                done = True
                                break
                        success = psim.won or psim.mario_col >= dinfo.get("success_col", 999)
                        dc.report_result(success=success, final_col=psim.mario_col)
                    print(f"  [DRILL] Done. {dc.report()}")

        if sim.won:
            wins += 1

        episode_rewards.append(total_reward)
        episode_columns.append(final_col)

        # ── Logging ────────────────────────────────────────────────
        if ep % args.log_interval == 0:
            avg_r = np.mean(episode_rewards[-args.log_interval:])
            avg_col = np.mean(episode_columns[-args.log_interval:])
            elapsed = time.time() - t0
            eps_per_sec = ep / elapsed
            stats = curriculum.stats()

            print(f"Episode {ep:>5d} | "
                  f"Zone {stats['current_zone']} ({stats['zone_name']}) | "
                  f"Rate {stats['success_rate']:.0%} | "
                  f"AvgCol {avg_col:.0f}/{stats['target_col']} | "
                  f"AvgR {avg_r:+.1f} | "
                  f"Wins {wins} | "
                  f"Best {stats['best_col_reached']} | "
                  f"{eps_per_sec:.1f} ep/s")

            if args.log_zones:
                print(curriculum.zone_summary())
                print()

    # ── Final report ──────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print(f"═══ Training Complete ═══")
    print(f"  Total episodes: {args.episodes}")
    print(f"  Total wins: {wins}")
    print(f"  Win rate: {wins/args.episodes:.1%}")
    print(f"  Time: {elapsed:.1f}s ({args.episodes/elapsed:.1f} ep/s)")
    print()
    print(curriculum.zone_summary())

    # Death hotspots
    hotspots = curriculum.get_death_hotspots()
    if hotspots:
        print()
        print(f"  Death Hotspots (top 5):")
        for h in hotspots[:5]:
            rng = h['col_range']
            print(f"    Col {rng[0]}-{rng[1]}: {h['count']} deaths")


if __name__ == "__main__":
    main()
