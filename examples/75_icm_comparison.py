"""
75_icm_comparison.py -- Compare PPO vs PPO-ICM on Mario ASCII.

Runs both agents for 100 episodes each and shows the difference
in exploration behavior, especially on Tier 2 (gaps requiring jumps).
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.games.mario.mario_simulator import MarioSimulator, Action
from src.games.mario.mario_adapter import MarioAdapter
from src.games.mario.mario_curriculum import MarioCurriculum
from src.games.mario.mario_agent import MarioRLAgent
from src.games.mario.mario_icm_agent import MarioICMAgent

EPISODES = 100
MAX_STEPS = 300


def run_agent(agent_cls, agent_kwargs, label, seed=42):
    """Train an agent and return metrics."""
    np.random.seed(seed)

    adapter = MarioAdapter()
    curriculum = MarioCurriculum(
        start_tier=1, advance_threshold=0.7,
        window_size=20, seed=seed,
    )

    agent = agent_cls(**agent_kwargs)
    use_icm = hasattr(agent, 'learn_with_next_obs')

    rewards = []
    wins = []
    tier_history = []
    unique_states = set()  # Track state diversity (exploration metric)
    t0 = time.perf_counter()

    for ep in range(EPISODES):
        level = curriculum.next_level()
        obs = adapter.reset(level)
        agent.reset()
        ep_reward = 0.0

        for step in range(MAX_STEPS):
            action = agent.step(obs)
            next_obs, reward, done, info = adapter.step(action)
            ep_reward += reward

            # Track exploration: hash Mario's position
            stats = adapter.stats()
            state_key = (curriculum.tier, stats["mario_pos"][0], stats["mario_pos"][1])
            unique_states.add(state_key)

            if use_icm:
                agent.learn_with_next_obs(reward, done, next_obs)
            else:
                agent.learn(reward, done)

            obs = next_obs
            if done:
                break

        won = level.won
        progress = level.max_x_reached / max(1, level.width)
        rewards.append(ep_reward)
        wins.append(int(won))
        tier_history.append(curriculum.tier)

        curriculum.record_result(won=won, progress=progress, steps=step+1, level=level)

        if curriculum.should_advance():
            old_tier = curriculum.tier
            new_tier = curriculum.advance()
            print(f"  [{label}] >>> ADVANCED tier {old_tier} -> {new_tier}")

        if ep % 20 == 0 or ep == EPISODES - 1:
            recent_r = rewards[-20:]
            recent_w = wins[-20:]
            elapsed = time.perf_counter() - t0
            print(f"  [{label}] Ep {ep:3d} | tier={curriculum.tier} "
                  f"| avg_r={np.mean(recent_r):+6.2f} "
                  f"| win={np.mean(recent_w):.0%} "
                  f"| unique_pos={len(unique_states)} "
                  f"| {elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "rewards": rewards,
        "wins": wins,
        "tiers": tier_history,
        "unique_states": len(unique_states),
        "time": elapsed,
        "final_win_rate": np.mean(wins[-20:]),
        "final_avg_reward": np.mean(rewards[-20:]),
    }


def main():
    print("=" * 60)
    print("  PPO vs PPO-ICM -- MARIO ASCII COMPARISON")
    print("=" * 60)
    print()

    # Run PPO baseline
    print("  === PPO BASELINE ===")
    ppo_result = run_agent(
        MarioRLAgent,
        {"obs_dim": 378, "n_actions": 6, "hidden1": 128, "hidden2": 64,
         "lr": 3e-4, "gamma": 0.99, "rollout_length": 128},
        "PPO",
        seed=42,
    )

    print()
    print("  === PPO-ICM (CURIOSITY) ===")
    icm_result = run_agent(
        MarioICMAgent,
        {"obs_dim": 378, "n_actions": 6, "hidden1": 128, "hidden2": 64,
         "lr": 3e-4, "gamma": 0.99, "rollout_length": 128,
         "icm_feature_dim": 32, "icm_hidden_dim": 64,
         "icm_lr": 1e-3, "intrinsic_lambda": 0.5},
        "ICM",
        seed=42,
    )

    # Compare results
    print()
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25s} {'PPO':>10s}  {'PPO-ICM':>10s}")
    print(f"  {'-'*25} {'-'*10}  {'-'*10}")
    print(f"  {'Final win rate':<25s} {ppo_result['final_win_rate']:>9.0%}  {icm_result['final_win_rate']:>9.0%}")
    print(f"  {'Final avg reward':<25s} {ppo_result['final_avg_reward']:>+9.2f}  {icm_result['final_avg_reward']:>+9.2f}")
    print(f"  {'Unique positions':<25s} {ppo_result['unique_states']:>10d}  {icm_result['unique_states']:>10d}")
    print(f"  {'Training time (s)':<25s} {ppo_result['time']:>10.0f}  {icm_result['time']:>10.0f}")
    print(f"  {'Max tier reached':<25s} {max(ppo_result['tiers']):>10d}  {max(icm_result['tiers']):>10d}")
    print()

    # Highlight exploration difference
    exploration_gain = icm_result['unique_states'] / max(1, ppo_result['unique_states'])
    print(f"  Exploration gain: ICM found {exploration_gain:.1f}x more unique positions")
    print("=" * 60)


if __name__ == "__main__":
    main()
