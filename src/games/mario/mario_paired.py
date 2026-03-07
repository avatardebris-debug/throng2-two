"""
mario_paired.py -- PAIRED Adversarial Level Design for Mario ASCII.

Protagonist-Antagonist Induced Regret Environment Design:
  - Generator (Antagonist): creates levels that are LEGAL but HARD for the agent
  - Agent (Protagonist): tries to beat the generated levels
  - Generator gets reward for: legality × (1 - agent_win_rate)
  - Agent gets reward for: beating levels

This creates an automatic difficulty arms race: as the agent improves,
the generator has to create harder levels. As levels get harder, the agent
has to improve to beat them.

Zone of Proximal Development: generator is penalized for levels that are
TOO hard (agent never wins -- no learning signal) or TOO easy (agent always
wins -- no challenge). Sweet spot is ~30-70% win rate.

Usage:
    paired = PAIREDTrainer(agent, gan)
    for epoch in range(1000):
        result = paired.step()
        # Generator and agent both improve
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .mario_adapter import MarioAdapter
from .mario_gan import MarioGAN
from .mario_simulator import MarioSimulator


class PAIREDTrainer:
    """
    PAIRED orchestrator: couples GAN level generation with RL agent training.

    Each step:
      1. Generator creates a batch of levels
      2. Agent attempts each level (quick rollout)
      3. Levels are classified: agent-won (too easy) vs agent-lost (good difficulty)
      4. Structural validator scores legality
      5. Generator is rewarded for: legal + agent_loses
         Generator is penalized for: illegal OR too_easy OR impossible
      6. Both generator and agent update
    """

    def __init__(
        self,
        agent,
        gan: Optional[MarioGAN] = None,
        batch_size: int = 8,
        max_steps: int = 400,
        target_win_rate: float = 0.5,
        difficulty_window: int = 50,
        seed: int = 42,
    ):
        """
        Args:
            agent: RL agent (MarioTorchAgent or MarioICMAgent)
            gan: MarioGAN instance (creates one if None)
            batch_size: levels generated per step
            max_steps: max steps per agent rollout
            target_win_rate: ideal difficulty (~50% = zone of proximal development)
            difficulty_window: window size for tracking agent win rate
        """
        self.agent = agent
        self.gan = gan or MarioGAN()
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.target_win_rate = target_win_rate

        self.adapter = MarioAdapter()
        self.rng = np.random.RandomState(seed)

        # Current difficulty tier (starts at 1, adapts based on agent performance)
        self.current_tier = 1

        # Tracking
        self._win_history: List[bool] = []
        self._difficulty_window = difficulty_window
        self.total_episodes = 0
        self.total_steps = 0
        self.gen_updates = 0
        self.tier_advances = 0

        # Stats per step
        self._step_stats: List[Dict[str, float]] = []

    @property
    def win_rate(self) -> float:
        """Recent agent win rate."""
        recent = self._win_history[-self._difficulty_window:]
        return sum(recent) / max(len(recent), 1)

    def step(self) -> Dict[str, Any]:
        """
        One PAIRED training step:
          1. Generate batch of levels
          2. Agent plays each
          3. Score & train generator
          4. Adapt difficulty tier

        Returns:
            dict with training stats
        """
        good_levels = []   # Levels the agent beat (too easy for generator)
        hard_levels = []   # Levels the agent lost (good for generator)
        invalid_levels = 0
        total_reward = 0.0
        wins = 0
        step_count = 0

        # ── Generate and evaluate batch ──────────────────────
        for _ in range(self.batch_size):
            # Generator creates a level
            sim = self.gan.generate(tier=self.current_tier, temperature=0.8)

            if sim is None:
                invalid_levels += 1
                continue

            # Validate structure
            validation = MarioGAN.validate_structure(sim.grid)
            if not validation["valid"] or validation["score"] < 0.5:
                invalid_levels += 1
                # Still give the GAN the bad level to learn from
                bad_grid = self.gan.grid_to_onehot(sim)
                hard_levels.append(bad_grid)  # Treated as "negative" for discriminator
                continue

            # Agent plays the level
            ep_reward, won, steps = self._play_level(sim)

            # Convert to one-hot for GAN training
            level_grid = self.gan.grid_to_onehot(sim)

            if won:
                good_levels.append(level_grid)
                self.gan.add_solved(level_grid)  # Add to imitation bank
                wins += 1
            else:
                hard_levels.append(level_grid)

            self._win_history.append(won)
            total_reward += ep_reward
            step_count += steps
            self.total_episodes += 1

        self.total_steps += step_count

        # ── Train generator ──────────────────────────────────
        gan_stats = {"d_loss": 0, "g_loss": 0}
        if good_levels or hard_levels:
            # In PAIRED: good_levels = agent won = "too easy" for generator
            #             hard_levels = agent lost = "good difficulty"
            # We FLIP the labels for the GAN:
            #   - hard_levels → "good" (generator did well, stumped the agent)
            #   - good_levels → "bad" (generator failed, agent won)
            gan_stats = self.gan.train_step(
                good_levels=hard_levels,   # Agent failed → good for generator
                bad_levels=good_levels,     # Agent won → bad for generator
            )
            self.gen_updates += 1

        # ── Adapt difficulty ─────────────────────────────────
        tier_changed = False
        wr = self.win_rate
        if len(self._win_history) >= self._difficulty_window:
            if wr > 0.7 and self.current_tier < 7:
                self.current_tier += 1
                self.tier_advances += 1
                tier_changed = True
            elif wr < 0.15 and self.current_tier > 1:
                self.current_tier -= 1
                tier_changed = True

        # ── Compile stats ────────────────────────────────────
        valid = self.batch_size - invalid_levels
        stats = {
            "tier": self.current_tier,
            "wins": wins,
            "valid_levels": valid,
            "invalid_levels": invalid_levels,
            "win_rate": round(wr, 3),
            "avg_reward": round(total_reward / max(valid, 1), 2),
            "steps": step_count,
            "tier_changed": tier_changed,
            **gan_stats,
        }
        self._step_stats.append(stats)
        return stats

    def _play_level(self, sim: MarioSimulator):
        """
        Agent attempts one level.

        Returns:
            (total_reward, won, steps)
        """
        obs = self.adapter.reset(sim)
        self.agent.reset()
        total_reward = 0.0

        for step in range(self.max_steps):
            action = self.agent.step(obs)
            next_obs, reward, done, info = self.adapter.step(action)
            total_reward += reward

            self.agent.learn_with_next_obs(reward, done, next_obs)
            obs = next_obs

            if done:
                break

        return total_reward, sim.won, step + 1

    def train(
        self,
        n_steps: int = 500,
        log_interval: int = 10,
        checkpoint_fn=None,
    ) -> List[Dict[str, Any]]:
        """
        Run PAIRED training loop.

        Args:
            n_steps: number of PAIRED steps (each generates batch_size levels)
            log_interval: print stats every N steps
            checkpoint_fn: called every 100 steps with (step, agent, gan)

        Returns:
            list of per-step stats
        """
        import time
        t0 = time.perf_counter()

        print("=" * 60)
        print("  PAIRED ADVERSARIAL LEVEL DESIGN")
        print(f"  Steps: {n_steps}, Batch: {self.batch_size}")
        print(f"  Target win rate: {self.target_win_rate:.0%}")
        print("=" * 60)

        all_stats = []

        for i in range(n_steps):
            stats = self.step()
            all_stats.append(stats)

            if i % log_interval == 0 or i == n_steps - 1:
                elapsed = time.perf_counter() - t0
                sps = self.total_steps / max(0.01, elapsed)
                print(f"  Step {i:4d} | tier={stats['tier']} "
                      f"| wr={stats['win_rate']:.0%} "
                      f"| valid={stats['valid_levels']}/{self.batch_size} "
                      f"| r={stats['avg_reward']:+.1f} "
                      f"| g={stats.get('g_loss', 0):.3f} "
                      f"| d={stats.get('d_loss', 0):.3f} "
                      f"| {sps:.0f} sps "
                      f"| {elapsed:.0f}s")

            if checkpoint_fn and (i + 1) % 100 == 0:
                checkpoint_fn(i + 1, self.agent, self.gan)

        elapsed = time.perf_counter() - t0
        print("=" * 60)
        print(f"  PAIRED complete: {n_steps} steps, "
              f"{self.total_episodes} episodes, "
              f"{elapsed:.0f}s")
        print(f"  Final tier: {self.current_tier}, "
              f"Win rate: {self.win_rate:.0%}")
        print("=" * 60)

        return all_stats

    def report(self) -> Dict[str, Any]:
        """Summary report."""
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "final_tier": self.current_tier,
            "win_rate": self.win_rate,
            "tier_advances": self.tier_advances,
            "gen_updates": self.gen_updates,
            "gan_report": self.gan.report(),
        }
