"""
mario_paired.py -- PAIRED Adversarial Level Design for Mario ASCII.

Protagonist-Antagonist Induced Regret Environment Design:
  - Generator (Antagonist): creates levels that are LEGAL but HARD for the agent
  - Agent (Protagonist): tries to beat the generated levels
  - Generator gets reward for: legality * (1 - agent_win_rate)
  - Agent gets reward for: beating levels

WARMUP PHASE: Before adversarial training, the GAN is pre-trained on
curriculum-generated levels so it knows what valid levels look like.

Usage:
    paired = PAIREDTrainer(agent, gan)
    for epoch in range(1000):
        result = paired.step()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .mario_adapter import MarioAdapter
from .mario_curriculum import MarioCurriculum
from .mario_gan import MarioGAN
from .mario_simulator import MarioSimulator


class PAIREDTrainer:
    """
    PAIRED orchestrator with warmup phase.

    Phase 1 (warmup): Generate curriculum levels, play them, pre-train GAN
    Phase 2 (adversarial): GAN generates levels, agent plays, both improve
    """

    def __init__(
        self,
        agent,
        gan: Optional[MarioGAN] = None,
        batch_size: int = 8,
        max_steps: int = 400,
        target_win_rate: float = 0.5,
        difficulty_window: int = 50,
        warmup_levels: int = 200,
        warmup_pretrain_epochs: int = 100,
        gan_mix_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.agent = agent
        self.gan = gan or MarioGAN()
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.target_win_rate = target_win_rate
        self.gan_mix_ratio = gan_mix_ratio

        self.adapter = MarioAdapter()
        self.curriculum = MarioCurriculum(start_tier=1, advance_threshold=0.7, seed=seed)
        self.rng = np.random.RandomState(seed)

        self.current_tier = 1
        self._win_history: List[bool] = []
        self._difficulty_window = difficulty_window
        self.total_episodes = 0
        self.total_steps = 0
        self.gen_updates = 0
        self.tier_advances = 0
        self._step_stats: List[Dict[str, float]] = []

        # Warmup config
        self._warmup_levels = warmup_levels
        self._warmup_pretrain_epochs = warmup_pretrain_epochs
        self._warmed_up = False

    @property
    def win_rate(self) -> float:
        recent = self._win_history[-self._difficulty_window:]
        return sum(recent) / max(len(recent), 1)

    def warmup(self):
        """
        Phase 1: Generate curriculum levels, play them, pre-train GAN.
        This teaches the GAN what valid Mario levels look like.
        """
        import time
        t0 = time.perf_counter()

        print("  -- WARMUP: Pre-training GAN on curriculum levels --")

        for i in range(self._warmup_levels):
            level = self.curriculum.next_level()

            # Play the level (trains the agent too)
            ep_reward, won, steps = self._play_level(level)
            self.total_episodes += 1
            self.total_steps += steps
            self._win_history.append(won)

            # Convert to one-hot and add to GAN's solved bank
            grid_onehot = self.gan.grid_to_onehot(level)
            self.gan.add_solved(grid_onehot)

            # Track curriculum progression
            progress = level.max_x_reached / max(1, level.width)
            self.curriculum.record_result(won=won, progress=progress,
                                          steps=steps, level=level)
            if self.curriculum.should_advance():
                old = self.curriculum.tier
                new = self.curriculum.advance()
                print(f"    Warmup: tier {old} -> {new}")

            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                wr = self.win_rate
                print(f"    Warmup {i+1}/{self._warmup_levels} "
                      f"| tier={self.curriculum.tier} "
                      f"| wr={wr:.0%} "
                      f"| bank={len(self.gan.solved_bank)} "
                      f"| {elapsed:.0f}s")

        # Pre-train GAN on collected levels
        print(f"  -- Pre-training GAN on {len(self.gan.solved_bank)} levels "
              f"({self._warmup_pretrain_epochs} epochs) --")
        pretrain_result = self.gan.pretrain_from_solved(
            epochs=self._warmup_pretrain_epochs, batch_size=32
        )
        print(f"    Pretrain loss: {pretrain_result['pretrain_loss']:.4f}")

        self.current_tier = self.curriculum.tier
        self._warmed_up = True

        elapsed = time.perf_counter() - t0
        print(f"  -- Warmup complete: {elapsed:.0f}s, "
              f"tier={self.current_tier}, bank={len(self.gan.solved_bank)} --")
        print()

    def step(self) -> Dict[str, Any]:
        """One PAIRED training step with curriculum fallback."""
        good_levels = []
        hard_levels = []
        invalid_levels = 0
        total_reward = 0.0
        wins = 0
        step_count = 0
        gan_generated = 0

        for _ in range(self.batch_size):
            # Decide: GAN level or curriculum level?
            use_gan = self.rng.random() < self.gan_mix_ratio and self._warmed_up

            if use_gan:
                sim = self.gan.generate(tier=self.current_tier, temperature=0.8)
                if sim is None:
                    invalid_levels += 1
                    sim = self.curriculum.next_level()
                else:
                    validation = MarioGAN.validate_structure(sim.grid)
                    if not validation["valid"] or validation["score"] < 0.5:
                        invalid_levels += 1
                        bad_grid = self.gan.grid_to_onehot(sim)
                        hard_levels.append(bad_grid)
                        sim = self.curriculum.next_level()
                    else:
                        gan_generated += 1
            else:
                sim = self.curriculum.next_level()

            # Agent plays the level
            ep_reward, won, steps = self._play_level(sim)

            # Convert to one-hot for GAN training
            level_grid = self.gan.grid_to_onehot(sim)

            if won:
                good_levels.append(level_grid)
                self.gan.add_solved(level_grid)
                wins += 1
            else:
                hard_levels.append(level_grid)

            self._win_history.append(won)
            total_reward += ep_reward
            step_count += steps
            self.total_episodes += 1

            # Track curriculum
            progress = sim.max_x_reached / max(1, sim.width)
            self.curriculum.record_result(won=won, progress=progress,
                                          steps=steps, level=sim)

        self.total_steps += step_count

        # Train generator
        gan_stats = {"d_loss": 0, "g_loss": 0}
        if good_levels or hard_levels:
            gan_stats = self.gan.train_step(
                good_levels=hard_levels,
                bad_levels=good_levels,
            )
            self.gen_updates += 1

        # Adapt tier
        tier_changed = False
        wr = self.win_rate
        if self.curriculum.should_advance():
            old = self.curriculum.tier
            new = self.curriculum.advance()
            self.current_tier = new
            self.tier_advances += 1
            tier_changed = True
        elif wr < 0.15 and self.current_tier > 1:
            self.current_tier -= 1
            tier_changed = True

        # Gradually increase GAN mix as it improves
        if gan_generated > 0 and invalid_levels < self.batch_size // 2:
            self.gan_mix_ratio = min(0.9, self.gan_mix_ratio + 0.01)

        valid = self.batch_size - invalid_levels
        stats = {
            "tier": self.current_tier,
            "wins": wins,
            "valid_levels": valid,
            "invalid_levels": invalid_levels,
            "gan_generated": gan_generated,
            "gan_mix": round(self.gan_mix_ratio, 2),
            "win_rate": round(wr, 3),
            "avg_reward": round(total_reward / max(self.batch_size, 1), 2),
            "steps": step_count,
            "tier_changed": tier_changed,
            **gan_stats,
        }
        self._step_stats.append(stats)
        return stats

    def _play_level(self, sim: MarioSimulator):
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
        import time
        t0 = time.perf_counter()

        print("=" * 60)
        print("  PAIRED ADVERSARIAL LEVEL DESIGN")
        print(f"  Steps: {n_steps}, Batch: {self.batch_size}")
        print(f"  Target win rate: {self.target_win_rate:.0%}")
        print("=" * 60)

        # Phase 1: Warmup
        if not self._warmed_up:
            self.warmup()

        # Phase 2: Adversarial training
        print("  -- Phase 2: Adversarial Training --")
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
                      f"| gan={stats['gan_generated']}/{self.batch_size} "
                      f"| mix={stats['gan_mix']:.0%} "
                      f"| r={stats['avg_reward']:+.1f} "
                      f"| {sps:.0f} sps "
                      f"| {elapsed:.0f}s")

            if stats.get("tier_changed"):
                print(f"  >>> TIER CHANGE -> {stats['tier']}")

            if checkpoint_fn and (i + 1) % 100 == 0:
                checkpoint_fn(i + 1, self.agent, self.gan)

        elapsed = time.perf_counter() - t0
        print("=" * 60)
        print(f"  PAIRED complete: {n_steps} steps, "
              f"{self.total_episodes} episodes, "
              f"{elapsed:.0f}s")
        print(f"  Final tier: {self.current_tier}, "
              f"Win rate: {self.win_rate:.0%}, "
              f"GAN mix: {self.gan_mix_ratio:.0%}")
        print("=" * 60)

        return all_stats

    def report(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "final_tier": self.current_tier,
            "win_rate": self.win_rate,
            "tier_advances": self.tier_advances,
            "gen_updates": self.gen_updates,
            "gan_mix_ratio": self.gan_mix_ratio,
            "gan_report": self.gan.report(),
        }
