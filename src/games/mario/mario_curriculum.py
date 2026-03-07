"""
mario_curriculum.py -- Tiered curriculum trainer for Mario ASCII engine.

Manages graduated training progression, advancing difficulty when
the agent demonstrates competence at the current tier.

Follows the LoloCurriculum pattern from throng5:
- Track per-tier statistics (wins, deaths, progress)
- Advance when success_rate >= threshold over window
- GAN integration: seed solved bank with completable levels

Usage:
    from src.games.mario.mario_curriculum import MarioCurriculum

    c = MarioCurriculum()
    level = c.next_level()
    # ... agent plays level ...
    c.record_result(won=True, progress=0.95, steps=150)
    if c.should_advance():
        print(f"Advanced to tier {c.advance()}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .mario_generator import MarioLevelGenerator
from .mario_gan import MarioGAN
from .mario_simulator import MarioSimulator


class MarioCurriculum:
    """
    Curriculum manager: generates levels at the current tier,
    tracks agent performance, and advances when ready.

    Integrates with MarioGAN:
    - Early tiers use procedural generator exclusively
    - As solved bank grows, GAN-generated levels are mixed in
    - GAN trains from solved/failed level data
    """

    def __init__(
        self,
        start_tier: int = 1,
        advance_threshold: float = 0.8,
        window_size: int = 50,
        gan_mix_ratio: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.tier = start_tier
        self.advance_threshold = advance_threshold
        self.window_size = window_size
        self.gan_mix_ratio = gan_mix_ratio

        self.generator = MarioLevelGenerator(seed=seed)
        self.gan = MarioGAN()
        self.rng = np.random.RandomState(seed)

        # Per-tier tracking
        self._tier_stats: Dict[int, Dict[str, List]] = {}
        self._ensure_tier_stats(start_tier)

        # Global counters
        self.total_episodes = 0
        self.total_wins = 0

        # GAN training
        self._good_levels: List[np.ndarray] = []
        self._bad_levels: List[np.ndarray] = []
        self._gan_trained = False

    def _ensure_tier_stats(self, tier: int):
        if tier not in self._tier_stats:
            self._tier_stats[tier] = {
                "wins": [],
                "progress": [],
                "steps": [],
            }

    def next_level(self) -> MarioSimulator:
        """
        Generate the next training level.

        Mixes procedural and GAN-generated levels based on
        GAN training progress and current tier.
        """
        # Try GAN generation if we have trained data
        if (self._gan_trained
                and self.rng.random() < self.gan_mix_ratio
                and len(self.gan.solved_bank) > 10):
            sim = self.gan.generate(tier=self.tier)
            if sim is not None and sim.is_completable():
                return sim

        # Fall back to procedural generator
        level = self.generator.generate(tier=self.tier)
        if level is None:
            # Emergency: generate flat ground
            level = MarioSimulator.from_flat_ground()
        return level

    def record_result(
        self,
        won: bool,
        progress: float = 0.0,
        steps: int = 0,
        level: Optional[MarioSimulator] = None,
    ) -> Dict[str, Any]:
        """
        Record the result of an episode.

        Args:
            won: Did the agent reach the flag?
            progress: Fraction of level traversed (0-1)
            steps: Number of steps taken
            level: The level that was played (for GAN training)

        Returns:
            Dict with current curriculum status
        """
        self._ensure_tier_stats(self.tier)
        stats = self._tier_stats[self.tier]
        stats["wins"].append(int(won))
        stats["progress"].append(progress)
        stats["steps"].append(steps)

        self.total_episodes += 1
        if won:
            self.total_wins += 1

        # Feed GAN
        if level is not None:
            onehot = self.gan.grid_to_onehot(level)
            if won:
                self.gan.add_solved(onehot)
                self._good_levels.append(onehot)
            else:
                self._bad_levels.append(onehot)

            # Trim buffers
            if len(self._good_levels) > 200:
                self._good_levels = self._good_levels[-200:]
            if len(self._bad_levels) > 200:
                self._bad_levels = self._bad_levels[-200:]

        return self.status()

    def train_gan(self) -> Dict[str, float]:
        """
        Train the GAN on accumulated level data.

        Should be called periodically (e.g., every 20 episodes).
        """
        result = {}

        # Pretrain if enough solved levels
        if len(self.gan.solved_bank) >= 5 and not self._gan_trained:
            pt_result = self.gan.pretrain_from_solved(epochs=20, batch_size=8)
            result["pretrain"] = pt_result

        # Adversarial training
        if self._good_levels and self._bad_levels:
            # Sample batches
            n_good = min(8, len(self._good_levels))
            n_bad = min(8, len(self._bad_levels))
            good = [self._good_levels[i] for i in
                    self.rng.choice(len(self._good_levels), n_good, replace=False)]
            bad = [self._bad_levels[i] for i in
                   self.rng.choice(len(self._bad_levels), n_bad, replace=False)]

            train_result = self.gan.train_step(good, bad)
            result["train"] = train_result
            self._gan_trained = True

        result["gan_report"] = self.gan.report()
        return result

    def should_advance(self) -> bool:
        """Check if the agent should advance to the next tier."""
        if self.tier >= 7:
            return False
        stats = self._tier_stats.get(self.tier, {}).get("wins", [])
        if len(stats) < self.window_size:
            return False
        recent = stats[-self.window_size:]
        return np.mean(recent) >= self.advance_threshold

    def advance(self) -> int:
        """Advance to the next tier."""
        if self.tier < 7:
            self.tier += 1
            self._ensure_tier_stats(self.tier)
            self.generator.complexity_tier = self.tier
        return self.tier

    def status(self) -> Dict[str, Any]:
        """Current curriculum status."""
        stats = self._tier_stats.get(self.tier, {"wins": [], "progress": [], "steps": []})
        wins = stats["wins"]
        progress = stats["progress"]

        recent_wins = wins[-self.window_size:] if wins else []
        recent_progress = progress[-self.window_size:] if progress else []

        return {
            "tier": self.tier,
            "total_episodes": self.total_episodes,
            "total_wins": self.total_wins,
            "tier_episodes": len(wins),
            "tier_win_rate": float(np.mean(recent_wins)) if recent_wins else 0.0,
            "tier_avg_progress": float(np.mean(recent_progress)) if recent_progress else 0.0,
            "advance_ready": self.should_advance(),
            "gan_trained": self._gan_trained,
            "solved_bank_size": len(self.gan.solved_bank),
        }

    def report(self) -> Dict[str, Any]:
        """Full curriculum report."""
        result = {
            "current_tier": self.tier,
            "total_episodes": self.total_episodes,
            "total_wins": self.total_wins,
            "tier_stats": {},
        }
        for tier, stats in self._tier_stats.items():
            wins = stats["wins"]
            result["tier_stats"][tier] = {
                "episodes": len(wins),
                "win_rate": float(np.mean(wins)) if wins else 0.0,
                "avg_progress": float(np.mean(stats["progress"])) if stats["progress"] else 0.0,
                "avg_steps": float(np.mean(stats["steps"])) if stats["steps"] else 0.0,
            }
        result["gan"] = self.gan.report()
        result["generator"] = self.generator.report()
        return result
