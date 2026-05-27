"""
level_sampler.py — Blends canonical, GAN, and procedural levels.

Provides a single `sample()` call that ThrongVecEnv workers use on
reset to get a fresh MarioSimulator from a mixed training distribution.

Training mix (default, adjustable):
  - Canonical World 1 levels: starts at 75%, decays toward 50%
  - GAN-generated levels:     the remainder after canonical + procedural
  - Procedural levels:        fixed 20%

The canonical ratio decays asymptotically:
  ratio(step) = target + (start - target) * exp(-step / half_life)

This means early training is heavily grounded in real World 1 geometry,
then gradually shifts to more GAN variation for generalization.

Usage:
    sampler = LevelSampler(canonical_start=0.75, canonical_target=0.50)
    
    def make_env():
        from src.games.mario.sim_noise import NoisyMarioSim
        return NoisyMarioSim(sampler.sample_factory())
    
    vec = ThrongVecEnv(make_env, n_envs=n, obs_dim=378)
"""

from __future__ import annotations
from copy import deepcopy
from typing import Callable, Dict, List, Optional
import math
import numpy as np

from .mario_simulator import MarioSimulator
from .mario_gan import MarioGAN
from .mario_generator import MarioLevelGenerator


class LevelSampler:
    """
    Blends canonical levels, GAN levels, and procedural levels
    with an asymptotically decaying canonical ratio.

    canonical_ratio(t) = target + (start - target) * exp(-t / half_life)
    procedural_ratio   = fixed
    gan_ratio           = 1.0 - canonical_ratio(t) - procedural_ratio
    """

    def __init__(
        self,
        canonical_start: float = 0.75,
        canonical_target: float = 0.50,
        procedural_ratio: float = 0.20,
        decay_half_life: int = 50_000,
        gan: Optional[MarioGAN] = None,
        generator: Optional[MarioLevelGenerator] = None,
        tier_range: tuple = (3, 7),
        seed: Optional[int] = None,
    ):
        """
        Args:
            canonical_start:  Initial canonical level ratio (e.g. 0.75)
            canonical_target: Asymptotic target ratio (e.g. 0.50)
            procedural_ratio: Fixed procedural level ratio (e.g. 0.20)
            decay_half_life:  Steps for canonical ratio to decay halfway
            gan:              MarioGAN instance (created if None)
            generator:        MarioLevelGenerator (created if None)
            tier_range:       (min_tier, max_tier) for generated levels
            seed:             Random seed
        """
        assert canonical_target + procedural_ratio <= 1.0, \
            "canonical_target + procedural_ratio must be <= 1.0"
        assert canonical_start >= canonical_target, \
            "canonical_start must be >= canonical_target"

        self.canonical_start = canonical_start
        self.canonical_target = canonical_target
        self.procedural_ratio = procedural_ratio
        self.decay_half_life = decay_half_life
        self.tier_range = tier_range

        self.gan = gan or MarioGAN()
        self.generator = generator or MarioLevelGenerator(seed=seed)
        self.rng = np.random.RandomState(seed)

        # Step counter for decay schedule
        self._step = 0

        # Load canonical levels (lazy)
        self._canonical: Optional[Dict[str, MarioSimulator]] = None

        # Stats
        self._counts = {'canonical': 0, 'gan': 0, 'procedural': 0}

    def _ensure_canonical(self):
        if self._canonical is None:
            from .world1_levels import load_world1
            self._canonical = load_world1()

    @property
    def canonical_ratio(self) -> float:
        """Current canonical ratio (decays from start → target)."""
        delta = self.canonical_start - self.canonical_target
        decay = math.exp(-self._step / max(self.decay_half_life, 1))
        return self.canonical_target + delta * decay

    @property
    def gan_ratio(self) -> float:
        """Current GAN ratio (fill between canonical and procedural)."""
        return max(0.0, 1.0 - self.canonical_ratio - self.procedural_ratio)

    def ratios(self) -> Dict[str, float]:
        """Current sampling ratios."""
        return {
            'canonical': round(self.canonical_ratio, 3),
            'gan': round(self.gan_ratio, 3),
            'procedural': round(self.procedural_ratio, 3),
            'step': self._step,
        }

    def advance(self, n: int = 1) -> None:
        """Advance the decay schedule by n steps."""
        self._step += n

    def sample(self) -> MarioSimulator:
        """
        Draw one MarioSimulator from the blended distribution.
        Advances the step counter by 1.
        """
        self._ensure_canonical()
        self._step += 1

        r = self.rng.random()
        cr = self.canonical_ratio
        pr = self.procedural_ratio

        if r < cr:
            return self._sample_canonical()
        elif r < cr + pr:
            return self._sample_procedural()
        else:
            return self._sample_gan()

    def sample_factory(self) -> Callable[[], MarioSimulator]:
        """
        Return a callable that produces a fresh sim on each call.
        Use this with NoisyMarioSim: NoisyMarioSim(sampler.sample_factory())
        """
        def factory():
            return self.sample()
        return factory

    # ── Internal samplers ─────────────────────────────────────────

    def _sample_canonical(self) -> MarioSimulator:
        """Deep-copy a random World 1 level."""
        self._counts['canonical'] += 1
        key = self.rng.choice(list(self._canonical.keys()))
        return deepcopy(self._canonical[key])

    def _sample_gan(self) -> MarioSimulator:
        """Generate a level from the GAN."""
        self._counts['gan'] += 1
        tier = self.rng.randint(self.tier_range[0], self.tier_range[1] + 1)
        sim = self.gan.generate(tier=tier)
        if sim is not None:
            return sim
        # Fallback to procedural if GAN fails
        return self._sample_procedural()

    def _sample_procedural(self) -> MarioSimulator:
        """Generate a level from the procedural generator."""
        self._counts['procedural'] += 1
        tier = self.rng.randint(self.tier_range[0], self.tier_range[1] + 1)
        sim = self.generator.generate(tier=tier)
        if sim is not None:
            return sim
        # Ultimate fallback: flat ground
        return MarioSimulator.from_flat_ground(n_screens=2)

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Sampling statistics."""
        total = sum(self._counts.values()) or 1
        return {
            'step': self._step,
            'ratios': self.ratios(),
            'counts': dict(self._counts),
            'actual_ratios': {
                k: round(v / total, 3) for k, v in self._counts.items()
            },
        }
