"""
mario_hpo.py — Bayesian Hyperparameter Optimization for MarioICMAgent.

Wraps BayesianOptimizer + a custom MarioParameterSpace to auto-tune:
  - PPO learning rate, gamma, rollout_length
  - ICM feature_dim, hidden_dim, learning rate
  - intrinsic_lambda (extrinsic vs curiosity balance)
  - Policy network hidden1, hidden2

The objective function runs a short evaluation rollout (N episodes on a fixed
level) and returns a composite score:
  score = avg_columns_reached + 5 * win_rate - 0.2 * avg_steps_to_die

Key design decisions:
  - Each HP config gets `eval_episodes` rollouts on a FIXED seed level
    so results are comparable across configs.
  - Rollout is capped at `max_steps_per_episode` (default 200) for speed.
  - GP is pre-warmed with `n_initial_random` random configs before EI kicks in.
  - Results saved to JSON for later inspection / warm-starting.

Usage:
    python examples/run_mario_hpo.py --trials 30 --eval-episodes 20 --tier 3
    python examples/run_mario_hpo.py --trials 50 --eval-episodes 15 --tier 5 --seed 99
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.meta_learning.bayesian_optimizer import BayesianOptimizer
from src.games.mario.mario_generator import MarioLevelGenerator
from src.games.mario.mario_adapter import MarioAdapter
from src.games.mario.mario_agent import MarioAgent, make_mario_agent


# ═══════════════════════════════════════════════════════════════
# MARIO PARAMETER SPACE
# ═══════════════════════════════════════════════════════════════

class MarioParameterSpace:
    """
    Hyperparameter search space for MarioICMAgent.

    Structured to be compatible with BayesianOptimizer's
    sample_random / to_array / from_array API.
    """

    # (name, low, high, type, default, description)
    PARAMS = [
        # PPO core
        ("lr",               1e-4,  1e-2,  "log",   3e-4,  "PPO policy learning rate"),
        ("gamma",            0.90,  0.999, "cont",  0.99,  "Discount factor"),
        ("rollout_length",   32,    256,   "int",   128,   "PPO rollout steps before update"),
        # Policy network size
        ("hidden1",          64,    256,   "int",   128,   "Policy hidden layer 1 size"),
        ("hidden2",          32,    128,   "int",   64,    "Policy hidden layer 2 size"),
        # ICM
        ("icm_lr",           1e-4,  1e-2,  "log",   1e-3,  "ICM learning rate"),
        ("icm_feature_dim",  16,    64,    "int",   32,    "ICM feature embedding size"),
        ("icm_hidden_dim",   32,    128,   "int",   64,    "ICM hidden layer size"),
        ("intrinsic_lambda", 0.01,  1.0,   "cont",  0.5,   "Weight for curiosity reward"),
    ]

    def __init__(self):
        self._names = [p[0] for p in self.PARAMS]

    def sample_random(self) -> Dict[str, float]:
        """Sample a random configuration."""
        config = {}
        for name, low, high, ptype, default, _ in self.PARAMS:
            if ptype == "log":
                config[name] = float(np.exp(np.random.uniform(np.log(low), np.log(high))))
            elif ptype == "int":
                config[name] = int(np.random.randint(low, high + 1))
            else:  # cont
                config[name] = float(np.random.uniform(low, high))
        return config

    def get_default_config(self) -> Dict[str, float]:
        """Return hand-tuned defaults."""
        return {name: default for name, _, _, _, default, _ in self.PARAMS}

    def to_array(self, config: Dict[str, float]) -> np.ndarray:
        """Convert config dict to normalized array for GP input."""
        arr = []
        for name, low, high, ptype, _, _ in self.PARAMS:
            v = config[name]
            if ptype == "log":
                # Map log-scale to [0, 1]
                v_norm = (np.log(v) - np.log(low)) / (np.log(high) - np.log(low))
            else:
                v_norm = (v - low) / (high - low)
            arr.append(float(np.clip(v_norm, 0.0, 1.0)))
        return np.array(arr, dtype=np.float64)

    def from_array(self, arr: np.ndarray) -> Dict[str, float]:
        """Reconstruct config dict from normalized array."""
        config = {}
        for i, (name, low, high, ptype, default, _) in enumerate(self.PARAMS):
            v_norm = float(np.clip(arr[i], 0.0, 1.0))
            if ptype == "log":
                v = float(np.exp(v_norm * (np.log(high) - np.log(low)) + np.log(low)))
            elif ptype == "int":
                v = int(round(v_norm * (high - low) + low))
                v = int(np.clip(v, low, high))
            else:
                v = float(v_norm * (high - low) + low)
            config[name] = v
        return config

    def pretty(self, config: Dict[str, float]) -> str:
        """Human-readable config string."""
        parts = []
        for name, _, _, ptype, default, _ in self.PARAMS:
            v = config.get(name, default)
            if ptype == "log" or ptype == "cont":
                if v < 0.01:
                    parts.append(f"{name}={v:.2e}")
                else:
                    parts.append(f"{name}={v:.4f}")
            else:
                parts.append(f"{name}={int(v)}")
        return "  " + "\n  ".join(parts)

    def count_parameters(self) -> int:
        return len(self.PARAMS)


# ═══════════════════════════════════════════════════════════════
# EVALUATION OBJECTIVE
# ═══════════════════════════════════════════════════════════════

class MarioHPOObjective:
    """
    Evaluates a hyperparameter configuration on a fixed set of Mario levels.

    Returns a scalar score:
        score = avg_col_fraction + 5 * win_rate - 0.2 * avg_death_fraction
    """

    def __init__(
        self,
        tier: int = 3,
        eval_episodes: int = 20,
        max_steps: int = 200,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.tier = tier
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.seed = seed
        self.verbose = verbose

        # Pre-generate a fixed pool of levels (same for every config)
        gen = MarioLevelGenerator(seed=seed)
        n_levels = max(3, eval_episodes // 5)
        self._levels = []
        for _ in range(n_levels):
            sim = gen.generate(tier=tier)
            if sim is not None:
                self._levels.append(sim.save())

        assert self._levels, f"Failed to generate any levels at tier {tier}"
        self._adapter = MarioAdapter()
        self._rng = np.random.RandomState(seed)

        # Track history
        self.history: List[Dict[str, Any]] = []

    def __call__(self, config: Dict[str, float]) -> float:
        """Evaluate one hyperparameter configuration. Returns score."""
        t0 = time.time()

        # Build agent with this config
        agent = agent_from_config(config)

        col_fractions = []
        wins = []
        step_fractions = []

        for ep in range(self.eval_episodes):
            # Rotate through pre-generated levels
            level_state = self._levels[ep % len(self._levels)]

            # Restore level from saved state
            # Use a real sim with the correct grid shape
            from src.games.mario.mario_simulator import MarioSimulator
            grid_shape = level_state["grid"].shape  # type: ignore[index]
            sim = MarioSimulator(np.zeros(grid_shape, dtype=np.uint8))
            sim.load(level_state)

            obs = self._adapter.reset(sim)
            agent.reset()

            step = 0
            for step in range(self.max_steps):
                action = agent.step(obs)
                obs_next, reward, done, info = self._adapter.step(action)
                agent.learn_with_next_obs(reward, done, obs_next)
                obs = obs_next
                if done:
                    break

            col_fractions.append(sim.mario_col / max(sim.width - 1, 1))
            wins.append(1 if sim.won else 0)
            step_fractions.append(step / self.max_steps)

        avg_col = float(np.mean(col_fractions))
        win_rate = float(np.mean(wins))
        avg_step = float(np.mean(step_fractions))

        # Composite score: progress + wins; penalize dying fast
        score = avg_col + 5.0 * win_rate - 0.2 * (1.0 - avg_step)

        elapsed = time.time() - t0
        entry = {
            "config": {k: float(v) for k, v in config.items()},
            "score": score,
            "avg_col": avg_col,
            "win_rate": win_rate,
            "avg_step": avg_step,
            "eval_time": round(elapsed, 2),
        }
        self.history.append(entry)

        if self.verbose:
            print(f"    score={score:.4f}  col={avg_col:.2f}  "
                  f"win={win_rate:.0%}  step={avg_step:.2f} ({elapsed:.1f}s)")

        return score


# ═══════════════════════════════════════════════════════════════
# HPO RUNNER
# ═══════════════════════════════════════════════════════════════

def run_hpo(
    tier: int = 3,
    n_trials: int = 30,
    n_initial_random: int = 8,
    eval_episodes: int = 20,
    max_steps: int = 200,
    seed: int = 42,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, float], float, List[Dict]]:
    """
    Run full Bayesian HPO for MarioICMAgent.

    Args:
        tier: Level generator tier (1-7, higher = harder).
        n_trials: Total evaluation budget.
        n_initial_random: GP warm-up random trials.
        eval_episodes: Episodes per config evaluation.
        max_steps: Max steps per episode.
        seed: RNG seed.
        save_path: If set, save results to JSON here.
        verbose: Print progress.

    Returns:
        (best_config, best_score, history)
    """
    param_space = MarioParameterSpace()
    objective = MarioHPOObjective(
        tier=tier,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print(f"═══ Mario Bayesian HPO ═══")
        print(f"  Trials: {n_trials}  (random: {n_initial_random}, BO: {n_trials - n_initial_random})")
        print(f"  Tier: {tier}  |  Eval episodes/config: {eval_episodes}")
        print(f"  Search space: {param_space.count_parameters()} parameters")
        print(f"  Level pool: {len(objective._levels)} levels")
        print()

    optimizer = BayesianOptimizer(
        parameter_space=param_space,
        objective_function=objective,
        n_initial_random=n_initial_random,
    )

    best_config, best_score = optimizer.optimize(n_trials=n_trials, verbose=verbose)

    if verbose:
        print(f"\n═══ Best Config (score={best_score:.4f}) ═══")
        print(param_space.pretty(best_config))

    if save_path:
        results = {
            "best_config": {k: float(v) for k, v in best_config.items()},
            "best_score": float(best_score),
            "history": objective.history,
            "params": {
                "tier": tier, "n_trials": n_trials,
                "eval_episodes": eval_episodes, "seed": seed,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\n  Results saved to: {save_path}")

    return best_config, best_score, objective.history


def load_best_config(path: str) -> Optional[Dict[str, float]]:
    """Load best config from a saved HPO results JSON."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("best_config")


def agent_from_config(config: Dict[str, float]) -> MarioAgent:
    """Build a MarioAgent (numpy + ICM) from an HPO config dict."""
    return MarioAgent.from_hpo_config(
        {
            "hidden1": float(config.get("hidden1", 128)),
            "hidden2": float(config.get("hidden2", 64)),
            "lr": float(config.get("lr", 3e-4)),
            "gamma": float(config.get("gamma", 0.99)),
            "rollout_length": float(config.get("rollout_length", 128)),
            "icm_feature_dim": float(config.get("icm_feature_dim", 32)),
            "icm_hidden_dim": float(config.get("icm_hidden_dim", 64)),
            "icm_lr": float(config.get("icm_lr", 1e-3)),
            "intrinsic_lambda": float(config.get("intrinsic_lambda", 0.5)),
        },
        obs_dim=378,
        n_actions=8,
        backend="numpy",
        curiosity=True,
    )
