"""
contrastive_hpo.py — Bayesian HPO for ContrastiveProjection hyperparameters.

Self-contained GP + Expected Improvement search over 5 hyperparameters.
Does NOT use the existing BayesianOptimizer (which has a hard-coded
ParameterSpace for Nash-pruning/neurogenesis and no objective_function hook).

Objective: intra_coherence - cross_game_bleeding
  intra_coherence  = mean cosine_sim(enc(obs), enc(obs + noise)) over N samples
                     (higher = same-obs augmentations cluster together)
  cross_bleeding   = mean cosine_sim between different games' z-centroids
                     (lower = games occupy distinct z-regions)

Usage:
    python examples/contrastive_hpo.py --games cartpole mountaincar --trials 20
    python examples/contrastive_hpo.py --quick  # 5 trials, fast
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.encoder.universal_encoder import ContrastiveProjection


# ═══════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════════

def intra_game_coherence(
    proj: ContrastiveProjection,
    obs: np.ndarray,
    noise_sigma: float = 0.05,
    n_pairs: int = 50,
    rng: np.random.RandomState = None,
) -> float:
    """
    Mean cosine similarity between same-obs pairs under Gaussian noise.
    Higher = more coherent (same-state augmentations cluster together).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    N = min(n_pairs, len(obs))
    sims = []
    for i in range(N):
        x = obs[i]
        x_noisy = x + rng.randn(*x.shape).astype(np.float32) * noise_sigma
        z  = proj(x)
        zn = proj(x_noisy)
        nz  = np.linalg.norm(z)  + 1e-8
        nzn = np.linalg.norm(zn) + 1e-8
        sims.append(float(np.dot(z / nz, zn / nzn)))
    return float(np.mean(sims))


def cross_game_bleeding(
    projs_obs: Dict[str, Tuple[ContrastiveProjection, np.ndarray]],
) -> float:
    """
    Mean cosine similarity between different games' z-centroids.
    Lower = better separation (games occupy distinct z-regions).
    """
    centroids = {}
    for game, (proj, obs) in projs_obs.items():
        n = min(50, len(obs))
        zs = np.stack([proj(obs[i]) for i in range(n)], axis=0)
        c = zs.mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[game] = c / (norm + 1e-8)

    games = list(centroids)
    if len(games) < 2:
        return 0.0

    sims = []
    for i in range(len(games)):
        for j in range(i + 1, len(games)):
            sims.append(float(np.dot(centroids[games[i]], centroids[games[j]])))
    return float(np.mean(sims))


# ═══════════════════════════════════════════════════════════════
# PARAMETER SPACE
# ═══════════════════════════════════════════════════════════════

PARAM_BOUNDS = {
    "temperature": (0.05, 0.5),
    "lr":          (1e-4, 1e-2),
    "aug_noise":   (0.01, 0.2),
    "aug_dropout": (0.0,  0.4),
    "aug_scale":   (0.0,  0.3),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())
LO = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES], dtype=np.float64)
HI = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES], dtype=np.float64)
D  = len(PARAM_NAMES)


def _to_unit(x: np.ndarray) -> np.ndarray:
    """Normalise param vector to [0,1]^D."""
    return (x - LO) / (HI - LO + 1e-12)


def _from_unit(u: np.ndarray) -> dict:
    """Map [0,1]^D back to named param dict."""
    x = LO + u * (HI - LO)
    return {k: float(x[i]) for i, k in enumerate(PARAM_NAMES)}


def _random_unit(rng, n=1):
    return rng.rand(n, D)


# ═══════════════════════════════════════════════════════════════
# SELF-CONTAINED GP + EI
# ═══════════════════════════════════════════════════════════════

def _ei(mu, sigma, best, xi=0.01):
    """Expected Improvement over best."""
    from scipy.stats import norm as scipy_norm
    improvement = mu - best - xi
    z = improvement / (sigma + 1e-8)
    ei = improvement * scipy_norm.cdf(z) + sigma * scipy_norm.pdf(z)
    ei[sigma < 1e-8] = 0.0
    return ei


def _gp_predict(X_train, y_train, X_test, length_scale=0.5, noise=1e-4):
    """
    Minimal RBF kernel GP (no sklearn dependency).
    Returns mu, sigma for X_test.
    """
    def rbf(A, B, ls):
        diffs = A[:, None, :] - B[None, :, :]       # (na, nb, d)
        return np.exp(-0.5 * np.sum(diffs ** 2, axis=-1) / ls ** 2)

    K    = rbf(X_train, X_train, length_scale) + noise * np.eye(len(X_train))
    K_s  = rbf(X_train, X_test,  length_scale)
    K_ss = np.diag(rbf(X_test,  X_test,  length_scale))

    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu = K_s.T @ alpha
        v  = np.linalg.solve(L, K_s)
        sigma = np.sqrt(np.maximum(0.0, K_ss - np.sum(v ** 2, axis=0)))
    except np.linalg.LinAlgError:
        mu    = np.full(len(X_test), float(np.mean(y_train)))
        sigma = np.ones(len(X_test)) * 0.1

    return mu, sigma


# ═══════════════════════════════════════════════════════════════
# CONTRASTIVE HPO
# ═══════════════════════════════════════════════════════════════

class ContrastiveHPO:
    """
    Bayesian HPO (RBF-GP + EI) for ContrastiveProjection hyperparameters.
    Each game gets its own projection with the correct in_dim.
    """

    def __init__(
        self,
        obs_by_game: Dict[str, np.ndarray],
        n_epochs_per_trial: int = 10,
        n_initial_random: int = 3,
        z_dim: int = 8,
        verbose: bool = False,
    ):
        self.obs_by_game = obs_by_game
        self.n_epochs = n_epochs_per_trial
        self.n_initial = n_initial_random
        self.z_dim = z_dim
        self.verbose = verbose

        self.best_config: dict = {}
        self.best_score: float = -float("inf")
        self.trial_history: list = []

        # Unit-space trial data for GP
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

    def _objective(self, config: dict) -> float:
        rng_eval = np.random.RandomState(42)
        projs_obs: Dict[str, Tuple[ContrastiveProjection, np.ndarray]] = {}

        for game, obs in self.obs_by_game.items():
            in_dim = obs.shape[1]  # per-game in_dim
            cp = ContrastiveProjection(in_dim, self.z_dim, seed=0)
            try:
                cp.fit(
                    obs,
                    n_epochs=self.n_epochs,
                    lr=float(config["lr"]),
                    temperature=float(config["temperature"]),
                    aug_noise=float(config["aug_noise"]),
                    aug_dropout=float(config["aug_dropout"]),
                    aug_scale=float(config["aug_scale"]),
                    batch_size=min(32, len(obs)),
                    verbose=False,
                )
            except Exception as e:
                if self.verbose:
                    print(f"    fit error for {game}: {e}")
                return -1.0
            projs_obs[game] = (cp, obs)

        intra_scores = [
            intra_game_coherence(proj, obs, rng=rng_eval)
            for proj, obs in projs_obs.values()
        ]
        intra = float(np.mean(intra_scores))
        bleed = cross_game_bleeding(projs_obs)
        score = intra - bleed

        self.trial_history.append({"config": config, "score": score,
                                   "intra": intra, "bleed": bleed})
        if score > self.best_score:
            self.best_score = score
            self.best_config = dict(config)

        if self.verbose:
            print(f"    intra={intra:.4f}  bleed={bleed:.4f}  score={score:.4f}")

        return score

    def run(self, n_trials: int = 20) -> dict:
        t0 = time.time()
        rng = np.random.RandomState(0)

        # Phase 1: random warm-up
        n_random = min(self.n_initial, n_trials)
        if self.verbose:
            print(f"  Phase 1: {n_random} random trials")
        for i in range(n_random):
            u = _random_unit(rng)[0]
            config = _from_unit(u)
            score = self._objective(config)
            self._X.append(u)
            self._y.append(score)
            if self.verbose:
                print(f"    trial {i+1}: score={score:.4f}")

        # Phase 2: GP + EI
        gp_trials = n_trials - n_random
        if self.verbose and gp_trials > 0:
            print(f"  Phase 2: {gp_trials} GP-guided trials")

        for i in range(gp_trials):
            X_train = np.array(self._X)
            y_train = np.array(self._y)
            best_so_far = float(np.max(y_train))

            # Candidate pool in unit space
            candidates = _random_unit(rng, n=500)
            mu, sigma = _gp_predict(X_train, y_train, candidates)

            try:
                ei = _ei(mu, sigma, best_so_far)
                u_next = candidates[np.argmax(ei)]
            except Exception:
                u_next = _random_unit(rng)[0]

            config = _from_unit(u_next)
            score = self._objective(config)
            self._X.append(u_next)
            self._y.append(score)
            if self.verbose:
                print(f"    trial {n_random+i+1}: score={score:.4f}  best={max(self._y):.4f}")

        return {
            "best_config": self.best_config,
            "best_score":  round(float(self.best_score), 6),
            "n_trials":    len(self.trial_history),
            "elapsed":     round(time.time() - t0, 1),
            "history":     self.trial_history,
        }


# ═══════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════

def collect_obs(game_name: str, n: int = 200) -> np.ndarray:
    env_map = {
        "cartpole":    "CartPole-v1",
        "mountaincar": "MountainCar-v0",
        "lunarlander": "LunarLander-v2",
    }
    env_id = env_map.get(game_name)
    if env_id is None:
        return np.random.randn(n, 8).astype(np.float32)
    try:
        import gymnasium as gym
        env = gym.make(env_id)
        obs_list = []
        obs, _ = env.reset()
        obs_list.append(np.asarray(obs, dtype=np.float32))
        for _ in range(n - 1):
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        return np.stack(obs_list, axis=0)
    except Exception:
        obs_dim = {"cartpole": 4, "mountaincar": 2, "lunarlander": 8}.get(game_name, 4)
        return np.random.randn(n, obs_dim).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Contrastive HPO for Throng2")
    parser.add_argument("--games",   nargs="+", default=["cartpole", "mountaincar"])
    parser.add_argument("--trials",  type=int,  default=20)
    parser.add_argument("--epochs",  type=int,  default=10)
    parser.add_argument("--n-obs",   type=int,  default=200)
    parser.add_argument("--quick",   action="store_true")
    parser.add_argument("--save",    type=str,  default="results/contrastive_hpo.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.trials = 5
        args.epochs = 5
        args.n_obs  = 100

    print(f"Collecting observations for {args.games}...")
    obs_by_game = {g: collect_obs(g, args.n_obs) for g in args.games}
    for g, obs in obs_by_game.items():
        print(f"  {g}: {obs.shape}")

    z_dim = max(4, min(obs.shape[1] for obs in obs_by_game.values()))
    print(f"\nRunning HPO ({args.trials} trials, {args.epochs} epochs/trial, z_dim={z_dim})...")

    hpo = ContrastiveHPO(
        obs_by_game=obs_by_game,
        n_epochs_per_trial=args.epochs,
        z_dim=z_dim,
        verbose=args.verbose,
    )
    results = hpo.run(n_trials=args.trials)

    print("\n=== Best Configuration ===")
    for k, v in results["best_config"].items():
        print(f"  {k}: {v:.6f}")
    print(f"  score (intra - bleed): {results['best_score']:.4f}")
    print(f"  n_trials: {results['n_trials']}, elapsed: {results['elapsed']}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()
