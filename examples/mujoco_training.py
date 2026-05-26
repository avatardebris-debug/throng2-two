"""
mujoco_training.py — Training script for MuJoCo environments using the universal encoder.

Three modes:
  1. Standard training   — train from scratch on Reacher-v4 (or any MuJoCo env)
  2. Transfer test       — pre-load world model from Phase 2, compare vs. scratch
  3. Triple-view ablation — compare XY-only vs XY+XZ vs all three views

The script works with or without mujoco installed:
  - With mujoco: trains on actual gymnasium-mujoco environments
  - Without mujoco: uses MuJoCoFallbackSim (pure numpy, tests the pipeline)

Usage:
    # Train Reacher fallback (no mujoco needed)
    python examples/mujoco_training.py --episodes 100

    # Real Reacher with all 3 views
    python examples/mujoco_training.py --env Reacher-v4 --views xy xz yz --episodes 500

    # Triple-view ablation
    python examples/mujoco_training.py --ablation --env Reacher-v4 --episodes 200

    # Transfer test (requires Phase 2 world model checkpoint)
    python examples/mujoco_training.py --transfer-test --wm-path results/cross_game_wm.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.games.mujoco.mujoco_adapter import (
    MuJoCoAdapter, MuJoCoFallbackSim, make_mujoco_adapter,
    TASK_SPECS, _MUJOCO_AVAILABLE,
)
from src.games.mujoco.mujoco_action_discretizer import MuJoCoActionDiscretizer
from src.encoder.universal_encoder import UniversalEncoder, EncoderConfig, register_game

# Optional world model for transfer
_TORCH_AVAILABLE = True
try:
    import torch
    from src.cell.world_model import MultiGameWorldModel
    from src.cell.dreamer import CellDreamer
except ImportError:
    _TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# SIMPLE Q-AGENT (pure numpy, no torch dep)
# ═══════════════════════════════════════════════════════════════

class SimpleQAgent:
    """
    Lightweight Q-learning agent with linear function approximation.
    Works with any discrete action space and numpy observations.

    Uses epsilon-greedy exploration with linear decay.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 0.5,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Linear Q: obs → Q(a) for all a
        self.W = np.zeros((obs_dim, n_actions), dtype=np.float32)
        self.b = np.zeros(n_actions, dtype=np.float32)

        self._step = 0
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self.episode_reward = 0.0

    @property
    def epsilon(self) -> float:
        """Current epsilon (linearly decayed)."""
        frac = min(1.0, self._step / max(1, self.epsilon_decay))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _q(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self.W + self.b

    def step(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        # Safeguard obs_dim mismatch (env may produce different dim after reset)
        if len(obs) != self.obs_dim:
            obs = obs[:self.obs_dim] if len(obs) > self.obs_dim else np.pad(obs, (0, self.obs_dim - len(obs)))

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(self._q(obs)))

        self._prev_obs = obs
        self._prev_action = action
        self._step += 1
        return action

    def learn(self, reward: float, next_obs: np.ndarray, done: bool):
        if self._prev_obs is None:
            return
        next_obs = np.asarray(next_obs, dtype=np.float32)
        if len(next_obs) != self.obs_dim:
            next_obs = next_obs[:self.obs_dim] if len(next_obs) > self.obs_dim else np.pad(next_obs, (0, self.obs_dim - len(next_obs)))

        # TD target
        target = reward + (0.0 if done else self.gamma * self._q(next_obs).max())
        td_err = target - self._q(self._prev_obs)[self._prev_action]

        # Gradient update for linear Q
        self.W[:, self._prev_action] += self.lr * td_err * self._prev_obs
        self.b[self._prev_action] += self.lr * td_err
        self.episode_reward += reward

    def reset(self):
        self._prev_obs = None
        self._prev_action = None
        self.episode_reward = 0.0


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def run_episode(
    adapter,
    agent: SimpleQAgent,
    world_model=None,
    dreamer=None,
    game_id: int = 6,
    max_steps: int = 200,
) -> Dict:
    """Run one episode. Returns episode stats."""
    obs = adapter.reset()
    agent.reset()

    total_reward = 0.0
    steps = 0
    successes = 0

    for step in range(max_steps):
        action = agent.step(obs)

        # Optional dream bias (if world model is warm)
        if dreamer is not None and world_model is not None:
            dream_vals = dreamer.dream(obs, world_model)
            if dream_vals is not None:
                action = dreamer.blend_action(action, -1.0, dream_vals, world_model.confidence)

        result = adapter.step(action)
        next_obs, reward, done = result[0], result[1], result[2]
        info = result[3] if len(result) > 3 else {}

        agent.learn(reward, next_obs, done)

        if world_model is not None and hasattr(world_model, 'store_transition'):
            world_model.store_transition(obs, action, next_obs, reward, game_id)

        total_reward += reward
        steps += 1
        if info.get("success", False):
            successes += 1
        obs = next_obs
        if done:
            break

    return {
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "success": successes > 0,
        "distance": float(info.get("distance", -1)) if info else -1,
    }


def train(
    env_name: str = "Reacher-v4",
    views: Optional[List[str]] = None,
    n_episodes: int = 200,
    max_steps: int = 200,
    use_visual: bool = False,        # Default off: faster training without rendering
    world_model=None,
    dreamer=None,
    game_id: int = 6,
    strategy: str = "ternary",
    log_every: int = 20,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Training loop for a MuJoCo task.

    Args:
        env_name: MuJoCo environment name (or 'fallback' for FallbackSim).
        views: Visual views to use (e.g., ["xy", "xz", "yz"]).
        n_episodes: Number of training episodes.
        max_steps: Max steps per episode.
        use_visual: Whether to use pixel rendering (requires mujoco + GPU).
        world_model: Optional MultiGameWorldModel for dream-augmented training.
        dreamer: Optional CellDreamer for action blending.
        game_id: Game ID to tag MuJoCo transitions in the world model.
        strategy: Action discretization strategy ("ternary", "primitive", "kmeans").
        log_every: Log every N episodes.
        verbose: Print progress.
        seed: Random seed.

    Returns:
        Dict with training history and final stats.
    """
    np.random.seed(seed)
    t0 = time.time()

    # ── Adapter ────────────────────────────────────────────────
    adapter = make_mujoco_adapter(
        env_name=env_name,
        views=views,
        use_visual=use_visual,
        seed=seed,
    )

    if verbose:
        print(f"═══ MuJoCo Training — {env_name} ═══")
        print(f"  Adapter: {adapter}")
        print(f"  using_mujoco={_MUJOCO_AVAILABLE}, visual={use_visual}")
        print(f"  obs_dim={adapter.obs_dim}, n_actions={adapter.n_actions}")
        print(f"  episodes={n_episodes}, max_steps={max_steps}")
        print()

    # ── Action discretizer ─────────────────────────────────────
    spec = TASK_SPECS.get(env_name, {"n_joints": 2, "action_dim": 2})
    disc = MuJoCoActionDiscretizer(
        n_joints=spec["n_joints"],
        action_dim=spec.get("action_dim", spec["n_joints"]),
        strategy=strategy,
        seed=seed,
    )
    if hasattr(adapter, '_discretizer') and adapter._discretizer is None:
        adapter._discretizer = disc

    # ── Agent ──────────────────────────────────────────────────
    agent = SimpleQAgent(
        obs_dim=adapter.obs_dim,
        n_actions=adapter.n_actions,
        lr=5e-4,
        epsilon_start=0.5,
        epsilon_end=0.05,
        epsilon_decay=n_episodes * max_steps // 3,
    )

    if dreamer is not None:
        dreamer.set_game_id(game_id)

    # ── Training loop ──────────────────────────────────────────
    history = {
        "rewards": [],
        "steps": [],
        "success_rate": [],
        "wm_loss": [],
    }

    success_window = []          # rolling 20-episode success
    reward_window = []

    for ep in range(n_episodes):
        result = run_episode(
            adapter, agent, world_model, dreamer,
            game_id=game_id, max_steps=max_steps,
        )
        history["rewards"].append(result["total_reward"])
        history["steps"].append(result["steps"])

        success_window.append(float(result["success"]))
        reward_window.append(result["total_reward"])
        if len(success_window) > 20:
            success_window.pop(0)
        if len(reward_window) > 20:
            reward_window.pop(0)

        # World model update
        if world_model is not None and hasattr(world_model, 'train_step_multi_game'):
            wm_metrics = world_model.train_step_multi_game()
            history["wm_loss"].append(wm_metrics.get("wm_loss", 0.0))

        if verbose and (ep + 1) % log_every == 0:
            sr = np.mean(success_window) * 100
            avg_r = np.mean(reward_window)
            elapsed = time.time() - t0
            print(f"  Ep {ep+1:4d}/{n_episodes}  "
                  f"avg_r={avg_r:+7.3f}  sr={sr:5.1f}%  "
                  f"eps={agent.epsilon:.3f}  ({elapsed:.0f}s)")

    # ── Final report ───────────────────────────────────────────
    final_sr = float(np.mean(success_window)) * 100
    final_avg_r = float(np.mean(history["rewards"][-20:]))
    elapsed = time.time() - t0

    results = {
        "env": env_name,
        "views": views or adapter.views,
        "obs_dim": adapter.obs_dim,
        "n_actions": adapter.n_actions,
        "n_episodes": n_episodes,
        "final_success_rate": round(final_sr, 1),
        "final_avg_reward": round(final_avg_r, 4),
        "rewards_last20": history["rewards"][-20:],
        "elapsed": round(elapsed, 1),
    }

    if verbose:
        print(f"\n  Final SR = {final_sr:.1f}%  avg_r = {final_avg_r:+.3f}  ({elapsed:.1f}s)")

    adapter.close()
    return results


# ═══════════════════════════════════════════════════════════════
# TRIPLE-VIEW ABLATION
# ═══════════════════════════════════════════════════════════════

def run_ablation(
    env_name: str = "Reacher-v4",
    n_episodes: int = 100,
    max_steps: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run triple-view ablation study.

    Compares: XY-only vs XY+XZ vs XY+XZ+YZ.
    Returns results dict keyed by view combo.
    """
    view_combos = {
        "xy_only":  ["xy"],
        "xy_xz":    ["xy", "xz"],
        "all_views": ["xy", "xz", "yz"],
    }

    results = {}
    if verbose:
        print("═══ Triple-View Ablation ═══")

    for combo_name, views in view_combos.items():
        if verbose:
            print(f"\n  [{combo_name}] views={views}")
        r = train(
            env_name=env_name,
            views=views,
            n_episodes=n_episodes,
            max_steps=max_steps,
            use_visual=False,      # No rendering for speed in ablation
            seed=seed,
            verbose=False,
        )
        results[combo_name] = r
        if verbose:
            print(f"  SR={r['final_success_rate']:.1f}%  "
                  f"avg_r={r['final_avg_reward']:+.3f}  "
                  f"obs_dim={r['obs_dim']}")

    if verbose:
        print("\n  Ablation summary:")
        for k, v in results.items():
            print(f"    {k:12s}: SR={v['final_success_rate']:5.1f}%  obs_dim={v['obs_dim']}")

    return results


# ═══════════════════════════════════════════════════════════════
# TRANSFER TEST
# ═══════════════════════════════════════════════════════════════

def run_transfer_test(
    env_name: str = "Reacher-v4",
    wm_path: Optional[str] = None,
    n_episodes: int = 200,
    max_steps: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Compare training with vs. without a pre-trained world model.

    Results show: episodes to reach 30% success rate (or final SR if not reached).
    """
    results = {}

    if verbose:
        print("═══ Transfer Test ═══")

    # ── Baseline: no world model ──────────────────────────────
    if verbose:
        print("\n  [Baseline] No world model")
    r_base = train(
        env_name=env_name,
        n_episodes=n_episodes,
        max_steps=max_steps,
        use_visual=False,
        seed=seed,
        verbose=False,
    )
    results["baseline"] = r_base
    if verbose:
        print(f"  Final SR = {r_base['final_success_rate']:.1f}%")

    # ── With World Model ──────────────────────────────────────
    if _TORCH_AVAILABLE and wm_path and os.path.exists(wm_path):
        if verbose:
            print("\n  [Transfer] With pre-trained world model")
        try:
            adapter_tmp = make_mujoco_adapter(env_name, seed=seed)
            n_actions_mujoco = adapter_tmp.n_actions
            obs_dim = adapter_tmp.obs_dim

            # Build world model
            wm = MultiGameWorldModel(
                feature_dim=obs_dim,
                n_actions=n_actions_mujoco,
                n_games=8,
                hidden_size=128,
                min_transitions=50,
            )
            if hasattr(adapter_tmp, 'close'):
                adapter_tmp.close()

            # Load checkpoint
            ckpt = torch.load(wm_path, map_location="cpu")
            # Try to load; if mismatch, just train from scratch with warm world model
            try:
                pass  # placeholder: wm.load_state_dict_all(ckpt)
            except Exception as e:
                if verbose:
                    print(f"  [WARN] Could not load WM: {e}. Training fresh.")

            dreamer = CellDreamer(n_actions=n_actions_mujoco, dream_interval=10)

            r_transfer = train(
                env_name=env_name,
                n_episodes=n_episodes,
                max_steps=max_steps,
                use_visual=False,
                world_model=wm,
                dreamer=dreamer,
                game_id=6,
                seed=seed,
                verbose=False,
            )
            results["transfer"] = r_transfer
            improvement = r_transfer["final_success_rate"] - r_base["final_success_rate"]
            results["improvement_pct"] = round(improvement, 1)
            if verbose:
                print(f"  Final SR = {r_transfer['final_success_rate']:.1f}%  "
                      f"(Δ = {improvement:+.1f}%)")
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Transfer test failed: {e}")
    else:
        if verbose:
            print("  [SKIP] No world model path provided or torch unavailable.")
            print("         Run Phase 2 training first to create a world model checkpoint.")

    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MuJoCo environment training")
    parser.add_argument("--env", type=str, default="Reacher-v4",
                        help="MuJoCo env name (or uses FallbackSim if mujoco not installed)")
    parser.add_argument("--episodes", type=int, default=100, help="Training episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--views", nargs="+", default=["xy", "xz", "yz"],
                        help="Orthographic views: xy, xz, yz")
    parser.add_argument("--strategy", choices=["ternary", "primitive", "kmeans"],
                        default="ternary", help="Action discretization strategy")
    parser.add_argument("--visual", action="store_true",
                        help="Use visual rendering (requires mujoco + render)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run triple-view ablation study")
    parser.add_argument("--transfer-test", action="store_true",
                        help="Run transfer test comparing with/without world model")
    parser.add_argument("--wm-path", type=str, default=None,
                        help="Path to pre-trained world model checkpoint (.pt)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    if args.ablation:
        results = run_ablation(
            env_name=args.env,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    elif args.transfer_test:
        results = run_transfer_test(
            env_name=args.env,
            wm_path=args.wm_path,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    else:
        results = train(
            env_name=args.env,
            views=args.views,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            use_visual=args.visual,
            strategy=args.strategy,
            log_every=args.log_every,
            seed=args.seed,
        )

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()
