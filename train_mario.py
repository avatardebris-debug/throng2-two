"""
train_mario.py — End-to-end Mario training with in-process vectorized sims.

Architecture: All N sims run in the main process (no multiprocessing).
Why? The ASCII sim does 27,000 sps single-threaded — IPC overhead from
multiprocessing costs MORE than the step itself. Each subprocess also
eats ~200MB RAM (Python interpreter), so on 8GB machines, multiprocessing
hurts more than it helps.

  In-process: N sims × 37μs/step = 0.37ms → batched policy → 0.03ms → done
  Multiprocess: N workers × (37μs step + 100μs IPC) + RAM pressure → slower

Usage:
    python train_mario.py                     # auto-detect envs
    python train_mario.py --n_envs 10         # 10 in-process sims
    python train_mario.py --steps 1000000     # total steps
    python train_mario.py --name run1         # checkpoint prefix
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.games.mario.level_sampler import LevelSampler
from src.games.mario.sim_noise import NoisyMarioSim, NoiseScheduler
from src.games.mario.mario_agent import MarioRLAgent


# ══════════════════════════════════════════════════════════════════════
#  In-process vectorized environment
# ══════════════════════════════════════════════════════════════════════

class InProcessVecEnv:
    """
    Runs N sims in a single thread. Zero IPC, zero subprocess RAM overhead.

    Optimal for fast envs (<100μs/step) like the ASCII Mario simulator.
    Uses ~1MB per sim instead of ~200MB per subprocess.

    API matches ThrongVecEnv for easy swapping:
        obs = vec.reset()                   # (N, obs_dim)
        obs, rew, done = vec.step(actions)  # actions: (N,)
    """

    def __init__(self, env_fn, n_envs: int, obs_dim: int, n_actions: int = 8):
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.sims = [env_fn() for _ in range(n_envs)]

        # Pre-allocate output buffers (avoid re-allocation every step)
        self._obs_buf  = np.zeros((n_envs, obs_dim), dtype=np.float32)
        self._rew_buf  = np.zeros(n_envs, dtype=np.float32)
        self._done_buf = np.zeros(n_envs, dtype=bool)

    @property
    def active_count(self) -> int:
        return self.n_envs

    def reset(self) -> np.ndarray:
        for i in range(self.n_envs):
            self._obs_buf[i] = self.sims[i].reset()
        return self._obs_buf.copy()

    def step(self, actions: np.ndarray):
        """Step all sims. Auto-resets done envs. Returns (obs, rew, done)."""
        for i in range(self.n_envs):
            obs, rew, done, _ = self.sims[i].step(int(actions[i]))
            self._rew_buf[i]  = rew
            self._done_buf[i] = done
            if done:
                self._obs_buf[i] = self.sims[i].reset()
            else:
                self._obs_buf[i] = obs

        return self._obs_buf.copy(), self._rew_buf.copy(), self._done_buf.copy()

    def status(self) -> dict:
        try:
            import psutil
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=None)
        except ImportError:
            ram, cpu = 0, 0
        return {"active": self.n_envs, "paused": 0, "ram_pct": ram, "cpu_pct": cpu}

    def close(self):
        for sim in self.sims:
            if hasattr(sim, 'close'):
                sim.close()
        self.sims.clear()


# ══════════════════════════════════════════════════════════════════════
#  Vectorized policy (single shared agent, batched forward)
# ══════════════════════════════════════════════════════════════════════

class VecPolicy:
    """
    Single MarioRLAgent with fully vectorized forward + buffer ops.
    All inner loops replaced with numpy operations.
    """

    def __init__(self, obs_dim: int, n_actions: int, **kwargs):
        self.agent = MarioRLAgent(obs_dim, n_actions, **kwargs)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self._episode_rewards = []
        self._current_ep_rewards = None
        self._obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self._obs_var = np.ones(obs_dim, dtype=np.float32)
        self._obs_count = 0

    def _normalize_obs_batch(self, obs: np.ndarray) -> np.ndarray:
        """Batch-update running mean/var then normalize. Vectorized."""
        n = obs.shape[0]
        # Update running stats with batch mean (approximate Welford for speed)
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        self._obs_count += n
        alpha = n / self._obs_count
        self._obs_mean = (1 - alpha) * self._obs_mean + alpha * batch_mean
        self._obs_var = (1 - alpha) * self._obs_var + alpha * batch_var
        std = np.sqrt(np.maximum(self._obs_var, 1e-8))
        return ((obs - self._obs_mean) / std).astype(np.float32)

    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """Fully vectorized action selection."""
        n = obs.shape[0]
        if self._current_ep_rewards is None or len(self._current_ep_rewards) != n:
            self._current_ep_rewards = np.zeros(n, dtype=np.float64)

        norm_obs = self._normalize_obs_batch(obs)

        # Batched forward — one BLAS matmul
        probs, values, _ = self.agent.net.forward_batch(norm_obs)

        # Vectorized action sampling via cumulative probability
        cum_probs = probs.cumsum(axis=-1)
        u = np.random.random((n, 1)).astype(np.float32)
        actions = (u < cum_probs).argmax(axis=-1).astype(np.int64)

        log_probs = np.log(probs[np.arange(n), actions] + 1e-10)
        self.agent._total_steps += n

        self._last_norm_obs = norm_obs
        self._last_actions = actions
        self._last_log_probs = log_probs
        self._last_values = values
        return actions

    def process_results(self, rewards: np.ndarray, dones: np.ndarray) -> dict:
        """Vectorized buffer filling + episode tracking."""
        n = len(rewards)
        completed = []
        update_stats = None

        # Track episode rewards
        self._current_ep_rewards += rewards

        # Bulk-extend the rollout buffer (list extend vs N appends)
        buf = self.agent
        buf._buf_obs.extend(self._last_norm_obs)
        buf._buf_actions.extend(self._last_actions.tolist())
        buf._buf_log_probs.extend(self._last_log_probs.tolist())
        buf._buf_values.extend(self._last_values.tolist())
        buf._buf_rewards.extend(rewards.tolist())
        buf._buf_dones.extend(dones.tolist())

        buf._episode_reward += rewards.sum()

        # Handle done episodes
        done_idxs = np.where(dones)[0]
        for i in done_idxs:
            buf._total_episodes += 1
            ep_rew = float(self._current_ep_rewards[i])
            buf._reward_history.append(ep_rew)
            completed.append(ep_rew)
            self._episode_rewards.append(ep_rew)
            self._current_ep_rewards[i] = 0.0

        if done_idxs.size > 0:
            buf._episode_reward = 0.0

        # Batched PPO update when buffer full
        if len(buf._buf_obs) >= buf.rollout_length:
            last_val = 0.0
            if self._last_norm_obs is not None and len(self._last_norm_obs) > 0:
                _, last_val, _ = buf.net.forward(self._last_norm_obs[0])
            update_stats = buf._update(last_val)
            buf._clear_buffer()

        return {
            'completed': completed,
            'n_updates': update_stats.get('n_updates', 0) if update_stats else 0,
        }

    @property
    def episode_rewards(self):
        return self._episode_rewards

    @property
    def recent_win_rate(self) -> float:
        if len(self._episode_rewards) < 10:
            return 0.0
        recent = self._episode_rewards[-100:]
        return sum(1 for r in recent if r > 5.0) / len(recent)


# ══════════════════════════════════════════════════════════════════════
#  Checkpoint
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(policy: VecPolicy, sampler: LevelSampler,
                    scheduler: NoiseScheduler, step: int, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    agent = policy.agent
    data = {
        'step': step,
        'episode_rewards': np.array(policy.episode_rewards[-1000:]),
        'sampler_step': sampler._step,
        'noise_scale': scheduler.noise_scale,
        'scheduler_results': np.array(scheduler._results[-1000:]),
        'W1': agent.net.W1, 'b1': agent.net.b1,
        'W2': agent.net.W2, 'b2': agent.net.b2,
        'Wp': agent.net.Wp, 'bp': agent.net.bp,
        'Wv': agent.net.Wv, 'bv': agent.net.bv,
    }
    np.savez_compressed(path, **data)
    print(f'  [Checkpoint] Saved to {path}')


# ══════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════

def train(args):
    print("=" * 60)
    print("  Mario Training Pipeline (in-process vectorized)")
    print("=" * 60)

    obs_dim = 378
    n_actions = 8

    # ── 1. Configure env count ────────────────────────────────
    if args.n_envs > 0:
        n_envs = args.n_envs
    else:
        # In-process sims use ~1MB each → limited only by CPU
        n_cores = os.cpu_count() or 4
        n_envs = max(4, n_cores)  # Use all cores worth of sims
        # More sims = more transitions per step = faster learning
        # (batched forward pass cost barely changes up to N=64)
    print(f"\n[Config] {n_envs} in-process sims (0 subprocesses, ~{n_envs}MB total)")

    # ── 2. Build components ───────────────────────────────────
    print("[Init] Building components...")

    sampler = LevelSampler(
        canonical_start=0.75,
        canonical_target=0.50,
        procedural_ratio=0.20,
        decay_half_life=50_000,
        seed=args.seed,
    )

    scheduler = NoiseScheduler(
        ramp_start=0.30,
        ramp_full=0.80,
        window=100,
    )

    noise_scale = [0.0]

    def make_env():
        return NoisyMarioSim(
            sampler.sample_factory(),
            seed=None,
            initial_noise_scale=noise_scale[0],
        )

    # ── 3. Create in-process vec env ──────────────────────────
    vec = InProcessVecEnv(make_env, n_envs=n_envs, obs_dim=obs_dim, n_actions=n_actions)

    # ── 4. Create policy ──────────────────────────────────────
    policy = VecPolicy(obs_dim, n_actions,
                       hidden1=128, hidden2=64, lr=3e-4,
                       rollout_length=2048)  # Standard PPO (SB3 default)

    ratios = sampler.ratios()
    print(f"  Sampler: canonical={ratios['canonical']:.0%} gan={ratios['gan']:.0%} proc={ratios['procedural']:.0%}")
    print(f"  Noise: phase=explore (scale=0.0)")
    print(f"  Rollout buffer: {policy.agent.rollout_length} transitions")

    # ── 5. Training loop ──────────────────────────────────────
    print(f"\n[Training] {args.steps:,} steps, log every {args.log_interval:,}")
    print()

    obs = vec.reset()
    total_steps = 0
    t0 = time.time()
    last_log_step = 0
    last_ckpt_step = 0
    best_avg_reward = -float('inf')

    try:
        while total_steps < args.steps:
            actions = policy.select_actions(obs)
            obs, rewards, dones = vec.step(actions)
            total_steps += n_envs

            result = policy.process_results(rewards, dones)

            # Update noise scheduler
            for ep_reward in result['completed']:
                scheduler.report(ep_reward > 5.0)

            sampler.advance(n_envs)

            # ── Logging ───────────────────────────────────
            if total_steps - last_log_step >= args.log_interval:
                last_log_step = total_steps
                elapsed = time.time() - t0
                sps = total_steps / elapsed

                noise_scale[0] = scheduler.noise_scale

                n_eps = len(policy.episode_rewards)
                avg_r = float(np.mean(policy.episode_rewards[-100:])) if n_eps > 0 else 0
                best_r = max(policy.episode_rewards[-100:]) if n_eps > 0 else 0

                sched = scheduler.stats()
                ratios = sampler.ratios()
                status = vec.status()

                print(f"  step={total_steps:>9,d}  "
                      f"sps={sps:>6,.0f}  "
                      f"eps={n_eps:>5d}  "
                      f"avg_r={avg_r:>7.2f}  "
                      f"best_r={best_r:>7.2f}  "
                      f"win={policy.recent_win_rate:>4.0%}  "
                      f"noise={sched['noise_scale']:.2f}({sched['phase'][:3]})  "
                      f"canon={ratios['canonical']:.0%}  "
                      f"RAM={status['ram_pct']:.0f}%")

                if avg_r > best_avg_reward and n_eps >= 50:
                    best_avg_reward = avg_r
                    save_checkpoint(policy, sampler, scheduler, total_steps,
                                    f'saves/{args.name}_best.npz')

            # ── Checkpointing ─────────────────────────────
            if total_steps - last_ckpt_step >= args.ckpt_interval:
                last_ckpt_step = total_steps
                save_checkpoint(policy, sampler, scheduler, total_steps,
                                f'saves/{args.name}_latest.npz')

    except KeyboardInterrupt:
        print("\n\n  [Interrupted] Saving checkpoint...")
        save_checkpoint(policy, sampler, scheduler, total_steps,
                        f'saves/{args.name}_interrupted.npz')

    finally:
        vec.close()

    # ── Final summary ─────────────────────────────────────────
    elapsed = time.time() - t0
    n_eps = len(policy.episode_rewards)
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Total steps:    {total_steps:,}")
    print(f"  Total episodes: {n_eps:,}")
    print(f"  Wall time:      {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Throughput:     {total_steps/elapsed:,.0f} steps/sec")
    if n_eps > 0:
        print(f"  Avg reward:     {np.mean(policy.episode_rewards[-100:]):.2f}")
        print(f"  Win rate:       {policy.recent_win_rate:.0%}")
        print(f"  Best avg:       {best_avg_reward:.2f}")
    print(f"  Noise phase:    {scheduler.stats()['phase']}")
    print(f"  Canonical ratio: {sampler.ratios()['canonical']:.0%}")

    save_checkpoint(policy, sampler, scheduler, total_steps,
                    f'saves/{args.name}_final.npz')


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mario Training Pipeline")
    parser.add_argument('--steps', type=int, default=500_000,
                        help='Total training steps (default: 500k)')
    parser.add_argument('--n_envs', type=int, default=0,
                        help='Number of in-process sims (0=auto, uses cpu_count)')
    parser.add_argument('--name', type=str, default='mario',
                        help='Checkpoint name prefix')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_interval', type=int, default=5000,
                        help='Steps between log lines')
    parser.add_argument('--ckpt_interval', type=int, default=50000,
                        help='Steps between checkpoints')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
