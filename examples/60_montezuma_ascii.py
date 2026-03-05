"""
60_montezuma_ascii.py — ASCII Encoder proof-of-concept on Montezuma's Revenge

Tests whether ASCII-compressed frames make Montezuma tractable for simple learners.

Three agents compared:
  random      — Pure random baseline (expected: rarely scores)
  q_ascii     — Q-learner on ASCII region features (compact state)
  ppo_ascii   — PPO on full ASCII flat vector (300-dim)

Note: ROMs are bundled with ale-py. If env not found, run:
  python -c "import gymnasium, ale_py; gymnasium.register_envs(ale_py)"

Run:
  python examples/60_montezuma_ascii.py
  python examples/60_montezuma_ascii.py --episodes 500 --render-every 100
"""

import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoder.ascii_encoder import AsciiEncoder
from src.encoder.grid_state import GridState


# ---------------------------------------------------------------------------
# Q-Learner (tabular, region features bucketed to discrete state)
# ---------------------------------------------------------------------------

class AsciiQLearner:
    """
    Q-learner using ASCII region features as state.
    State: tuple of bucketed region mean densities (compact, generalizes)
    """
    def __init__(self, n_actions: int, n_regions: int = 12,
                 lr: float = 0.1, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: float = 0.995):
        self.n_actions = n_actions
        self.n_regions = n_regions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.q_table = {}  # state_key → np.array (n_actions,)
        self._step = 0

    def _key(self, features: np.ndarray) -> tuple:
        # Bucket each feature into 5 levels for tractable state space
        return tuple((features * 5).astype(int).clip(0, 4).tolist())

    def _get_q(self, key: tuple) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]

    def act(self, features: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        key = self._key(features)
        return int(np.argmax(self._get_q(key)))

    def learn(self, feat: np.ndarray, action: int, reward: float,
              next_feat: np.ndarray, done: bool):
        key = self._key(feat)
        next_key = self._key(next_feat)
        q = self._get_q(key)
        q_next = self._get_q(next_key)
        target = reward + (0.0 if done else self.gamma * np.max(q_next))
        q[action] += self.lr * (target - q[action])
        self._step += 1
        if done:
            self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


# ---------------------------------------------------------------------------
# Minimal PPO on flat ASCII (reuse cell's PPOHead if available)
# ---------------------------------------------------------------------------

class MinimalPPO:
    """Lightweight PPO actor-critic on flat ASCII features."""
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 3e-4,
                 rollout_len: int = 512):
        try:
            from src.cell.ppo_head import PPOHead
            self._ppo = PPOHead(obs_dim, n_actions, hidden=64, lr=lr,
                                rollout_length=rollout_len)
            self._use_cell = True
        except Exception as e:
            print(f"  [PPO] Falling back to random (PPOHead unavailable: {e})")
            self.n_actions = n_actions
            self._use_cell = False

    def act(self, obs: np.ndarray) -> int:
        if not self._use_cell:
            return np.random.randint(self.n_actions)
        return self._ppo.select_action(obs)

    def learn(self, reward: float, done: bool):
        if self._use_cell:
            self._ppo.store_reward(reward, done)
            if done:
                self._ppo.update()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_montezuma(render: bool = False):
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)  # ensure ALE envs are registered
    mode = "human" if render else "rgb_array"
    try:
        env = gym.make(
            "ALE/MontezumaRevenge-v5",
            render_mode=mode,
            frameskip=4,
            repeat_action_probability=0.0,
            full_action_space=False,  # only 18 game-relevant actions
        )
    except Exception:
        env = gym.make("MontezumaRevenge-v4", render_mode=mode)
    return env


def run_agent(agent_name: str, n_episodes: int, enc: AsciiEncoder,
              gs: GridState, render_every: int = 0) -> dict:

    try:
        env = make_montezuma(render=False)
    except Exception as e:
        print(f"  [ERROR] Could not create Montezuma env: {e}")
        print("  Install with: pip install gymnasium[atari] ale-py")
        print("  ROMs: AutoROM --accept-license")
        return {"label": agent_name, "error": str(e)}

    n_actions = env.action_space.n
    print(f"  Actions: {n_actions}, obs: {env.observation_space.shape}")

    # Build agent
    if agent_name == "random":
        agent = None
    elif agent_name == "q_ascii":
        agent = AsciiQLearner(n_actions=n_actions, n_regions=gs.n_regions)
    elif agent_name == "ppo_ascii":
        flat_dim = enc.rows * enc.cols
        agent = MinimalPPO(obs_dim=flat_dim, n_actions=n_actions, rollout_len=512)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    rewards, found_reward, q_sizes = [], 0, []
    t_enc_total, t_start = 0.0, time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        enc.reset()
        gs.reset()
        total, done, step_count = 0.0, False, 0
        prev_feat = None

        while not done:
            # Encode frame → ASCII → features
            t0 = time.perf_counter()
            grid = enc.encode(obs)
            feat = gs.features(grid)
            t_enc_total += time.perf_counter() - t0

            # Act
            if agent is None:
                action = env.action_space.sample()
            elif agent_name == "q_ascii":
                action = agent.act(feat)
            else:
                action = agent.act(gs.flat(grid))

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            step_count += 1

            # Log any reward event with RAM coords + ASCII snapshot
            if reward > 0:
                frame_num = env.unwrapped.ale.getEpisodeFrameNumber()
                try:
                    ram = env.unwrapped.ale.getRAM()
                    # RAM bytes vary by ROM; common Montezuma player coords
                    px, py = int(ram[42]), int(ram[43])
                    lives = int(ram[58])
                    coord_str = f" player_ram=({px},{py}) lives={lives}"
                except Exception:
                    coord_str = ""
                print(f"    [REWARD] ep={ep+1} frame={frame_num} step={step_count}"
                      f" reward={reward:.0f} total={total:.0f}{coord_str}")
                # Print ASCII grid at reward moment
                reward_grid = enc.encode(obs)
                print(enc.to_text(reward_grid))
                print()

            # Learn
            if agent_name == "q_ascii" and prev_feat is not None:
                next_feat = gs.features(enc.encode(obs))
                agent.learn(prev_feat, action, reward, next_feat, done)
            elif agent_name == "ppo_ascii":
                agent.learn(reward, done)

            prev_feat = feat.copy()

        rewards.append(total)
        if total > 0:
            found_reward += 1

        if isinstance(agent, AsciiQLearner):
            q_sizes.append(len(agent.q_table))

        # Render ASCII snapshot every N episodes
        if render_every > 0 and (ep + 1) % render_every == 0:
            sample_grid = enc.encode(obs)
            print(f"\n  [ASCII snapshot] ep {ep+1}, reward={total:.0f}")
            print(enc.to_text(sample_grid))
            print()

        if (ep + 1) % 100 == 0 or ep == 0:
            avg = np.mean(rewards[-min(100, len(rewards)):])
            eps_val = getattr(agent, 'epsilon', None)
            eps_str = f"  eps={eps_val:.3f}" if eps_val is not None else ""
            q_str = f"  Q-states={q_sizes[-1]}" if q_sizes else ""
            print(f"  [{agent_name:10s}] ep {ep+1:5d}  "
                  f"avg100={avg:6.1f}  found_reward={found_reward}{eps_str}{q_str}")

    env.close()
    avg_enc_ms = t_enc_total / max(1, sum(1 for _ in range(n_episodes))) * 1000

    return {
        "label": agent_name,
        "avg_last100": round(float(np.mean(rewards[-100:])), 1),
        "max_reward":  round(float(max(rewards)), 1),
        "found_reward_eps": found_reward,
        "found_reward_pct": round(100 * found_reward / n_episodes, 1),
        "total_time_s": round(time.time() - t_start, 1),
        "q_table_size": q_sizes[-1] if q_sizes else "n/a",
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--rows",     type=int, default=20)
    parser.add_argument("--cols",     type=int, default=15)
    parser.add_argument("--render-every", type=int, default=0,
                        help="Print ASCII snapshot every N episodes (0=off)")
    parser.add_argument("--agents", nargs="+",
                        default=["random", "q_ascii", "ppo_ascii"])
    args = parser.parse_args()

    enc = AsciiEncoder(rows=args.rows, cols=args.cols)
    gs  = GridState(rows=args.rows, cols=args.cols)

    print("=" * 65)
    print(f"ASCII ENCODER — Montezuma's Revenge ({args.episodes} eps)")
    print("=" * 65)
    print(f"  Grid: {args.rows}×{args.cols} = {args.rows*args.cols} values")
    print(f"  Region features: {gs.feature_dim}-dim Q-learner state")
    print(f"  Flat features:   {args.rows*args.cols}-dim PPO state")
    print()

    # Quick timing benchmark
    fake_frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for _ in range(1000):
        enc.encode(fake_frame)
    ms_per_frame = (time.perf_counter() - t0)
    print(f"  Encoding speed: {ms_per_frame:.3f}ms per frame (1000-iteration avg)")
    print(f"  ASCII preview of random frame:")
    print(enc.to_text(enc.encode(fake_frame)))
    print()

    results = {}
    for agent_name in args.agents:
        print(f"[{agent_name}]")
        r = run_agent(agent_name, args.episodes, enc, gs,
                      render_every=args.render_every)
        results[agent_name] = r
        print()

    # Summary
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    agents = args.agents
    fmt = f"{{:28s}}" + "".join(f"{{:>12s}}" for _ in agents)
    print(fmt.format("", *agents))
    print("-" * 65)
    for key, title in [
        ("avg_last100",      "Avg last 100 eps"),
        ("max_reward",       "Max episode reward"),
        ("found_reward_eps", "Episodes with reward"),
        ("found_reward_pct", "% episodes with reward"),
        ("total_time_s",     "Total time (s)"),
        ("q_table_size",     "Q-table states"),
    ]:
        vals = [str(results[a].get(key, "err")) for a in agents]
        print(fmt.format(title, *vals))

    print()
    print("Key question: Did any agent score > 0 on Montezuma?")
    for a in agents:
        r = results[a]
        if "error" in r:
            print(f"  {a}: ERROR — {r['error']}")
        elif r.get("found_reward_eps", 0) > 0:
            print(f"  {a}: YES — scored in {r['found_reward_eps']} episodes "
                  f"(max={r['max_reward']})")
        else:
            print(f"  {a}: No reward found in {args.episodes} episodes")


if __name__ == "__main__":
    main()
