"""
61_montezuma_ram_qlearner.py — RAM-based Q-learner for Montezuma's Revenge

State: (room, x//8, y//8, key_collected) — ~700 states/room, matches Lolo pattern
Reward: shaped (subgoals + novelty + death penalty) from throng5's calibrated engine

RAM bytes (verified):
  RAM[3]  = room     (1 = start room)
  RAM[42] = player_x (0-159)
  RAM[43] = player_y (0-255)
  RAM[56] = key byte (0xFF when key held)
  RAM[65] = items    (bit1 set when key held)
  RAM[58] = lives

Key location: x=15, y=201, room=1 (calibrated from human RAM log 2026-02-22)

Run:
  python examples/61_montezuma_ram_qlearner.py
  python examples/61_montezuma_ram_qlearner.py --episodes 500 --render-every 50
"""

import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.encoder.ascii_encoder import AsciiEncoder


# ---------------------------------------------------------------------------
# RAM state extractor (from throng5 ram_decoders.py, embedded here)
# ---------------------------------------------------------------------------

def read_ram_state(ram: np.ndarray) -> dict:
    """Extract calibrated Montezuma state from 128-byte RAM."""
    player_x  = int(ram[42])
    player_y  = int(ram[43])
    room      = int(ram[3])
    lives     = int(ram[58])
    key_raw   = int(ram[56])
    items     = int(ram[65])
    key_held  = (key_raw == 0xFF) and bool(items & 0x02)
    return {
        "x":    player_x,
        "y":    player_y,
        "room": room,
        "lives": lives,
        "key":  key_held,
    }


def q_state_key(s: dict, bin_size: int = 8) -> tuple:
    """
    (room, x_bin, y_bin, key) — same pattern as Lolo's (puzzle, row, col, hearts).
    bin_size=8 → ~20 x-bins × ~32 y-bins × 7 rooms × 2 key states ≈ ~9000 states.
    """
    return (
        s["room"],
        s["x"] // bin_size,
        s["y"] // bin_size,
        int(s["key"]),
    )


# ---------------------------------------------------------------------------
# Novelty tracker (from throng5 montezuma_subgoals.py)
# ---------------------------------------------------------------------------

class NoveltyTracker:
    """Count-based novelty bonus: reward = scale / sqrt(visit_count)."""
    def __init__(self, bin_size: int = 8, scale: float = 0.5):
        self._bin    = bin_size
        self._scale  = scale
        self._counts: dict = {}

    def visit(self, s: dict) -> float:
        key = (s["room"], s["x"] // self._bin, s["y"] // self._bin)
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        return self._scale / (count ** 0.5)

    @property
    def n_unique(self) -> int:
        return len(self._counts)


# ---------------------------------------------------------------------------
# Subgoal detector (simplified from throng5 montezuma_subgoals.py)
# ---------------------------------------------------------------------------

class SubgoalDetector:
    """
    14 spatial milestones for room 1. Each fires at most once per episode.
    Calibrated from live RAM logs and human playthroughs.
    """

    GOALS = [
        # name, condition_fn, reward
        ("ladder_top",         lambda x,y,r,k: r==1 and x<73 and y<235,           0.3),
        ("center_descended",   lambda x,y,r,k: r==1 and 60<=x<=90 and y<=200,     0.8),
        ("platform_right",     lambda x,y,r,k: r==1 and x>100 and 185<=y<=200,    1.2),
        ("rope_grabbed",       lambda x,y,r,k: r==1 and x>108 and y<195,          2.0),
        ("right_platform",     lambda x,y,r,k: r==1 and x>128 and 145<=y<=185,    3.0),
        ("right_ladder_top",   lambda x,y,r,k: r==1 and x>130 and 148<=y<=165,    1.5),
        ("lower_floor",        lambda x,y,r,k: r==1 and y<=150,                   2.0),
        ("skull_crossed",      lambda x,y,r,k: r==1 and x<=37 and y<=155,         6.0),
        ("left_ladder_base",   lambda x,y,r,k: r==1 and 18<=x<=23 and y<=152,     6.3),
        ("left_ladder_mid",    lambda x,y,r,k: r==1 and 18<=x<=23 and 153<=y<=181,6.6),
        ("key_side",           lambda x,y,r,k: r==1 and x<=25 and y>=182,         7.0),
        ("key_corner",         lambda x,y,r,k: r==1 and x<25 and 185<=y<=215,     8.0),
        ("key_collected",      lambda x,y,r,k: k,                                 5000.0),
        ("room_advanced",      lambda x,y,r,k: r > 1,                             10000.0),
    ]

    def __init__(self):
        self._awarded: set = set()
        self._prev_room = None

    def reset(self):
        self._awarded.clear()
        self._prev_room = None

    def check(self, s: dict) -> float:
        x, y, r, k = s["x"], s["y"], s["room"], s["key"]
        bonus = 0.0
        for name, cond, reward in self.GOALS:
            if name not in self._awarded and cond(x, y, r, k):
                self._awarded.add(name)
                bonus += reward
                if name == "key_collected":
                    # Unlock outbound milestones for return leg
                    for unlock in ("rope_grabbed", "right_platform", "platform_right"):
                        self._awarded.discard(unlock)
        self._prev_room = r
        return bonus

    @property
    def n_subgoals(self) -> int:
        return len(self._awarded)


# ---------------------------------------------------------------------------
# RAM Q-Learner
# ---------------------------------------------------------------------------

class MontezumaRAMQLearner:
    """
    Tabular Q-learner on (room, x//8, y//8, key) state.
    Uses shaped reward: novelty + subgoals + death penalty.
    """

    def __init__(
        self,
        n_actions:    int   = 18,
        bin_size:     int   = 8,
        lr:           float = 0.2,
        gamma:        float = 0.99,
        eps_start:    float = 1.0,
        eps_end:      float = 0.05,
        eps_decay:    float = 0.997,
        death_pen:    float = -5.0,
    ):
        self.n_actions = n_actions
        self.bin_size  = bin_size
        self.lr        = lr
        self.gamma     = gamma
        self.epsilon   = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.death_pen = death_pen

        self.q_table:  dict = {}
        self._novelty  = NoveltyTracker(bin_size=bin_size, scale=0.5)
        self._subgoals = SubgoalDetector()
        self._prev_lives: int = 5
        self._prev_key:   tuple = None
        self._prev_action: int  = 0
        self._total_eps    = 0

    def _get_q(self, key: tuple) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[key]

    def act(self, s: dict) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        key = q_state_key(s, self.bin_size)
        return int(np.argmax(self._get_q(key)))

    def learn(self, s: dict, action: int, s_next: dict,
              game_reward: float, done: bool) -> float:
        """Update Q-table; return shaped reward for logging."""
        # Shaped reward
        novelty  = self._novelty.visit(s)
        subgoal  = self._subgoals.check(s_next)
        death    = self.death_pen if s_next["lives"] < s["lives"] else 0.0
        shaped   = game_reward + novelty + subgoal + death

        key      = q_state_key(s, self.bin_size)
        key_next = q_state_key(s_next, self.bin_size)
        q        = self._get_q(key)
        q_next   = self._get_q(key_next)
        target   = shaped + (0.0 if done else self.gamma * np.max(q_next))
        q[action] += self.lr * (target - q[action])

        if done:
            self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
            self._total_eps += 1
            self._subgoals.reset()

        return shaped

    def stats(self) -> dict:
        return {
            "q_states":   len(self.q_table),
            "unique_pos": self._novelty.n_unique,
            "subgoals":   self._subgoals.n_subgoals,
            "epsilon":    round(self.epsilon, 3),
            "episodes":   self._total_eps,
        }


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def make_env():
    import gymnasium as gym, ale_py
    gym.register_envs(ale_py)
    return gym.make(
        "ALE/MontezumaRevenge-v5",
        render_mode="rgb_array",
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",     type=int, default=500)
    parser.add_argument("--render-every", type=int, default=0,
                        help="Print ASCII snapshot + RAM state every N episodes")
    parser.add_argument("--bin-size",     type=int, default=8,
                        help="Position bin size for Q-state (smaller = finer grid)")
    args = parser.parse_args()

    enc = AsciiEncoder(rows=20, cols=15)
    agent = MontezumaRAMQLearner(bin_size=args.bin_size)

    print("=" * 65)
    print(f"MONTEZUMA RAM Q-LEARNER  ({args.episodes} eps, bin={args.bin_size})")
    print("=" * 65)
    print("  State: (room, x//bin, y//bin, key_collected)")
    print("  Reward: game + novelty + subgoals(14) + death_penalty")
    print("  Key: x=15, y=201, room=1  [calibrated from RAM log]")
    print()

    try:
        env = make_env()
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Actions: {env.action_space.n}, obs: {env.observation_space.shape}")
    print()

    rewards, shaped_rewards, found_key, found_room2 = [], [], 0, 0
    t_start = time.time()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        enc.reset()
        ram    = env.unwrapped.ale.getRAM()
        s      = read_ram_state(ram)
        done   = False
        ep_r   = 0.0
        ep_sr  = 0.0

        while not done:
            action = agent.act(s)
            obs, game_r, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            ram    = env.unwrapped.ale.getRAM()
            s_next = read_ram_state(ram)
            shaped = agent.learn(s, action, s_next, game_r, done)
            ep_r  += game_r
            ep_sr += shaped

            if game_r > 0:
                print(f"    [REWARD] ep={ep+1} game={game_r:.0f} total={ep_r:.0f}"
                      f"  pos=({s_next['x']},{s_next['y']}) room={s_next['room']}"
                      f"  key={s_next['key']}")

            s = s_next

        rewards.append(ep_r)
        shaped_rewards.append(ep_sr)
        if ep_r >= 100:
            found_key += 1
        if ep_r >= 300:
            found_room2 += 1

        # ASCII + RAM snapshot
        if args.render_every > 0 and (ep + 1) % args.render_every == 0:
            grid = enc.encode(obs)
            st   = agent.stats()
            print(f"\n  [ep {ep+1}] game_r={ep_r:.0f}  shaped_r={ep_sr:.1f}")
            print(f"  RAM: room={s['room']}  x={s['x']}  y={s['y']}  key={s['key']}")
            print(f"  Q-states={st['q_states']}  unique_pos={st['unique_pos']}"
                  f"  subgoals={st['subgoals']}  eps={st['epsilon']}")
            print(enc.to_text(grid))
            print()

        if (ep + 1) % 100 == 0:
            st   = agent.stats()
            avg  = np.mean(rewards[-100:])
            avgs = np.mean(shaped_rewards[-100:])
            elapsed = time.time() - t_start
            print(f"  ep {ep+1:5d}  avg_game={avg:6.1f}  avg_shaped={avgs:8.1f}"
                  f"  Q={st['q_states']:5d}  pos={st['unique_pos']:5d}"
                  f"  eps={st['epsilon']:.3f}  t={elapsed:.0f}s")

    env.close()

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    st = agent.stats()
    print(f"  Episodes:          {args.episodes}")
    print(f"  Key found (100+):  {found_key} ({100*found_key/args.episodes:.1f}%)")
    print(f"  Room 2 (300+):     {found_room2} ({100*found_room2/args.episodes:.1f}%)")
    print(f"  Max game reward:   {max(rewards):.0f}")
    print(f"  Avg last 100 game: {np.mean(rewards[-100:]):.1f}")
    print(f"  Q-table states:    {st['q_states']}")
    print(f"  Unique positions:  {st['unique_pos']}")
    print(f"  Total time:        {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
