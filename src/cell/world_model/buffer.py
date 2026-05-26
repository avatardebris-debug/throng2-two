"""Multi-game replay buffer."""
import numpy as np
from collections import deque
from typing import Dict

class MultiGameReplayBuffer:
    """
    Joint replay buffer that stores transitions from multiple games
    and samples them in a balanced way.

    Each game gets its own deque so infrequent games aren't drowned
    out by high-throughput ones (e.g. Mario at 60 ep/s vs MuJoCo at 0.5).

    Sampling modes:
      "uniform"  — equal probability per transition across all games
      "balanced" — equal probability per game, then uniform within each game
    """

    def __init__(
        self,
        capacity_per_game: int = 5000,
        sampling: str = "balanced",
        horizon_n: int = 8,
        horizon_gamma: float = 0.99,
    ):
        self.capacity_per_game = capacity_per_game
        self.sampling = sampling
        self.horizon_n = horizon_n
        self.horizon_gamma = horizon_gamma

        self._buffers: Dict[int, deque] = {}  # game_id → deque of transitions
        self._total_stored = 0

        # N-step accumulator: per-game sliding window of recent transitions.
        # Once a window reaches horizon_n length, we emit one horizon entry.
        # Each pending entry is (state, action, next_state, reward).
        self._nstep_pending: Dict[int, deque] = {}

        # Horizon replay buffer: stores (state_t, action_t, z_{t+N}, cum_reward, game_id)
        self._horizon_buffers: Dict[int, deque] = {}

    def add(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        game_id: int,
    ):
        """Store a transition tagged with a game_id."""
        if game_id not in self._buffers:
            self._buffers[game_id] = deque(maxlen=self.capacity_per_game)
        self._buffers[game_id].append((
            np.asarray(state, dtype=np.float32),
            int(action),
            np.asarray(next_state, dtype=np.float32),
            float(reward),
            int(game_id),
        ))
        self._total_stored += 1

        # ── N-step horizon accumulation ────────────────────────
        if game_id not in self._nstep_pending:
            self._nstep_pending[game_id] = deque(maxlen=self.horizon_n)
            self._horizon_buffers[game_id] = deque(maxlen=self.capacity_per_game)

        pending = self._nstep_pending[game_id]
        pending.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            np.asarray(next_state, dtype=np.float32),
            float(reward),
        ))

        if len(pending) >= self.horizon_n:
            # Compute N-step discounted return from the earliest entry
            s0, a0 = pending[0][0], pending[0][1]
            z_N = pending[-1][2]  # state reached after N steps
            cum_r = float(sum(
                (self.horizon_gamma ** k) * pending[k][3]
                for k in range(self.horizon_n)
            ))
            self._horizon_buffers[game_id].append((
                s0, a0, z_N, cum_r, int(game_id),
            ))

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions.

        Returns:
            List of (state, action, next_state, reward, game_id) tuples.
        """
        games_with_data = [gid for gid, buf in self._buffers.items() if len(buf) > 0]
        if not games_with_data:
            return []

        if self.sampling == "balanced":
            # Sample equally from each game that has data
            per_game = max(1, batch_size // len(games_with_data))
            batch = []
            for gid in games_with_data:
                buf = self._buffers[gid]
                n = min(per_game, len(buf))
                indices = np.random.choice(len(buf), n, replace=False)
                batch.extend(buf[i] for i in indices)
        else:
            # Concatenate all into one pool, sample uniformly
            all_transitions = []
            for buf in self._buffers.values():
                all_transitions.extend(buf)
            n = min(batch_size, len(all_transitions))
            indices = np.random.choice(len(all_transitions), n, replace=False)
            batch = [all_transitions[i] for i in indices]

        return batch

    @property
    def total_stored(self) -> int:
        return self._total_stored

    @property
    def size(self) -> int:
        return sum(len(b) for b in self._buffers.values())

    def game_sizes(self) -> Dict[int, int]:
        return {gid: len(buf) for gid, buf in self._buffers.items()}

    def is_ready(self, min_per_game: int = 50) -> bool:
        """True if EVERY registered game has at least min_per_game transitions."""
        if not self._buffers:
            return False
        return all(len(b) >= min_per_game for b in self._buffers.values())

    def is_game_ready(self, game_id: int, min_transitions: int = 50) -> bool:
        """True if one specific game has enough transitions to use its head."""
        buf = self._buffers.get(game_id)
        return buf is not None and len(buf) >= min_transitions

    def sample_for_game(
        self,
        game_id: int,
        batch_size: int,
    ) -> list:
        """
        Sample a batch from a single game's buffer.

        Useful for per-game readiness: train a game's head as soon as
        that game's buffer is ready, without waiting for all other games.
        """
        buf = self._buffers.get(game_id)
        if buf is None or len(buf) == 0:
            return []
        n = min(batch_size, len(buf))
        indices = np.random.choice(len(buf), n, replace=False)
        return [buf[i] for i in indices]

    def sample_horizon(self, batch_size: int) -> list:
        """
        Sample a batch of N-step horizon targets (balanced across games).

        Returns:
            List of (state_t, action_t, z_{t+N}, cum_reward, game_id) tuples.
            Empty list if no horizon data is available yet.
        """
        games_with_data = [
            gid for gid, buf in self._horizon_buffers.items() if len(buf) > 0
        ]
        if not games_with_data:
            return []

        per_game = max(1, batch_size // len(games_with_data))
        batch = []
        for gid in games_with_data:
            buf = self._horizon_buffers[gid]
            n = min(per_game, len(buf))
            indices = np.random.choice(len(buf), n, replace=False)
            batch.extend(buf[i] for i in indices)
        return batch

    def horizon_size(self) -> Dict[int, int]:
        """Number of N-step entries per game."""
        return {gid: len(buf) for gid, buf in self._horizon_buffers.items()}

    def is_horizon_ready(self, min_per_game: int = 20) -> bool:
        """True if every game has at least min_per_game horizon entries."""
        if not self._horizon_buffers:
            return False
        return all(len(b) >= min_per_game for b in self._horizon_buffers.values())

