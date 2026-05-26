# -*- coding: utf-8 -*-
"""
elite_replay.py -- Best-N trajectory store per game.

Keeps the top-N (default 3) complete trajectories ever seen for a game.
Seed with human playthroughs; ML runs auto-evict the worst once they
surpass it. On reload the buffer picks up exactly where it left off.

FCEUX .fm2 import stub included (parse_fm2) -- wires in NES recordings
when you are ready to use them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------
# TRAJECTORY DATACLASS
# ---------------------------------------------------------------

@dataclass
class Trajectory:
    """One complete episode recording."""
    actions:  List[int]              # action at each step
    score:    float                  # total (shaped or raw) episode reward
    label:    str  = "agent"         # "human" | "agent"
    game:     str  = ""
    episode:  int  = 0
    # obs_seq stored separately (can be None to save memory)
    # obs_seq: Optional[List[np.ndarray]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "actions": self.actions,
            "score":   self.score,
            "label":   self.label,
            "game":    self.game,
            "episode": self.episode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            actions  = d["actions"],
            score    = float(d["score"]),
            label    = d.get("label", "agent"),
            game     = d.get("game", ""),
            episode  = d.get("episode", 0),
        )


# ---------------------------------------------------------------
# ELITE REPLAY BUFFER  (single game)
# ---------------------------------------------------------------

class EliteReplayBuffer:
    """
    Keeps the top-N trajectories for one game.

    Eviction policy: when a new trajectory arrives and the buffer is full,
    evict the one with the lowest score IF the new score exceeds it.
    Human-labelled trajectories are protected until the agent score
    exceeds them (same eviction rule applies -- no special immunity).
    """

    def __init__(self, game: str, n: int = 3):
        self.game = game
        self.n    = n
        self._elites: List[Trajectory] = []

    # -- query -------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._elites)

    @property
    def min_score(self) -> float:
        if not self._elites:
            return -float("inf")
        return min(t.score for t in self._elites)

    @property
    def max_score(self) -> float:
        if not self._elites:
            return -float("inf")
        return max(t.score for t in self._elites)

    def scores(self) -> List[float]:
        return sorted((t.score for t in self._elites), reverse=True)

    # -- mutation -----------------------------------------------------

    def add(self, traj: Trajectory) -> bool:
        """
        Try to add trajectory. Returns True if accepted.
        Accepted if buffer not full OR new score > current worst.
        """
        if len(self._elites) < self.n:
            self._elites.append(traj)
            return True
        worst_idx = int(np.argmin([t.score for t in self._elites]))
        if traj.score > self._elites[worst_idx].score:
            evicted = self._elites[worst_idx]
            self._elites[worst_idx] = traj
            return True
        return False

    def seed_human(
        self,
        actions:  List[int],
        score:    float,
        episode:  int = -1,
    ) -> bool:
        """Convenience wrapper -- label='human'."""
        traj = Trajectory(
            actions=actions, score=score,
            label="human", game=self.game, episode=episode,
        )
        return self.add(traj)

    def sample(self, rng: Optional[np.random.RandomState] = None) -> Trajectory:
        """Uniform random sample from stored elites."""
        if not self._elites:
            raise ValueError(f"EliteReplayBuffer for {self.game!r} is empty")
        if rng is None:
            rng = np.random.RandomState()
        idx = rng.randint(len(self._elites))
        return self._elites[idx]

    # -- persistence -------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = [t.to_dict() for t in self._elites]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"game": self.game, "n": self.n, "elites": data}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EliteReplayBuffer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        buf = cls(game=d["game"], n=d["n"])
        buf._elites = [Trajectory.from_dict(t) for t in d.get("elites", [])]
        return buf

    def __repr__(self) -> str:
        sc = [f"{t.score:.1f}({t.label[0]})" for t in self._elites]
        return f"EliteReplayBuffer({self.game!r}, n={self.n}, scores={sc})"


# ---------------------------------------------------------------
# ELITE REPLAY MANAGER  (multi-game)
# ---------------------------------------------------------------

class EliteReplayManager:
    """
    One EliteReplayBuffer per game. Handles injection into training loop.
    """

    def __init__(self, games: List[str], n: int = 3):
        self.n = n
        self._buffers: Dict[str, EliteReplayBuffer] = {
            g: EliteReplayBuffer(g, n=n) for g in games
        }
        self._rng = np.random.RandomState(0)

    # -- per-game access ---------------------------------------------

    def buffer(self, game: str) -> EliteReplayBuffer:
        if game not in self._buffers:
            self._buffers[game] = EliteReplayBuffer(game, n=self.n)
        return self._buffers[game]

    def try_add(self, game: str, actions: List[int], score: float,
                episode: int = 0, label: str = "agent") -> bool:
        """Add trajectory if it qualifies as an elite."""
        traj = Trajectory(actions=actions, score=score,
                          label=label, game=game, episode=episode)
        accepted = self.buffer(game).add(traj)
        return accepted

    def seed_human(self, game: str, actions: List[int], score: float) -> bool:
        return self.buffer(game).seed_human(actions, score)

    # -- injection into training loop --------------------------------

    def inject_episode(
        self,
        runner,
        enc,
        game_name: str,
        world_model,
        game_id:   int,
        force:     bool = False,
        p:         float = 0.20,
    ) -> Optional[dict]:
        """
        With probability p (or always if force=True), replay a sampled elite
        trajectory through the runner and push transitions into world_model
        as if the agent had just performed them.

        Returns a stats dict, or None if skipped / buffer empty.
        """
        buf = self.buffer(game_name)
        if buf.size == 0:
            return None
        if not force and self._rng.rand() > p:
            return None

        traj   = buf.sample(rng=self._rng)
        n_act  = getattr(runner, "n_actions", 4)
        total_r = 0.0
        steps   = 0
        transitions_added = 0

        try:
            obs = runner.reset()
        except Exception:
            return None

        try:
            prev_z = enc.encode(game_name, np.asarray(obs, dtype=np.float32).flatten())
        except Exception:
            prev_z = np.zeros(enc.out_dim, dtype=np.float32)

        for action in traj.actions:
            action = int(action) % n_act   # safety clamp

            try:
                next_obs, reward, done = runner.step(action)
            except Exception:
                break

            try:
                next_z = enc.encode(
                    game_name, np.asarray(next_obs, dtype=np.float32).flatten()
                )
            except Exception:
                next_z = np.zeros(enc.out_dim, dtype=np.float32)

            if world_model is not None:
                try:
                    world_model.store_transition(prev_z, action, next_z, reward, game_id)
                    transitions_added += 1
                except Exception:
                    pass

            total_r  += reward
            prev_z    = next_z
            steps    += 1

            if done:
                break

        return {
            "game":               game_name,
            "elite_score":        traj.score,
            "replayed_score":     total_r,
            "steps":              steps,
            "transitions_added":  transitions_added,
            "label":              traj.label,
        }

    # -- persistence -------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for game, buf in self._buffers.items():
            safe = game.replace("/", "_")
            buf.save(os.path.join(directory, f"elite_{safe}.json"))
        with open(os.path.join(directory, "elite_meta.json"), "w") as f:
            json.dump({"n": self.n, "games": list(self._buffers)}, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "EliteReplayManager":
        meta_path = os.path.join(directory, "elite_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No elite_meta.json in {directory!r}")
        with open(meta_path) as f:
            meta = json.load(f)
        mgr = cls(games=meta["games"], n=meta["n"])
        for game in meta["games"]:
            safe = game.replace("/", "_")
            p = os.path.join(directory, f"elite_{safe}.json")
            if os.path.exists(p):
                mgr._buffers[game] = EliteReplayBuffer.load(p)
        return mgr

    def stats(self) -> Dict[str, dict]:
        return {
            g: {"size": b.size, "scores": b.scores(), "min": b.min_score, "max": b.max_score}
            for g, b in self._buffers.items()
        }

    def __repr__(self) -> str:
        return f"EliteReplayManager(n={self.n}, games={list(self._buffers)})"


# ---------------------------------------------------------------
# FCEUX .fm2 IMPORTER
# ---------------------------------------------------------------

# Button positions in the 8-char P1 field:
#   index  0  1  2  3  4  5  6  7
#   button R  L  D  U  S  s  B  A
#
# MarioRunner discrete actions (src/games/mario/mario_adapter.py):
#   0=noop  1=right  2=left  3=jump(A)
#   4=run+right(B+R)  5=run+jump+right(B+R+A)
#   6=run+left(B+L)   7=run+jump+left(B+L+A)

_BTN_R, _BTN_L = 0, 1
_BTN_B, _BTN_A = 6, 7


def _buttons_to_action(field: str) -> int:
    """Map an 8-char button field from fm2 data line to a MarioRunner action int."""
    if len(field) < 8:
        return 0
    press = [c != "." for c in field]
    R = press[_BTN_R]
    L = press[_BTN_L]
    B = press[_BTN_B]
    A = press[_BTN_A]
    if B and R and A:
        return 5   # run-jump-right
    if B and L and A:
        return 7   # run-jump-left
    if B and R:
        return 4   # run-right
    if B and L:
        return 6   # run-left
    if A:
        return 3   # jump
    if R:
        return 1   # right
    if L:
        return 2   # left
    return 0       # noop


def parse_fm2(path: str) -> Tuple[List[int], int]:
    """
    Parse a FCEUX .fm2 movie file into (action_list, frame_count).

    Data lines have the format:  |lag|P1field|P2field||
    where P1field is 8 chars: R L D U S s B A (dot=not pressed).

    Returns:
        (actions, frame_count) where actions maps each frame to a
        MarioRunner action int {0..7}.
    """
    actions: List[int] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            # parts: ['', lag, P1field, P2field, '', ...]
            if len(parts) < 3:
                continue
            p1_field = parts[2]
            actions.append(_buttons_to_action(p1_field))
    return actions, len(actions)

