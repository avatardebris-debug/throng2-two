# -*- coding: utf-8 -*-
"""
fceux_adapter.py -- Python side of the FCEUX file-based bridge.

Implements the same runner interface as MarioRunner / GymRunner so it
drops into cross_game_training.py and EliteReplayManager without changes.

Communication via shared directory (default C:/fceux_bridge/):
    obs.txt   -- Lua WRITES observation + reward + done
    act.txt   -- Python WRITES action for Lua to read
    ready.txt -- Python WRITES to signal "I'm ready, start bridging"

Protocol each step:
    1. Python reads obs.txt  (polls until Lua writes it)
    2. Python writes act.txt (Lua polls for this)
    3. Lua presses buttons, advances frame, writes new obs.txt

Observation vector (8 floats, normalised 0-1):
    [mario_x, mario_y, page_x, world, level, lives, score, is_big]

Actions (9):
    0=noop  1=right  2=left  3=jump
    4=run+right  5=run+jump+right  6=run+left  7=run+jump+left  8=down
    -1 = reset (special)
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════
# BRIDGE DIRECTORY
# ══════════════════════════════════════════════════════════════

BRIDGE_DIR = os.environ.get("FCEUX_BRIDGE_DIR", "C:/Users/avata/fceux_bridge")


class FCEUXAdapter:
    """
    Drop-in runner that drives the real NES ROM via FCEUX + Lua file bridge.

    Usage:
        adapter = FCEUXAdapter()
        obs = adapter.reset()
        obs, reward, done = adapter.step(5)  # run-jump-right
        adapter.close()
    """

    OBS_DIM   = 8
    N_ACTIONS = 9

    def __init__(
        self,
        bridge_dir:  str   = BRIDGE_DIR,
        timeout:     float = 300.0,
        poll_interval: float = 0.02,   # 20ms between file polls
        verbose:     bool  = False,
        **kwargs,  # absorb unused args like host/port/fm2_path
    ):
        self.bridge_dir    = bridge_dir
        self.timeout       = timeout
        self.poll_interval = poll_interval
        self.verbose       = verbose

        self.n_actions = self.N_ACTIONS
        self.obs_dim   = self.OBS_DIM

        self._obs_path   = os.path.join(bridge_dir, "obs.txt")
        self._act_path   = os.path.join(bridge_dir, "act.txt")
        self._act_tmp    = os.path.join(bridge_dir, "act.tmp")
        self._ready_path = os.path.join(bridge_dir, "ready.txt")

        self._last_obs: np.ndarray = np.zeros(self.OBS_DIM, dtype=np.float32)

        self._setup()

    def _setup(self) -> None:
        """Create bridge dir and signal readiness to Lua."""
        os.makedirs(self.bridge_dir, exist_ok=True)
        # Clean up stale files
        for f in (self._obs_path, self._act_path, self._act_tmp, self._ready_path):
            try:
                os.remove(f)
            except OSError:
                pass
        # Signal to Lua: "Python is ready"
        with open(self._ready_path, "w") as f:
            f.write("ready\n")
        if self.verbose:
            print(f"[FCEUXAdapter] Bridge dir: {self.bridge_dir}")
            print(f"[FCEUXAdapter] ready.txt written — waiting for Lua to start...")

    # ── runner interface ─────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Signal reset to Lua, return first observation."""
        self._write_action(-1)  # -1 = reset signal
        obs, _, _ = self._read_obs()
        self._last_obs = obs
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Send action, receive (obs, reward, done)."""
        action = int(action) % self.n_actions
        self._write_action(action)
        obs, reward, done = self._read_obs()
        self._last_obs = obs
        return obs, float(reward), bool(done)

    # ── file I/O ────────────────────────────────────────────────

    def _write_action(self, action: int) -> None:
        """Write action directly to act.txt with retry on Windows locking errors."""
        for attempt in range(50):   # retry up to 250ms total
            try:
                with open(self._act_path, "w") as f:
                    f.write(f"{action}\n")
                return  # success
            except OSError:
                time.sleep(0.005)   # 5ms — let Lua finish reading previous act.txt
        if self.verbose:
            print(f"[FCEUXAdapter] write gave up after 50 retries (action={action})")

    def _read_obs(self) -> Tuple[np.ndarray, float, int]:
        """Poll obs.txt until Lua writes it, parse and delete (so next step gets a fresh one)."""
        t0 = time.time()
        _err_count = 0
        while True:
            try:
                with open(self._obs_path, "r") as f:
                    line = f.read().strip()
                if line:
                    # Delete so we don't re-read stale obs next step
                    try:
                        os.remove(self._obs_path)
                    except OSError:
                        pass
                    return self._parse_obs(line)
            except FileNotFoundError:
                pass
            except OSError:
                _err_count += 1
                if self.verbose and _err_count % 50 == 1:
                    print(f"[FCEUXAdapter] obs.txt locked, retrying... ({_err_count})")

            if time.time() - t0 > self.timeout:
                if self.verbose:
                    print(f"[FCEUXAdapter] timeout waiting for obs.txt")
                return self._last_obs, 0.0, 1

            time.sleep(self.poll_interval)

    def _parse_obs(self, line: str) -> Tuple[np.ndarray, float, int]:
        """Parse 'f1,f2,...,f8|reward|done' format."""
        try:
            parts = line.split("|")
            obs_vals = [float(x) for x in parts[0].split(",")]
            obs = np.array(obs_vals[:self.OBS_DIM], dtype=np.float32)
            reward = float(parts[1]) if len(parts) > 1 else 0.0
            done   = int(parts[2])   if len(parts) > 2 else 0
            return obs, reward, done
        except Exception as e:
            if self.verbose:
                print(f"[FCEUXAdapter] parse error: {e} | line: {line!r}")
            return self._last_obs, 0.0, 0

    # ── lifecycle ────────────────────────────────────────────────

    def close(self) -> None:
        """Clean up bridge files."""
        for f in (self._obs_path, self._act_path, self._act_tmp, self._ready_path):
            try:
                os.remove(f)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (f"FCEUXAdapter(bridge_dir={self.bridge_dir!r}, "
                f"n_actions={self.n_actions}, obs_dim={self.obs_dim})")


# ══════════════════════════════════════════════════════════════
# FCEUX PROCESS LAUNCHER
# ══════════════════════════════════════════════════════════════

FCEUX_DEFAULT_PATHS = [
    r"C:\fceux-win64\fceux64.exe",
    r"C:\fceux64.exe",
    r"C:\fceux\fceux64.exe",
    r"C:\fceux\fceux.exe",
    r"C:\tools\fceux\fceux64.exe",
    r"C:\tools\fceux\fceux.exe",
    r"C:\Program Files\FCEUX\fceux.exe",
    r"C:\Program Files (x86)\FCEUX\fceux.exe",
]

NES_ROM_DIR = r"C:\Users\avata\aicompete\throng5\roms\nes"

MARIO_ROM_PATHS = [
    r"C:\Users\avata\aicompete\throng5\roms\nes\Super Mario Bros. + Duck Hunt (USA).nes",
    r"C:\Super Mario Bros. + Duck Hunt (USA).nes",
]


def find_mario_rom() -> Optional[str]:
    """Return path to the Mario+Duck Hunt ROM, or None."""
    for p in MARIO_ROM_PATHS:
        if Path(p).exists():
            return p
    return None


def find_fceux() -> Optional[str]:
    """Try common install locations, return path or None."""
    for p in FCEUX_DEFAULT_PATHS:
        if Path(p).exists():
            return p
    return None


def launch_fceux(
    rom_path:    str,
    lua_script:  str,
    fceux_exe:   Optional[str] = None,
    bridge_dir:  str = BRIDGE_DIR,
) -> subprocess.Popen:
    """Start FCEUX with the Lua bridge script."""
    exe = fceux_exe or find_fceux()
    if exe is None:
        raise FileNotFoundError(
            "fceux.exe not found. Install FCEUX and pass fceux_exe= or "
            "add it to one of: " + ", ".join(FCEUX_DEFAULT_PATHS)
        )

    env = dict(os.environ)
    env["FCEUX_BRIDGE_DIR"] = bridge_dir

    cmd = [exe, "--lua", lua_script, rom_path]
    proc = subprocess.Popen(cmd, env=env)
    time.sleep(2.0)
    return proc
