r"""
fceux_mario_training.py -- Train on the real NES Mario ROM via FCEUX file bridge.

Usage:
    # Step 1: Start FCEUX with the Lua bridge loaded:
    #   src\games\mario\fceux_launcher.bat "path\to\Mario.nes"
    #   FCEUX screen will show "Waiting for Python..."

    # Step 2: Start Python training (any order is fine):
    python examples/fceux_mario_training.py --episodes 200 --verbose

    # Resume from checkpoint:
    python examples/fceux_mario_training.py --episodes 100 --resume results/checkpoints_mario_fceux/ep000200

Options:
    --rom PATH            Path to NES ROM (auto-detected)
    --episodes N          Training episodes (default 200)
    --auto-launch         Let Python start FCEUX subprocess
    --seed-demos          Seed elite buffer from fm2 before training
    --fm2 PATH            .fm2 file for elite seeding
    --verbose
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.games.mario.fceux_adapter import FCEUXAdapter, launch_fceux, find_mario_rom
from src.learning.elite_replay import EliteReplayManager, parse_fm2
from src.learning.checkpoint_manager import CheckpointManager


# World model (optional torch dep)
try:
    from src.cell.world_model import MultiGameWorldModel
    _WM_AVAILABLE = True
except ImportError:
    _WM_AVAILABLE = False

LUA_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "games", "mario", "fceux_bridge.lua"
)
GAME_NAME  = "mario"
GAME_ID    = 0


# ──────────────────────────────────────────────────────────────
# SIMPLE AGENT (reuse from cross_game_training)
# ──────────────────────────────────────────────────────────────

class SimpleAgent:
    """Epsilon-greedy linear Q-agent for the FCEUX runner."""
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3, eps: float = 0.3):
        self.n_actions = n_actions
        self.eps = eps
        self.lr  = lr
        self.W = np.random.randn(obs_dim, n_actions).astype(np.float32) * 0.01
        self.b = np.zeros(n_actions, dtype=np.float32)
        self._last_obs = None
        self._last_act = 0

    def reset(self):
        self._last_obs = None

    def step(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        self._last_obs = obs
        if np.random.random() < self.eps:
            act = np.random.randint(self.n_actions)
        else:
            act = int(np.argmax(obs @ self.W + self.b))
        self._last_act = act
        return act

    def learn(self, reward: float, next_obs: np.ndarray, done: bool):
        if self._last_obs is None:
            return
        obs = self._last_obs
        q_pred = obs @ self.W + self.b
        target = q_pred.copy()
        nq = next_obs @ self.W + self.b
        target[self._last_act] = reward + (0.99 * np.max(nq) * (not done))
        err = target - q_pred
        self.W += self.lr * np.outer(obs, err)
        self.b += self.lr * err


# ──────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────

def run_training(args):
    print("═══ FCEUX Mario Training ═══")
    print(f"  ROM:      {args.rom}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Lua:      {LUA_SCRIPT}")
    if args.fm2:
        print(f"  fm2:      {args.fm2}")
    print()

    # -- Launch FCEUX if requested --------------------------------
    fceux_proc = None
    if args.auto_launch:
        print("  Launching FCEUX...")
        fceux_proc = launch_fceux(
            rom_path   = args.rom,
            lua_script = LUA_SCRIPT,
            fm2_path   = args.fm2 or "",
            fceux_exe  = args.fceux,
        )
        print(f"  FCEUX PID: {fceux_proc.pid}")
    else:
        print("  Waiting for you to start FCEUX with fceux_bridge.lua loaded...")
        print(f"  Command:  fceux --lua \"{LUA_SCRIPT}\" \"{args.rom}\"")
        print()

    # -- Connect adapter ------------------------------------------
    if args.wait_for_ready:
        print("  Fix your controller inputs in FCEUX, then press Enter here...")
        input("  [Press Enter to continue] ")
        print()

    adapter = FCEUXAdapter(
        bridge_dir = args.bridge_dir,
        timeout    = args.timeout,
        verbose    = args.verbose,
    )
    print("  Bridge ready — Lua will connect when it sees ready.txt\n")

    # -- Verify bridge is alive (quick 5-step check) ---------------
    print("  Verifying bridge connection (5 steps)...")
    obs = adapter.reset()
    for i in range(5):
        obs, _, done = adapter.step(1)  # step right
        print(f"    step {i+1}: obs={obs.round(3)}")
        if done:
            obs = adapter.reset()
    print("  Bridge alive.\n")

    # -- World model (optional) ------------------------------------
    # FCEUX obs is already 8 normalised floats — no PCA needed.
    world_model = None
    if _WM_AVAILABLE:
        try:
            world_model = MultiGameWorldModel(
                feature_dim=adapter.obs_dim, n_actions=adapter.n_actions,
                n_games=1, hidden_size=128,
            )
        except Exception as e:
            print(f"  [WM] disabled: {e}")

    agent = SimpleAgent(obs_dim=adapter.obs_dim, n_actions=adapter.n_actions)

    # -- Elite replay + checkpoint --------------------------------
    ckpt_base    = args.checkpoint_path or "results/checkpoints_mario_fceux"
    elite_replay = EliteReplayManager(games=[GAME_NAME], n=args.elite_n)
    ckpt_manager = CheckpointManager(ckpt_base, keep_last=3)
    start_ep     = 0

    if args.resume:
        try:
            start_ep = ckpt_manager.load(
                args.resume, world_model, enc, {}, elite_replay
            )
            print(f"  Resumed from checkpoint: episode {start_ep}")
        except Exception as e:
            print(f"  [WARN] checkpoint load failed: {e}")

    # Seed human demos from fm2 if requested
    if args.seed_demos and args.fm2:
        try:
            actions, frames = parse_fm2(args.fm2)
            score = float(frames) / 10.0 + 500.0
            ok = elite_replay.seed_human(GAME_NAME, actions, score)
            print(f"  Seeded fm2 demo: {frames} frames, score={score:.0f}, accepted={ok}")
        except Exception as e:
            print(f"  [WARN] fm2 seed failed: {e}")

    print()

    # -- Episode loop ---------------------------------------------
    rewards = []
    t0 = time.time()
    obs = adapter.reset()
    agent.reset()

    for ep in range(args.episodes):
        total_r = 0.0
        steps   = 0
        actions_this_ep = []
        obs = adapter.reset()
        agent.reset()

        while True:
            action = agent.step(obs)
            next_obs, reward, done = adapter.step(action)
            agent.learn(reward, next_obs, done)

            # World model bookkeeping (raw obs, no encoder needed)
            if world_model is not None:
                try:
                    world_model.store_transition(obs, action, next_obs, reward, GAME_ID)
                except Exception:
                    pass

            actions_this_ep.append(action)
            total_r += reward
            steps   += 1
            obs = next_obs
            if done:
                break

        rewards.append(total_r)
        elite_replay.try_add(GAME_NAME, actions_this_ep, total_r, ep + start_ep)

        # World model training
        if world_model is not None and world_model._multi_buffer.size > 50:
            for _ in range(5):
                try:
                    world_model.multi_train_step()
                except Exception:
                    pass

        # Checkpoint
        if args.checkpoint_every > 0 and (ep + 1) % args.checkpoint_every == 0:
            try:
                ckpt_manager.save(
                    ep + start_ep + 1, world_model, None, {}, elite_replay
                )
            except Exception as e:
                if args.verbose:
                    print(f"  [Checkpoint] save failed: {e}")

        # Log
        if (ep + 1) % args.log_every == 0:
            avg = np.mean(rewards[-args.log_every:])
            mx  = max(rewards[-args.log_every:])
            elapsed = int(time.time() - t0)
            print(f"  Episode {ep+1:4d}/{args.episodes}  ({elapsed}s)")
            print(f"    avg_r={avg:8.2f}  best={mx:.2f}  "
                  f"elite={elite_replay.buffer(GAME_NAME).scores()}")
            if world_model is not None:
                st = world_model.multi_stats()
                print(f"    WM: loss={st.get('avg_loss',0):.4f}  "
                      f"buffer={st.get('multi_buffer_size',0)}")
            print()

    # -- Done -----------------------------------------------------
    adapter.close()
    if fceux_proc:
        fceux_proc.terminate()
        print("  FCEUX terminated.")

    print(f"Training complete. Best reward: {max(rewards):.2f}")
    return rewards


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train on real NES Mario via FCEUX bridge")
    parser.add_argument("--rom", type=str, default=None,
                        help="Path to NES ROM (auto-detected if omitted)")
    parser.add_argument("--episodes",         type=int, default=200)
    parser.add_argument("--fceux",            type=str, default=None, help="Path to fceux.exe")
    parser.add_argument("--auto-launch",      action="store_true",   help="Start FCEUX automatically")
    parser.add_argument("--fm2",              type=str, default="",  help="fm2 for elite demo seeding")
    parser.add_argument("--bridge-dir",       type=str, default="C:/Users/avata/fceux_bridge",
                        help="Shared directory for file IPC (default C:/fceux_bridge)")
    parser.add_argument("--timeout",          type=float, default=300.0,
                        help="Seconds to wait for Lua response (default 300)")
    parser.add_argument("--wait-for-ready",   action="store_true",
                        help="Pause for user to fix FCEUX controller inputs")
    parser.add_argument("--z-dim",            type=int, default=32)
    parser.add_argument("--log-every",        type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-path",  type=str, default=None)
    parser.add_argument("--elite-n",          type=int, default=3)
    parser.add_argument("--resume",           type=str, default=None)
    parser.add_argument("--seed-demos",       action="store_true",   help="Seed elite buffer from fm2")
    parser.add_argument("--verbose",          action="store_true")
    args = parser.parse_args()

    # Auto-detect ROM if not supplied
    if not args.rom:
        args.rom = find_mario_rom()
        if not args.rom:
            print("ERROR: ROM not found. Pass --rom path/to/Mario.nes")
            return
        print(f"  ROM auto-detected: {args.rom}")

    run_training(args)


if __name__ == "__main__":
    main()
