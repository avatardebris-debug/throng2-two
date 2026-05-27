"""
watch_agent.py --- Visual Verification Tool.

Usage:
    python examples/watch_agent.py --game pong
    
This script:
    1. Loads the latest brain weights for the specified game.
    2. Runs a REAL Atari interaction (not imagined).
    3. Renders the 15 x 20 ASCII grid to the console.
    4. Shows the agent's internal state and rewards.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.getcwd())

from src.games.atari.atari_adapter import make_atari_adapter
from src.encoder.universal_encoder import UniversalEncoder
from src.cell.world_model import CellWorldModel
from src.learning.q_policy import QPolicy

def render_grid(grid_flat: np.ndarray, rows: int = 15, cols: int = 20):
    """Render flattened grid to console."""
    grid = grid_flat.reshape((rows, cols))
    
    # ANSI escape to clear terminal (might vary by OS, but clean for most)
    # os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "=" * 42)
    for r in range(rows):
        line = "["
        for c in range(cols):
            val = grid[r, c]
            if val > 0.8: line += "# "   # Solid object
            elif val > 0.3: line += "o " # Moving object / ball
            elif val > 0.05: line += ". " # Background / low density
            else: line += "  "           # Empty
        line += "]"
        print(line)
    print("=" * 42)

def main():
    parser = argparse.ArgumentParser(description="Throng2 Agent Watcher")
    parser.add_argument("--game", type=str, default="pong", help="Atari game to watch")
    parser.add_argument("--z-dim", type=int, default=16, help="Latent space dimension")
    parser.add_argument("--fps", type=float, default=10, help="Frames per second to display")
    args = parser.parse_args()

    # --- Setup Directories ---
    ckpt_dir = Path(f"checkpoints/{args.game}")
    path_wm = ckpt_dir / "world_model.pth"
    path_enc = ckpt_dir / "encoder.npz"
    path_policy = ckpt_dir / "policy.pth"

    if not path_wm.exists():
        print(f"\n[ERROR] No checkpoint found for {args.game} in {ckpt_dir}.")
        print("Please run marathon_gauntlet.py first to train the agent.")
        return

    # --- Initialize Components ---
    print(f"\n[LOAD] Loading Brain for {args.game.upper()}...")
    encoder = UniversalEncoder(game_name=args.game, z_dim=args.z_dim)
    wm = CellWorldModel(feature_dim=encoder.out_dim, n_actions=6)
    policy = QPolicy(input_dim=encoder.out_dim, n_actions=6)
    
    # Load weights
    wm.load_weights(str(path_wm))
    encoder.load_projection(str(path_enc))
    policy.load(str(path_policy))
    
    # --- Environment ---
    print(f"  [INIT] Launching REAL environment...")
    adapter = make_atari_adapter(args.game)
    obs = adapter.reset()
    
    total_reward = 0
    step = 0
    
    try:
        while True:
            # 1. Encode observation
            # We want to show the preprocessed grid to the user
            # UniversalEncoder._preprocess() returns the flat grid
            grid_flat = encoder._preprocess(obs)
            
            # 2. Select action (Greedy / Eval mode)
            action = policy.select_action(encoder.encode(obs), eval_mode=True)
            
            # 3. Step real environment
            next_obs, reward, done, _ = adapter.step(action)
            total_reward += reward
            step += 1
            
            # 4. Render
            render_grid(grid_flat)
            print(f"  Step: {step} | Reward: {total_reward} | Action: {action} | Epsilon: {policy.epsilon:.3f}")
            
            if done:
                print("\n  [EPISODE END] Restarting...")
                time.sleep(1.0)
                obs = adapter.reset()
                total_reward = 0
                step = 0
            else:
                obs = next_obs
            
            # Control speed for visibility
            time.sleep(1.0 / args.fps)
            
    except KeyboardInterrupt:
        print("\n\n[EXIT] Watcher terminated.")

if __name__ == "__main__":
    main()
