"""
mario_gan.py -- GAN for procedural Mario level generation.

Pure numpy implementation. Generator outputs level grids with tile type
channels. Discriminator scores level quality (completable, appropriately
difficult, not too easy, not unsolvable).

Architecture:
  Generator:  z(64) + tier(8) -> 72 -> 256 -> 512 -> cells*channels
  Discriminator: cells*channels -> 512 -> 256 -> 1 (quality score)

Tile channels (11): EMPTY, GROUND, BRICK, QUESTION, PIPE_L, PIPE_R,
                     COIN, PIT, PLATFORM, FLAG, PLAYER

Post-processing enforces valid ground profile, exactly 1 player start,
flag at end, and border constraints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .mario_simulator import (
    Enemy, EnemyType, MarioSimulator, Tile, N_TILE_TYPES,
)


# Grid dimensions for GAN output (single screen)
GRID_H = MarioSimulator.GRID_H      # 16
GRID_W = 20                          # 1 screen width
GRID_CELLS = GRID_H * GRID_W        # 320
N_CHANNELS = N_TILE_TYPES            # 11
OUTPUT_SIZE = GRID_CELLS * N_CHANNELS  # 3520


# ════════════════════════════════════════════════════════════════
# Activation functions (pure numpy)
# ════════════════════════════════════════════════════════════════

def _leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def _leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha).astype(np.float32)

def _sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def _softmax_2d(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)

def _gumbel_softmax(logits, temperature=1.0):
    g = -np.log(-np.log(np.random.uniform(1e-10, 1.0, logits.shape) + 1e-10) + 1e-10)
    y = _softmax_2d((logits + g) / max(temperature, 0.01))
    return y


# ════════════════════════════════════════════════════════════════
# GENERATOR
# ════════════════════════════════════════════════════════════════

class MarioGenerator:
    """
    Neural network that generates Mario level grids.

    Input:  z(64) + tier_onehot(8) = 72
    Output: (320, 11) logits -- per-cell distribution over tile types

    Architecture: 72 -> 256(LeakyReLU) -> 512(LeakyReLU) -> 3520(reshape)
    """

    def __init__(self, z_dim: int = 64, lr: float = 0.0002):
        self.z_dim = z_dim
        self.lr = lr
        inp = z_dim + 8  # 72

        self.W1 = np.random.randn(256, inp).astype(np.float32) * np.sqrt(2.0 / inp)
        self.b1 = np.zeros(256, np.float32)
        self.W2 = np.random.randn(512, 256).astype(np.float32) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros(512, np.float32)
        self.W3 = np.random.randn(OUTPUT_SIZE, 512).astype(np.float32) * np.sqrt(2.0 / 512)
        self.b3 = np.zeros(OUTPUT_SIZE, np.float32)

        self._adam = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
                      for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def forward(self, z: np.ndarray, tier_onehot: np.ndarray, temperature: float = 1.0):
        x = np.concatenate([z, tier_onehot])
        h1_pre = self.W1 @ x + self.b1
        h1 = _leaky_relu(h1_pre)
        h2_pre = self.W2 @ h1 + self.b2
        h2 = _leaky_relu(h2_pre)
        logits_flat = self.W3 @ h2 + self.b3
        logits = logits_flat.reshape(GRID_CELLS, N_CHANNELS)
        probs = _gumbel_softmax(logits, temperature)
        cache = {"x": x, "h1_pre": h1_pre, "h1": h1,
                 "h2_pre": h2_pre, "h2": h2, "logits": logits}
        return probs, cache

    def backward(self, d_probs: np.ndarray, cache: dict):
        d_logits_flat = d_probs.reshape(-1)
        dW3 = np.outer(d_logits_flat, cache["h2"])
        db3 = d_logits_flat

        dh2 = self.W3.T @ d_logits_flat
        dh2 *= _leaky_relu_grad(cache["h2_pre"])
        dW2 = np.outer(dh2, cache["h1"])
        db2 = dh2

        dh1 = self.W2.T @ dh2
        dh1 *= _leaky_relu_grad(cache["h1_pre"])
        dW1 = np.outer(dh1, cache["x"])
        db1 = dh1

        return {"W1": dW1, "b1": db1, "W2": dW2,
                "b2": db2, "W3": dW3, "b3": db3}

    def update(self, grads: dict):
        beta1, beta2, eps = 0.5, 0.999, 1e-8
        params = self._params()
        for name, p in params.items():
            adam = self._adam[name]
            adam["t"] += 1
            g = np.clip(grads[name], -1.0, 1.0)
            adam["m"] = beta1 * adam["m"] + (1 - beta1) * g
            adam["v"] = beta2 * adam["v"] + (1 - beta2) * g**2
            m_hat = adam["m"] / (1 - beta1**adam["t"])
            v_hat = adam["v"] / (1 - beta2**adam["t"])
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def generate_grid(self, tier: int = 1, temperature: float = 0.8) -> np.ndarray:
        z = np.random.randn(self.z_dim).astype(np.float32)
        tier_oh = np.zeros(8, np.float32)
        tier_oh[min(tier - 1, 7)] = 1.0
        probs, _ = self.forward(z, tier_oh, temperature)
        return probs


# ════════════════════════════════════════════════════════════════
# DISCRIMINATOR
# ════════════════════════════════════════════════════════════════

class MarioDiscriminator:
    """
    Scores level quality: high for completable + appropriately difficult.

    Input:  (320, 11) one-hot grid flattened = 3520
    Output: scalar score in [0, 1]

    Architecture: 3520 -> 512(LeakyReLU) -> 256(LeakyReLU) -> 1(sigmoid)
    """

    def __init__(self, lr: float = 0.0002):
        self.lr = lr
        inp = OUTPUT_SIZE

        self.W1 = np.random.randn(512, inp).astype(np.float32) * np.sqrt(2.0 / inp)
        self.b1 = np.zeros(512, np.float32)
        self.W2 = np.random.randn(256, 512).astype(np.float32) * np.sqrt(2.0 / 512)
        self.b2 = np.zeros(256, np.float32)
        self.W3 = np.random.randn(1, 256).astype(np.float32) * np.sqrt(2.0 / 256)
        self.b3 = np.zeros(1, np.float32)

        self._adam = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
                      for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def forward(self, grid_probs: np.ndarray):
        x = grid_probs.reshape(-1)
        h1_pre = self.W1 @ x + self.b1
        h1 = _leaky_relu(h1_pre)
        h2_pre = self.W2 @ h1 + self.b2
        h2 = _leaky_relu(h2_pre)
        out_pre = self.W3 @ h2 + self.b3
        score = _sigmoid(out_pre[0])
        cache = {"x": x, "h1_pre": h1_pre, "h1": h1,
                 "h2_pre": h2_pre, "h2": h2, "out_pre": out_pre, "score": score}
        return score, cache

    def backward(self, d_score: float, cache: dict):
        s = cache["score"]
        d_out = np.array([d_score * s * (1 - s)], np.float32)

        dW3 = np.outer(d_out, cache["h2"])
        db3 = d_out

        dh2 = self.W3.T @ d_out
        dh2 = dh2.flatten() * _leaky_relu_grad(cache["h2_pre"])
        dW2 = np.outer(dh2, cache["h1"])
        db2 = dh2

        dh1 = self.W2.T @ dh2
        dh1 *= _leaky_relu_grad(cache["h1_pre"])
        dW1 = np.outer(dh1, cache["x"])
        db1 = dh1

        d_input = self.W1.T @ dh1
        d_grid = d_input.reshape(GRID_CELLS, N_CHANNELS)

        return {"W1": dW1, "b1": db1, "W2": dW2,
                "b2": db2, "W3": dW3, "b3": db3}, d_grid

    def update(self, grads: dict):
        beta1, beta2, eps = 0.5, 0.999, 1e-8
        params = self._params()
        for name, p in params.items():
            adam = self._adam[name]
            adam["t"] += 1
            g = np.clip(grads[name], -1.0, 1.0)
            adam["m"] = beta1 * adam["m"] + (1 - beta1) * g
            adam["v"] = beta2 * adam["v"] + (1 - beta2) * g**2
            m_hat = adam["m"] / (1 - beta1**adam["t"])
            v_hat = adam["v"] / (1 - beta2**adam["t"])
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


# ════════════════════════════════════════════════════════════════
# GAN ORCHESTRATOR
# ════════════════════════════════════════════════════════════════

# Enemy types available per tier
_TIER_ENEMIES = {
    1: [],
    2: [],
    3: [EnemyType.GOOMBA],
    4: [EnemyType.GOOMBA, EnemyType.TURTLE],
    5: [EnemyType.GOOMBA, EnemyType.TURTLE, EnemyType.PIRANHA],
    6: [EnemyType.GOOMBA, EnemyType.TURTLE, EnemyType.PIRANHA, EnemyType.LAKITU],
    7: [EnemyType.GOOMBA, EnemyType.TURTLE, EnemyType.PIRANHA, EnemyType.LAKITU],
}


class MarioGAN:
    """
    Full GAN orchestrator for Mario level generation.

    Generates levels, post-processes into valid MarioSimulator instances,
    and trains Generator/Discriminator from completability results.
    """

    def __init__(self, z_dim: int = 64, lr: float = 0.0002):
        self.G = MarioGenerator(z_dim=z_dim, lr=lr)
        self.D = MarioDiscriminator(lr=lr)
        self.rng = np.random.RandomState(42)

        # Solved level bank -- generator learns to imitate these
        self.solved_bank: List[np.ndarray] = []
        self._pretrain_steps = 0

        # Training stats
        self._gen_count = 0
        self._d_losses: List[float] = []
        self._g_losses: List[float] = []

    def add_solved(self, grid_probs: np.ndarray) -> None:
        """Add a completable level grid to the imitation bank."""
        self.solved_bank.append(grid_probs.copy())
        if len(self.solved_bank) > 2000:
            self.solved_bank = self.solved_bank[-2000:]

    def pretrain_from_solved(self, epochs: int = 50, batch_size: int = 16) -> Dict[str, float]:
        """
        Supervised pre-training: generator learns to reconstruct
        completable levels (MSE loss). This teaches the generator what
        valid Mario levels look like BEFORE adversarial training.
        """
        if len(self.solved_bank) < 2:
            return {"pretrain_loss": 0.0, "steps": 0, "bank_size": len(self.solved_bank)}

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            indices = self.rng.choice(len(self.solved_bank),
                                      size=min(batch_size, len(self.solved_bank)),
                                      replace=False)
            for idx in indices:
                target = self.solved_bank[idx]
                loss = self._pretrain_step(target)
                total_loss += loss
                steps += 1
                self._pretrain_steps += 1

        avg_loss = total_loss / max(steps, 1)
        return {"pretrain_loss": float(avg_loss), "steps": steps,
                "bank_size": len(self.solved_bank)}

    def generate(self, tier: int = 1, temperature: float = 0.8) -> Optional[MarioSimulator]:
        """Generate one level. Returns MarioSimulator or None."""
        probs = self.G.generate_grid(tier, temperature)
        sim = self._postprocess(probs, tier)
        self._gen_count += 1
        return sim

    def _postprocess(self, probs: np.ndarray, tier: int) -> Optional[MarioSimulator]:
        """
        Convert (320, 11) probability grid -> valid MarioSimulator.

        Multi-stage structural validation:
          1. Enforce ground profile
          2. Fix pipe pairing (pipes must be 2-wide, resting on ground)
          3. Remove floating tiles (blocks/platforms need support)
          4. Player start + flag placement
          5. Enemy placement on valid surfaces
          6. Final structure validation
        """
        # Argmax to get discrete tiles
        grid_flat = np.argmax(probs, axis=-1)
        grid = grid_flat.reshape(GRID_H, GRID_W).astype(np.uint8)
        grid = np.clip(grid, 0, N_TILE_TYPES - 1)

        GR = MarioSimulator.GROUND_ROW  # 13

        # ── 1. Enforce ground (bottom 3 rows) ─────────────────────
        for col in range(GRID_W):
            all_pit = True
            for row in range(GR, GRID_H):
                if grid[row, col] != Tile.PIT:
                    all_pit = False
            if not all_pit:
                for row in range(GR, GRID_H):
                    grid[row, col] = Tile.GROUND

        # Ensure start/end columns always have ground (safe zones)
        for col in range(0, 4):
            grid[GR:, col] = Tile.GROUND
        for col in range(GRID_W - 3, GRID_W):
            grid[GR:, col] = Tile.GROUND

        # ── 2. Clear sky (top 4 rows) ────────────────────────────
        for row in range(0, 4):
            for col in range(GRID_W):
                if grid[row, col] not in (Tile.EMPTY, Tile.COIN):
                    grid[row, col] = Tile.EMPTY

        # ── 3. Fix pipe pairing ───────────────────────────────────
        # Pipes must be PIPE_L immediately left of PIPE_R, both
        # sitting on ground, extending upward from ground level.
        # Remove any orphaned/floating pipe tiles.
        for row in range(4, GR):
            for col in range(GRID_W):
                if grid[row, col] == Tile.PIPE_L:
                    # Must have PIPE_R to the right
                    if col + 1 < GRID_W and grid[row, col + 1] == Tile.PIPE_R:
                        # Check both columns are grounded: pipe must connect
                        # down to ground row or to another pipe tile below
                        if not self._pipe_is_grounded(grid, row, col):
                            grid[row, col] = Tile.EMPTY
                            grid[row, col + 1] = Tile.EMPTY
                    else:
                        grid[row, col] = Tile.EMPTY  # Orphan PIPE_L
                elif grid[row, col] == Tile.PIPE_R:
                    # Must have PIPE_L to the left
                    if col - 1 >= 0 and grid[row, col - 1] == Tile.PIPE_L:
                        pass  # Already handled by PIPE_L check
                    else:
                        grid[row, col] = Tile.EMPTY  # Orphan PIPE_R

        # ── 4. Clean floating tiles ────────────────────────────────
        # Remove tiles that shouldn't be floating:
        #   - Ground in the air -> remove (keep only PLATFORM/BRICK/QUESTION)
        #   - PIT in the air -> remove
        #   - Platform/brick/question only kept if within jump reach (4 rows)
        #     of ground or another solid surface below
        for row in range(0, GR):
            for col in range(GRID_W):
                tile = grid[row, col]
                if tile == Tile.GROUND:
                    grid[row, col] = Tile.EMPTY  # Ground shouldn't float
                elif tile == Tile.PIT:
                    grid[row, col] = Tile.EMPTY  # PIT shouldn't be in sky
                elif tile in (Tile.PLATFORM, Tile.BRICK, Tile.QUESTION):
                    # Keep only if reachable: solid surface within 4 rows below
                    reachable = False
                    for check_row in range(row + 1, min(row + 5, GRID_H)):
                        below = grid[check_row, col]
                        if below in (Tile.GROUND, Tile.PLATFORM, Tile.BRICK,
                                     Tile.PIPE_L, Tile.PIPE_R):
                            reachable = True
                            break
                    if not reachable:
                        grid[row, col] = Tile.EMPTY

        # ── 5. Limit pit width (max 3 tiles, jumpable) ───────────
        pit_run = 0
        for col in range(GRID_W):
            if grid[GR, col] == Tile.PIT:
                pit_run += 1
                if pit_run > 3:
                    # Too wide to jump -- fill with ground
                    grid[GR:, col] = Tile.GROUND
                    pit_run = 0
            else:
                pit_run = 0

        # ── 6. Player start ──────────────────────────────────────
        grid[grid == Tile.PLAYER] = Tile.EMPTY
        player_placed = False
        for col in range(1, 4):
            row = GR - 1
            if grid[row, col] == Tile.EMPTY and self._has_support(grid, row, col):
                grid[row, col] = Tile.PLAYER
                player_placed = True
                break
        if not player_placed:
            grid[GR - 1, 2] = Tile.PLAYER

        # ── 7. Flag at end ───────────────────────────────────────
        grid[grid == Tile.FLAG] = Tile.EMPTY
        flag_col = GRID_W - 2
        for fr in range(GR - 3, GR):
            if grid[fr, flag_col] == Tile.EMPTY:
                grid[fr, flag_col] = Tile.FLAG

        # ── 8. Enemies on valid surfaces ─────────────────────────
        enemies: List[Enemy] = []
        enemy_types = _TIER_ENEMIES.get(tier, [])
        if enemy_types:
            n_enemies = min(4, max(1, int(np.sum(grid == Tile.EMPTY)) // 40))
            placed = 0
            for _ in range(n_enemies * 3):  # Extra attempts
                ec = self.rng.randint(5, GRID_W - 3)
                er = GR - 1
                if (grid[er, ec] == Tile.EMPTY
                        and self._has_support(grid, er, ec)
                        and grid[GR, ec] != Tile.PIT):
                    etype = self.rng.choice(enemy_types)
                    enemies.append(Enemy(
                        etype=etype, row=er, col=ec,
                        direction=-1 if self.rng.random() < 0.5 else 1
                    ))
                    placed += 1
                    if placed >= n_enemies:
                        break

        # ── 9. Build simulator ───────────────────────────────────
        try:
            sim = MarioSimulator(grid, enemies)
            return sim
        except Exception:
            return None

    def _has_support(self, grid: np.ndarray, row: int, col: int) -> bool:
        """Check if there's solid ground/support below this position."""
        if row + 1 >= GRID_H:
            return False
        return grid[row + 1, col] in (Tile.GROUND, Tile.PLATFORM, Tile.BRICK,
                                       Tile.PIPE_L, Tile.PIPE_R)

    def _pipe_is_grounded(self, grid: np.ndarray, row: int, col: int) -> bool:
        """Check if a pipe tile connects down to ground level."""
        for r in range(row + 1, GRID_H):
            left = grid[r, col]
            right = grid[r, col + 1] if col + 1 < GRID_W else Tile.EMPTY
            if left == Tile.GROUND and right == Tile.GROUND:
                return True  # Reached ground row
            if left not in (Tile.PIPE_L,) or right not in (Tile.PIPE_R,):
                return False  # Gap in pipe column
        return False

    @staticmethod
    def validate_structure(grid: np.ndarray) -> Dict[str, Any]:
        """
        Multi-criteria structural validation of a level grid.

        Returns:
          score: 0.0-1.0 quality score (1.0 = perfect)
          violations: list of string descriptions of problems
          metrics: dict of per-criteria scores

        This is the "rule-based critic" -- runs before the learned
        discriminator, provides hard constraints + quality signal.
        """
        violations = []
        metrics = {}
        GR = MarioSimulator.GROUND_ROW

        # 1. Ground continuity: start and end must have ground
        has_start_ground = any(grid[GR, c] == Tile.GROUND for c in range(3))
        has_end_ground = any(grid[GR, c] == Tile.GROUND
                            for c in range(grid.shape[1] - 3, grid.shape[1]))
        if not has_start_ground:
            violations.append("no_start_ground")
        if not has_end_ground:
            violations.append("no_end_ground")
        metrics["ground_endpoints"] = 1.0 if (has_start_ground and has_end_ground) else 0.0

        # 2. Pipe pairing: every PIPE_L must have PIPE_R to its right
        orphan_pipes = 0
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if grid[row, col] == Tile.PIPE_L:
                    if col + 1 >= grid.shape[1] or grid[row, col + 1] != Tile.PIPE_R:
                        orphan_pipes += 1
                elif grid[row, col] == Tile.PIPE_R:
                    if col - 1 < 0 or grid[row, col - 1] != Tile.PIPE_L:
                        orphan_pipes += 1
        if orphan_pipes > 0:
            violations.append(f"orphan_pipes:{orphan_pipes}")
        metrics["pipe_pairing"] = 1.0 if orphan_pipes == 0 else max(0, 1.0 - orphan_pipes * 0.2)

        # 3. No ground tiles in the sky
        sky_ground = 0
        for row in range(0, GR):
            for col in range(grid.shape[1]):
                if grid[row, col] == Tile.GROUND:
                    sky_ground += 1
        if sky_ground > 0:
            violations.append(f"floating_ground:{sky_ground}")
        metrics["no_sky_ground"] = 1.0 if sky_ground == 0 else max(0, 1.0 - sky_ground * 0.1)

        # 4. Pit width: no pits wider than 3 (unjumpable)
        max_pit_run = 0
        pit_run = 0
        for col in range(grid.shape[1]):
            if grid[GR, col] == Tile.PIT:
                pit_run += 1
                max_pit_run = max(max_pit_run, pit_run)
            else:
                pit_run = 0
        if max_pit_run > 3:
            violations.append(f"pit_too_wide:{max_pit_run}")
        metrics["pit_width"] = 1.0 if max_pit_run <= 3 else 0.0

        # 5. Flag presence (player tile is consumed by MarioSimulator on init,
        #    so we only check for FLAG which persists in the grid)
        has_flag = Tile.FLAG in grid
        if not has_flag:
            violations.append("no_flag")
        metrics["player_flag"] = 1.0 if has_flag else 0.0

        # 6. Platform reachability heuristic: platforms should be
        #    within jump height (4 tiles) of ground or another platform
        unreachable = 0
        for row in range(4, GR):
            for col in range(grid.shape[1]):
                if grid[row, col] == Tile.PLATFORM:
                    # Check if reachable from below (within 4 tiles)
                    reachable = False
                    for check_row in range(row + 1, min(row + 5, GRID_H)):
                        below = grid[check_row, col]
                        if below in (Tile.GROUND, Tile.PLATFORM, Tile.BRICK,
                                     Tile.PIPE_L, Tile.PIPE_R):
                            reachable = True
                            break
                    if not reachable:
                        unreachable += 1
        if unreachable > 0:
            violations.append(f"unreachable_platforms:{unreachable}")
        total_plats = int(np.sum(grid == Tile.PLATFORM))
        metrics["platform_reach"] = 1.0 if total_plats == 0 else max(
            0, 1.0 - unreachable / max(total_plats, 1))

        # Composite score
        weights = {"ground_endpoints": 0.25, "pipe_pairing": 0.15,
                   "no_sky_ground": 0.15, "pit_width": 0.15,
                   "player_flag": 0.20, "platform_reach": 0.10}
        score = sum(metrics[k] * weights.get(k, 0.1) for k in metrics)

        return {
            "score": round(float(score), 3),
            "violations": violations,
            "metrics": metrics,
            "valid": len(violations) == 0,
        }

    def train_step(
        self,
        good_levels: List[np.ndarray],
        bad_levels: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Balanced GAN training step.

        Balance mechanisms:
          1. Loss-ratio gating: skip D training if D is already too strong
          2. 2:1 G:D step ratio
          3. Running average tracks balance
        """
        d_loss = 0.0
        g_loss = 0.0

        # Check balance
        avg_d = float(np.mean(self._d_losses[-20:])) if len(self._d_losses) >= 5 else 999.0
        avg_g = float(np.mean(self._g_losses[-20:])) if len(self._g_losses) >= 5 else 999.0
        train_d = (avg_d >= avg_g * 0.5) or len(self._d_losses) < 10

        # Train Discriminator
        if train_d:
            for grid in good_levels:
                score, cache = self.D.forward(grid)
                d_loss += -np.log(score + 1e-10)
                d_grads, _ = self.D.backward(1.0 / (score + 1e-10), cache)
                self.D.update(d_grads)

            for grid in bad_levels:
                score, cache = self.D.forward(grid)
                d_loss += -np.log(1 - score + 1e-10)
                d_grads, _ = self.D.backward(-1.0 / (1 - score + 1e-10), cache)
                self.D.update(d_grads)
        else:
            for grid in good_levels:
                score, _ = self.D.forward(grid)
                d_loss += -np.log(score + 1e-10)
            for grid in bad_levels:
                score, _ = self.D.forward(grid)
                d_loss += -np.log(1 - score + 1e-10)

        # Train Generator (2:1 ratio)
        n_gen = max(len(good_levels) * 2, 4)
        for _ in range(n_gen):
            z = np.random.randn(self.G.z_dim).astype(np.float32)
            tier_oh = np.zeros(8, np.float32)
            tier_oh[0] = 1.0

            probs, g_cache = self.G.forward(z, tier_oh, temperature=0.8)
            score, d_cache = self.D.forward(probs)

            gen_loss = -np.log(score + 1e-10)
            g_loss += gen_loss

            _, d_grid_grad = self.D.backward(1.0 / (score + 1e-10), d_cache)
            g_grads = self.G.backward(d_grid_grad, g_cache)
            self.G.update(g_grads)

        n_total = max(len(good_levels) + len(bad_levels), 1)
        d_loss /= n_total
        g_loss /= max(n_gen, 1)

        self._d_losses.append(d_loss)
        self._g_losses.append(g_loss)

        return {"d_loss": float(d_loss), "g_loss": float(g_loss),
                "d_trained": train_d, "balance": round(avg_d / max(avg_g, 0.01), 2)}

    def _pretrain_step(self, target: np.ndarray) -> float:
        """One supervised pre-training step (MSE loss)."""
        z = np.random.randn(self.G.z_dim).astype(np.float32)
        tier_oh = np.zeros(8, np.float32)
        tier_oh[0] = 1.0

        probs, cache = self.G.forward(z, tier_oh, temperature=1.0)

        diff = probs - target
        loss = float(np.mean(diff ** 2))
        d_output = 2.0 * diff / diff.size

        grads = self.G.backward(d_output, cache)
        self.G.update(grads)
        return loss

    def grid_to_onehot(self, sim: MarioSimulator) -> np.ndarray:
        """Convert a MarioSimulator grid to (320, 11) one-hot for the GAN."""
        # Take the first 20 columns (1 screen)
        w = min(sim.width, GRID_W)
        grid = sim.grid[:, :w].copy()

        onehot = np.zeros((GRID_CELLS, N_CHANNELS), dtype=np.float32)
        flat = grid.flatten()
        for i in range(min(len(flat), GRID_CELLS)):
            tile_idx = min(int(flat[i]), N_CHANNELS - 1)
            onehot[i, tile_idx] = 1.0
        return onehot

    def report(self) -> Dict[str, Any]:
        return {
            "generated": self._gen_count,
            "solved_bank": len(self.solved_bank),
            "pretrain_steps": self._pretrain_steps,
            "d_loss_avg": float(np.mean(self._d_losses[-50:])) if self._d_losses else 0,
            "g_loss_avg": float(np.mean(self._g_losses[-50:])) if self._g_losses else 0,
            "g_params": sum(p.size for p in self.G._params().values()),
            "d_params": sum(p.size for p in self.D._params().values()),
        }
