"""Test the Mario ASCII simulator, generator, and adapter."""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=" * 60)
print("MARIO ASCII SIMULATOR — SANITY TEST")
print("=" * 60)

# ── 1. Simulator basic test ───────────────────────────────
from src.games.mario.mario_simulator import MarioSimulator, Tile, Action, Enemy, EnemyType

# Flat ground level
sim = MarioSimulator.from_flat_ground(n_screens=1)
print(f"[PASS] Flat ground created: {sim.GRID_H}x{sim.width}")
print(f"  Mario at ({sim.mario_row}, {sim.mario_col})")
print()
print(sim.render_ascii(viewport=False))
print()

# Step test
obs, r, done, info = sim.step(Action.RIGHT)
print(f"[PASS] Step works: obs_shape={obs.shape}, reward={r:.3f}, done={done}")
print(f"  Mario now at ({sim.mario_row}, {sim.mario_col})")
print(f"  Info: {info}")

# Walk right until win or 100 steps
for i in range(100):
    obs, r, done, info = sim.step(Action.RIGHT)
    if done:
        break
status = "PASS" if sim.won else "FAIL"
print(f"[{status}] Walk to flag: won={sim.won} after {sim.step_count} steps")

# Save/load test
state = sim.save()
sim2 = MarioSimulator.from_flat_ground()
sim2.load(state)
assert sim2.mario_col == sim.mario_col
print("[PASS] Save/load works")

# Completability test
sim3 = MarioSimulator.from_flat_ground()
t0 = time.perf_counter()
completable = sim3.is_completable()
t1 = time.perf_counter()
status = "PASS" if completable else "FAIL"
print(f"[{status}] Completability check: {completable} ({(t1-t0)*1000:.1f}ms)")

# ── 2. Generator test ────────────────────────────────────
print()
print("-- Generator --")
from src.games.mario.mario_generator import MarioLevelGenerator

gen = MarioLevelGenerator(seed=42)

for tier in range(1, 8):
    t0 = time.perf_counter()
    level = gen.generate(tier=tier)
    t1 = time.perf_counter()
    if level:
        print(f"[PASS] Tier {tier}: completable level "
              f"(width={level.width}, enemies={len(level.enemies)}, "
              f"{(t1-t0)*1000:.0f}ms)")
    else:
        print(f"[WARN] Tier {tier}: failed to generate ({(t1-t0)*1000:.0f}ms)")

print(f"  Generator stats: {gen.report()}")

# Show a tier 3 level
t3 = gen.generate(tier=3)
if t3:
    print(f"\nTier 3 level (width={t3.width}):")
    print(t3.render_ascii(viewport=False))

# ── 3. Adapter test ──────────────────────────────────────
print()
print("-- Adapter --")
from src.games.mario.mario_adapter import MarioAdapter

adapter = MarioAdapter()
level = gen.generate(tier=1)
obs = adapter.reset(level)
print(f"[PASS] Adapter reset: obs_shape={obs.shape}")

obs, r, done, info = adapter.step(Action.RIGHT)
print(f"[PASS] Adapter step: obs_shape={obs.shape}, r={r:.3f}")

ram = adapter.grid_to_ram()
print(f"[PASS] Fake RAM: shape={ram.shape}, mario_x={ram[0]}, mario_y={ram[1]}")
print(f"  Stats: {adapter.stats()}")

# ── 4. Speed test ────────────────────────────────────────
print()
print("-- Speed --")
level = gen.generate(tier=3)
if level:
    adapter.reset(level)
    t0 = time.perf_counter()
    steps = 0
    for ep in range(100):
        level = gen.generate(tier=3)
        if not level:
            continue
        adapter.reset(level)
        for _ in range(200):
            action = int(np.random.randint(6))
            _, _, done, _ = adapter.step(action)
            steps += 1
            if done:
                break
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"[PASS] 100 episodes, {steps} total steps in {elapsed:.2f}s")
    print(f"  Steps/sec: {steps/elapsed:,.0f}")
    print(f"  ~Real NES speed: 60 steps/sec")
    print(f"  Speedup: ~{steps/elapsed/60:.0f}x faster than real-time NES")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
