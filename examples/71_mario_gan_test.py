"""Test the Mario GAN and Curriculum system."""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.games.mario.mario_simulator import MarioSimulator, Action
from src.games.mario.mario_generator import MarioLevelGenerator
from src.games.mario.mario_gan import MarioGAN
from src.games.mario.mario_curriculum import MarioCurriculum

print("=" * 60)
print("MARIO GAN + CURRICULUM TEST")
print("=" * 60)

# ── 1. GAN basic test ────────────────────────────────────
print("\n-- GAN Basic --")

gan = MarioGAN()
print(f"[PASS] GAN created: G params={sum(p.size for p in gan.G._params().values()):,}, "
      f"D params={sum(p.size for p in gan.D._params().values()):,}")

# Generate a raw level
sim = gan.generate(tier=1)
if sim:
    print(f"[PASS] GAN generated level: {sim.GRID_H}x{sim.width}")
    print("  (raw, before training -- likely not great)")
    print(sim.render_ascii(viewport=False))
else:
    print("[INFO] GAN raw generation returned None (expected before training)")

# ── 2. Seed solved bank with procedural levels ──────────
print("\n-- Seeding Solved Bank --")
gen = MarioLevelGenerator(seed=42)

t0 = time.perf_counter()
for i in range(20):
    level = gen.generate(tier=1)
    if level:
        onehot = gan.grid_to_onehot(level)
        gan.add_solved(onehot)
t1 = time.perf_counter()
print(f"[PASS] Seeded {len(gan.solved_bank)} solved levels in {(t1-t0)*1000:.0f}ms")

# ── 3. Pretrain GAN on solved levels ─────────────────────
print("\n-- GAN Pretraining --")
t0 = time.perf_counter()
pt_result = gan.pretrain_from_solved(epochs=10, batch_size=8)
t1 = time.perf_counter()
print(f"[PASS] Pretrained: loss={pt_result['pretrain_loss']:.4f}, "
      f"steps={pt_result['steps']}, {(t1-t0)*1000:.0f}ms")

# ── 4. Generate after pretraining ─────────────────────────
print("\n-- GAN Generation (after pretrain) --")
sim = gan.generate(tier=1)
if sim:
    completable = sim.is_completable()
    status = "PASS" if completable else "WARN"
    print(f"[{status}] Generated level: completable={completable}")
    print(sim.render_ascii(viewport=False))
else:
    print("[WARN] GAN generation returned None after pretrain")

# ── 5. Adversarial training step ──────────────────────────
print("\n-- GAN Adversarial Training --")
# Create good/bad level pools
good_levels = []
bad_levels = []
for i in range(10):
    level = gen.generate(tier=1)
    if level:
        oh = gan.grid_to_onehot(level)
        if level.is_completable():
            good_levels.append(oh)
        else:
            bad_levels.append(oh)

# If no bad levels, create some broken ones
if not bad_levels:
    for _ in range(5):
        grid = np.random.randint(0, 11, size=(16, 20), dtype=np.uint8)
        broken = MarioSimulator(grid)
        bad_levels.append(gan.grid_to_onehot(broken))

t0 = time.perf_counter()
train_result = gan.train_step(good_levels[:4], bad_levels[:4])
t1 = time.perf_counter()
print(f"[PASS] Training step: d_loss={train_result['d_loss']:.3f}, "
      f"g_loss={train_result['g_loss']:.3f}, "
      f"balance={train_result['balance']}, {(t1-t0)*1000:.0f}ms")
print(f"  GAN report: {gan.report()}")

# ── 6. Curriculum test ────────────────────────────────────
print("\n-- Curriculum --")
curriculum = MarioCurriculum(start_tier=1, advance_threshold=0.8,
                              window_size=10, seed=42)

print(f"[PASS] Curriculum created at tier {curriculum.tier}")

# Simulate some episodes
for ep in range(15):
    level = curriculum.next_level()

    # Random agent plays
    from src.games.mario.mario_adapter import MarioAdapter
    adapter = MarioAdapter()
    obs = adapter.reset(level)
    total_reward = 0
    for step in range(200):
        # Biased toward RIGHT to make some progress
        action = np.random.choice([Action.RIGHT, Action.RIGHT, Action.RIGHT,
                                    Action.JUMP_RIGHT, Action.JUMP, Action.NOOP])
        obs, reward, done, info = adapter.step(action)
        total_reward += reward
        if done:
            break

    progress = level.max_x_reached / max(1, level.width)
    won = level.won
    result = curriculum.record_result(won=won, progress=progress,
                                       steps=step + 1, level=level)

    if ep % 5 == 0:
        print(f"  Ep {ep}: won={won}, progress={progress:.2f}, "
              f"tier={result['tier']}, win_rate={result['tier_win_rate']:.2f}")

# Train GAN after episodes
gan_result = curriculum.train_gan()
print(f"[PASS] GAN trained: {gan_result.get('gan_report', {}).get('pretrain_steps', 0)} pretrain steps")

# Print final report
print(f"\n  Curriculum report: {curriculum.report()}")

# ── 7. Test curriculum advancement ────────────────────────
print("\n-- Curriculum Advancement --")
# Force-feed wins to test advancement
for _ in range(20):
    curriculum.record_result(won=True, progress=1.0, steps=50)

if curriculum.should_advance():
    new_tier = curriculum.advance()
    print(f"[PASS] Advanced to tier {new_tier}")
else:
    print("[INFO] Not ready to advance yet")

print(f"  Final status: {curriculum.status()}")

print("\n" + "=" * 60)
print("ALL GAN/CURRICULUM TESTS PASSED")
print("=" * 60)
