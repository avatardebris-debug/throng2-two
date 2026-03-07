"""Quick test of the structural validator."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.games.mario.mario_simulator import MarioSimulator, Tile
from src.games.mario.mario_generator import MarioLevelGenerator
from src.games.mario.mario_gan import MarioGAN
import numpy as np

print("== Structural Validator Test ==")

# 1. Procedural levels (should be valid)
gen = MarioLevelGenerator(seed=42)
for tier in range(1, 4):
    level = gen.generate(tier=tier)
    if level:
        v = MarioGAN.validate_structure(level.grid)
        status = "PASS" if v["valid"] else "WARN"
        print(f"[{status}] Tier {tier} procedural: score={v['score']}, violations={v['violations']}")

# 2. Deliberately broken level
print()
broken = np.full((16, 20), Tile.EMPTY, dtype=np.uint8)
broken[5, 8] = Tile.PIPE_L   # Orphan pipe
broken[7, 3] = Tile.GROUND   # Floating ground
v = MarioGAN.validate_structure(broken)
print(f"[INFO] Broken level: score={v['score']}, violations={v['violations']}")

# 3. GAN postprocessor
print()
gan = MarioGAN()
for _ in range(10):
    level = gen.generate(tier=1)
    if level:
        gan.add_solved(gan.grid_to_onehot(level))
gan.pretrain_from_solved(epochs=5, batch_size=4)

fixed_count = 0
for i in range(5):
    sim = gan.generate(tier=1)
    if sim:
        v = MarioGAN.validate_structure(sim.grid)
        status = "PASS" if v["valid"] else "WARN"
        if v["valid"]:
            fixed_count += 1
        viols = str(v["violations"][:3])
        print(f"  GAN gen {i}: score={v['score']}, valid={v['valid']}, violations={viols}")

print(f"\n[RESULT] {fixed_count}/5 GAN levels passed structural validation")

# 4. Show a GAN level
sim = gan.generate(tier=1)
if sim:
    print(f"\nGAN level (post-validation):")
    print(sim.render_ascii(viewport=False))
    v = MarioGAN.validate_structure(sim.grid)
    print(f"Score: {v['score']}, Valid: {v['valid']}, Violations: {v['violations']}")
