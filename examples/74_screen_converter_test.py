"""
74_screen_converter_test.py -- Test the game-to-ASCII converter.

Verifies:
  1. ASCII -> pixel image rendering
  2. Image -> ASCII roundtrip accuracy
  3. Mario + enemy detection
  4. Multiple level tiers (different tile compositions)
  5. Full sim-to-real pipeline: screenshot -> grid -> simulator -> agent action
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.games.mario.mario_simulator import MarioSimulator, Tile, Action
from src.games.mario.mario_generator import MarioLevelGenerator
from src.games.mario.screen_to_ascii import (
    ascii_to_image, image_to_ascii, grid_to_simulator,
    roundtrip_test, save_image,
)


def test_basic_roundtrip():
    """Test roundtrip on flat ground level."""
    print("=" * 60)
    print("  TEST 1: Basic Roundtrip (flat ground)")
    print("=" * 60)

    sim = MarioSimulator.from_flat_ground(n_screens=1)
    result = roundtrip_test(sim)

    print(f"  Image shape: {result['image_shape']}")
    print(f"  Total tiles: {result['total_tiles']}")
    print(f"  Matching: {result['matching']}")
    print(f"  Accuracy: {result['accuracy']:.1%}")
    print(f"  Mario detected: {result['mario_detected']}")
    print(f"  Mario correct: {result['mario_correct']}")
    print(f"  Mario actual: {result['mario_pos_actual']}")
    print(f"  Mario found:  {result['mario_pos_detected']}")
    print()
    print("  Per-tile accuracy:")
    for char, stats in result["tile_accuracy"].items():
        print(f"    '{char}': {stats['correct']}/{stats['total']} = {stats['accuracy']:.0%}")
    print()

    assert result["accuracy"] > 0.8, f"Roundtrip accuracy too low: {result['accuracy']}"
    assert result["mario_detected"], "Mario not detected!"
    print("  PASSED")
    return result


def test_generated_levels():
    """Test roundtrip on procedurally generated levels."""
    print()
    print("=" * 60)
    print("  TEST 2: Generated Level Roundtrip")
    print("=" * 60)

    gen = MarioLevelGenerator(seed=42)
    accuracies = []

    for tier in [1, 3, 5]:
        level = gen.generate(tier=tier)
        result = roundtrip_test(level)
        accuracies.append(result["accuracy"])
        print(f"  Tier {tier}: accuracy={result['accuracy']:.1%}, "
              f"mario={'OK' if result['mario_correct'] else 'MISS'}, "
              f"enemies={result['enemies_detected']}")

    avg = np.mean(accuracies)
    print(f"\n  Average accuracy: {avg:.1%}")
    assert avg > 0.7, f"Average roundtrip accuracy too low: {avg}"
    print("  PASSED")


def test_entity_detection():
    """Test Mario and enemy detection specifically."""
    print()
    print("=" * 60)
    print("  TEST 3: Entity Detection")
    print("=" * 60)

    gen = MarioLevelGenerator(seed=123)
    # Generate a level with enemies
    level = gen.generate(tier=5)

    # Render to image
    img = ascii_to_image(level, tile_size=16, viewport=False)

    # Convert back
    grid, entities = image_to_ascii(img, grid_rows=16, grid_cols=level.width)

    print(f"  Level size: {level.GRID_H}x{level.width}")
    print(f"  Original Mario: ({level.mario_row}, {level.mario_col})")
    print(f"  Detected Mario: {entities['mario_pos']}")
    print(f"  Original enemies: {len(level.enemies)}")
    print(f"  Detected enemies: {len(entities['enemies'])}")
    for e in entities["enemies"][:5]:
        print(f"    {e['type']} at ({e['row']}, {e['col']})")

    assert entities["mario_pos"] is not None, "Mario not detected!"
    print("  PASSED")


def test_sim_to_real_pipeline():
    """Test the full sim-to-real pipeline."""
    print()
    print("=" * 60)
    print("  TEST 4: Sim-to-Real Pipeline")
    print("=" * 60)

    # Simulate: "real" game produces a screenshot
    real_sim = MarioSimulator.from_flat_ground(n_screens=1)
    for _ in range(5):
        real_sim.step(Action.RIGHT)

    # Step 1: Screenshot (image from "real" game)
    screenshot = ascii_to_image(real_sim, tile_size=16, viewport=True)
    print(f"  Screenshot shape: {screenshot.shape}")

    # Step 2: Convert to ASCII grid
    grid, entities = image_to_ascii(screenshot, grid_rows=16, grid_cols=20)
    print(f"  Grid shape: {grid.shape}")
    print(f"  Mario detected at: {entities['mario_pos']}")

    # Step 3: Build simulator from detected grid
    reconstructed_sim = grid_to_simulator(grid, entities)
    print(f"  Reconstructed sim: {reconstructed_sim.width} wide, "
          f"Mario at ({reconstructed_sim.mario_row}, {reconstructed_sim.mario_col})")

    # Step 4: Agent can use the reconstructed sim
    obs = reconstructed_sim.get_obs()
    print(f"  Observation vector: {obs.shape} ({obs.shape[0]} features)")

    # Step 5: Take an action
    _, reward, done, info = reconstructed_sim.step(Action.RIGHT)
    print(f"  Action RIGHT -> reward={reward:.2f}, done={done}")

    print()
    print("  Pipeline: screenshot -> ASCII -> simulator -> action")
    print("  PASSED")


def test_image_save():
    """Test PPM image saving."""
    print()
    print("=" * 60)
    print("  TEST 5: Image Export")
    print("=" * 60)

    sim = MarioSimulator.from_flat_ground(n_screens=1)
    # Step forward a few times
    for _ in range(3):
        sim.step(Action.RIGHT)

    img = ascii_to_image(sim, tile_size=16, viewport=True)
    out_dir = os.path.dirname(__file__)
    ppm_path = os.path.join(out_dir, "mario_screenshot.ppm")
    save_image(img, ppm_path)
    print(f"  Saved: {ppm_path} ({os.path.getsize(ppm_path)} bytes)")

    # Also save the full level
    full_img = ascii_to_image(sim, tile_size=16, viewport=False)
    full_path = os.path.join(out_dir, "mario_full_level.ppm")
    save_image(full_img, full_path)
    print(f"  Saved: {full_path} ({os.path.getsize(full_path)} bytes)")

    print("  PASSED")


def main():
    print()
    print("  MARIO SCREEN-TO-ASCII CONVERTER -- TEST SUITE")
    print()

    t0 = time.perf_counter()

    r1 = test_basic_roundtrip()
    test_generated_levels()
    test_entity_detection()
    test_sim_to_real_pipeline()
    test_image_save()

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print(f"  ALL TESTS PASSED ({elapsed:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
