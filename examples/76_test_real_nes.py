"""Test screen_to_ascii on real NES FCEUX screenshot."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# We need to read PNG without PIL — let's check if PIL is available
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.games.mario.mario_simulator import Tile, TILE_CHAR
from src.games.mario.screen_to_ascii import (
    image_to_ascii, grid_to_simulator, COLORS, _color_distance,
)

SCREENSHOT = r"C:\Users\avata\.gemini\antigravity\brain\cfa935ac-79cb-4816-93a1-af54eb4ca425\media__1772846880281.png"


def load_image(path):
    """Load PNG as numpy RGB array."""
    if HAS_PIL:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    else:
        # Try using matplotlib
        import matplotlib.image as mpimg
        img = mpimg.imread(path)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        return img


def main():
    print("=" * 60)
    print("  REAL NES SCREENSHOT -> ASCII CONVERSION TEST")
    print("=" * 60)

    # Load screenshot
    img = load_image(SCREENSHOT)
    print(f"  Image size: {img.shape} (H={img.shape[0]}, W={img.shape[1]})")

    # NES native: 256x240. FCEUX may scale.
    # Our converter expects to divide into 16 rows x 20 cols
    # But NES also has a HUD at top (first 2 tile rows usually)
    h, w = img.shape[:2]
    print(f"  Aspect ratio: {w/h:.3f} (NES native=1.067)")

    # Sample some pixel colors to see what we're dealing with
    print()
    print("  Sample pixel colors (for palette calibration):")
    # Sky area (top-center)
    sky_px = img[40, w//2]
    print(f"    Sky:    RGB={tuple(sky_px)}")
    # Ground area (bottom row)
    gnd_px = img[h-20, w//4]
    print(f"    Ground: RGB={tuple(gnd_px)}")
    # Mario area (should be reddish)
    print(f"    Top-left: RGB={tuple(img[10, 10])}")
    print(f"    Center:   RGB={tuple(img[h//2, w//2])}")

    # Check distances to our palette
    print()
    print("  Sky pixel distance to our palette 'sky_blue':")
    print(f"    dist = {_color_distance(tuple(sky_px), COLORS['sky_blue']):.1f}")
    print(f"    Our sky_blue = {COLORS['sky_blue']}")
    print()
    print("  Ground pixel distance to our palette 'ground_brown':")
    print(f"    dist = {_color_distance(tuple(gnd_px), COLORS['ground_brown']):.1f}")
    print(f"    Our ground_brown = {COLORS['ground_brown']}")

    # Run the converter
    # NES screen: first ~2 tile rows are HUD (MARIO, WORLD, TIME text)
    # Crop to game area: skip top 32px (2 tile rows)
    # NES visible: 256x240, HUD takes ~32px at top
    hud_height = int(h * 32 / 240)  # Scale HUD height with image
    game_img = img[hud_height:, :]
    print(f"\n  Cropped to game area: {game_img.shape}")

    # Convert: 14 tile rows (16 minus 2 HUD), 16 columns visible
    # Actually NES shows ~20 tile columns in viewport
    grid, entities = image_to_ascii(game_img, grid_rows=14, grid_cols=16)

    print(f"\n  Detected grid ({grid.shape[0]}x{grid.shape[1]}):")
    for row in range(grid.shape[0]):
        line = ""
        for col in range(grid.shape[1]):
            tile = grid[row, col]
            ch = TILE_CHAR.get(tile, '?')
            line += ch
        print(f"    {line}")

    print(f"\n  Mario detected at: {entities['mario_pos']}")
    print(f"  Enemies detected: {len(entities['enemies'])}")
    for e in entities['enemies']:
        print(f"    {e['type']} at ({e['row']}, {e['col']})")

    # Count tile types
    print("\n  Tile distribution:")
    from collections import Counter
    flat = grid.flatten()
    counts = Counter(flat)
    for tile_val, count in sorted(counts.items()):
        ch = TILE_CHAR.get(tile_val, '?')
        pct = count / len(flat) * 100
        print(f"    '{ch}' (Tile={tile_val}): {count} ({pct:.0f}%)")

    print("\n  === Color palette distances for key areas ===")
    # Sample tile-sized regions and show mean color + closest match
    tile_h = game_img.shape[0] // 14
    tile_w = game_img.shape[1] // 16
    for label, row, col in [("Sky", 2, 8), ("Ground", 12, 4), ("Pipe?", 7, 5), ("Brick?", 8, 0)]:
        y0, y1 = row * tile_h, (row+1) * tile_h
        x0, x1 = col * tile_w, (col+1) * tile_w
        if y1 <= game_img.shape[0] and x1 <= game_img.shape[1]:
            region = game_img[y0:y1, x0:x1]
            mean = tuple(int(c) for c in region.mean(axis=(0,1)))
            print(f"    {label:8s} tile({row},{col}): mean RGB={mean}")
            # Find closest palette entry
            best_name, best_dist = "", 9999
            for name, ref in COLORS.items():
                d = _color_distance(mean, ref)
                if d < best_dist:
                    best_dist = d
                    best_name = name
            print(f"             closest: {best_name} ({COLORS[best_name]}) dist={best_dist:.1f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
