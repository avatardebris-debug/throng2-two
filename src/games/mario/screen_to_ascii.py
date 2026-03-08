"""
screen_to_ascii.py -- Convert game screenshots to ASCII tile grids.

The sim-to-real bridge: takes pixel images (from NES emulator or
screenshots) and outputs the same ASCII tile grid format used by
MarioSimulator. This enables:

  1. Train agent on fast ASCII simulator
  2. At test time: screenshot -> ASCII -> agent picks action -> send to real game

NES Mario Bros screen: 256x240 pixels, 16x16 pixel tiles
Our grid: 16 rows x 20 columns (visible viewport)

Tile classification uses dominant color matching against known
NES color palettes. Also provides:
  - ascii_to_image(): render ASCII grid as pixel image (for self-test)
  - roundtrip_test(): ASCII -> image -> ASCII (verify converter works)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.games.mario.mario_simulator import (
    MarioSimulator, Tile, TILE_CHAR, Enemy, EnemyType,
)


# ═══════════════════════════════════════════════════════════════
# NES MARIO COLOR PALETTES (RGB)
# ═══════════════════════════════════════════════════════════════

# Standard NES Super Mario Bros color values (approximate sRGB)
COLORS = {
    # Sky / background
    "sky_blue":     (92, 148, 252),
    "sky_black":    (0, 0, 0),       # Underground/night levels

    # Ground / bricks
    "ground_brown": (200, 76, 12),    # Ground blocks
    "brick_orange": (228, 92, 16),    # Breakable bricks
    "brick_dark":   (172, 52, 0),     # Brick shadow

    # Question blocks
    "question_gold":(252, 188, 60),   # ? block
    "question_dark":(228, 148, 24),   # ? block shadow

    # Pipes
    "pipe_green":   (0, 168, 0),      # Pipe body
    "pipe_light":   (88, 216, 84),    # Pipe highlight
    "pipe_dark":    (0, 120, 0),      # Pipe shadow

    # Coins
    "coin_yellow":  (252, 224, 120),  # Coin sparkle
    "coin_orange":  (252, 188, 60),   # Coin body

    # Mario
    "mario_red":    (228, 0, 0),      # Mario's hat/shirt
    "mario_skin":   (252, 188, 148),  # Mario's face/hands
    "mario_brown":  (172, 52, 0),     # Mario's hair

    # Enemies
    "goomba_brown": (172, 100, 48),   # Goomba body
    "turtle_green": (0, 168, 0),      # Turtle shell
    "piranha_red":  (228, 0, 0),      # Piranha plant

    # Flag
    "flag_green":   (0, 168, 68),     # Flag
    "flag_white":   (252, 252, 252),  # Flagpole
    "flag_pole":    (172, 172, 172),  # Pole

    # Background scenery (NOT gameplay tiles — must be filtered OUT)
    # NES Mario uses palette-swapped versions of clouds and bushes.
    # Clouds appear in sky rows and use light blue / white.
    # Bushes appear near ground and use a distinct green that is NOT pipe green.
    "cloud_white":  (252, 252, 252),  # Cloud body (very bright white)
    "cloud_light":  (120, 200, 252),  # Cloud highlight (light cyan)
    "bush_green":   (0, 104, 0),      # Bush dark body (darker than pipe)
    "bush_light":   (88, 176, 0),     # Bush highlight (yellow-green)
    "bush_mid":     (0, 168, 40),     # Bush mid-tone (slightly off from pipe)

    # Platform (gameplay — NOT scenery)
    "platform_gray":(168, 168, 168),  # Moving platforms
}

# Colors that indicate BACKGROUND SCENERY (ignore, return EMPTY)
BACKGROUND_COLORS = [
    COLORS["cloud_white"],
    COLORS["cloud_light"],
    COLORS["bush_green"],
    COLORS["bush_light"],
    COLORS["bush_mid"],
    COLORS["flag_white"],   # Flagpole whites also appear in clouds
]
BACKGROUND_THRESHOLD = 50.0  # Tight match required to call it scenery

# Tile type -> characteristic RGB colors (for matching)
TILE_PALETTE = {
    Tile.EMPTY:    [COLORS["sky_blue"], COLORS["sky_black"]],
    Tile.GROUND:   [COLORS["ground_brown"], COLORS["brick_dark"]],
    Tile.BRICK:    [COLORS["brick_orange"], COLORS["brick_dark"]],
    Tile.QUESTION: [COLORS["question_gold"], COLORS["question_dark"]],
    Tile.PIPE_L:   [COLORS["pipe_green"], COLORS["pipe_dark"]],
    Tile.PIPE_R:   [COLORS["pipe_green"], COLORS["pipe_light"]],
    Tile.COIN:     [COLORS["coin_yellow"], COLORS["coin_orange"]],
    Tile.PIT:      [COLORS["sky_black"]],
    Tile.PLATFORM: [COLORS["platform_gray"]],
    Tile.FLAG:     [COLORS["flag_green"], COLORS["flag_white"]],
}

# Entity colors (for Mario/enemy detection)
ENTITY_COLORS = {
    "mario":    [COLORS["mario_red"], COLORS["mario_skin"]],
    "goomba":   [COLORS["goomba_brown"]],
    "turtle":   [COLORS["turtle_green"]],
    "piranha":  [COLORS["piranha_red"]],
}


# ═══════════════════════════════════════════════════════════════
# ASCII TO IMAGE (for self-testing)
# ═══════════════════════════════════════════════════════════════

# Simplified tile colors for rendering (RGB)
# Use same colors as TILE_PALETTE so roundtrip works
TILE_RENDER_COLOR = {
    Tile.EMPTY:    COLORS["sky_blue"],      # (92, 148, 252)
    Tile.GROUND:   COLORS["ground_brown"],  # (200, 76, 12)
    Tile.BRICK:    COLORS["brick_orange"],  # (228, 92, 16)
    Tile.QUESTION: COLORS["question_gold"], # (252, 188, 60)
    Tile.PIPE_L:   COLORS["pipe_dark"],     # (0, 120, 0)
    Tile.PIPE_R:   COLORS["pipe_light"],    # (88, 216, 84)
    Tile.COIN:     COLORS["coin_yellow"],   # (252, 224, 120)
    Tile.PIT:      (20, 0, 0),              # Near black
    Tile.PLATFORM: COLORS["platform_gray"], # (168, 168, 168)
    Tile.FLAG:     COLORS["flag_green"],    # (0, 168, 68)
    Tile.PLAYER:   COLORS["mario_red"],     # (228, 0, 0)
}

ENTITY_RENDER_COLOR = {
    "mario":   (255, 0, 0),           # Red
    "goomba":  (165, 82, 40),         # Brown
    "turtle":  (0, 180, 0),           # Green
    "piranha": (255, 50, 50),         # Red
    "lakitu":  (200, 200, 255),       # Light blue
}


def ascii_to_image(
    sim: MarioSimulator,
    tile_size: int = 16,
    viewport: bool = True,
) -> np.ndarray:
    """
    Render a MarioSimulator as a pixel image (for self-testing).

    Args:
        sim: MarioSimulator instance
        tile_size: pixels per tile (16 = NES native)
        viewport: if True, render 20-column viewport; else full level

    Returns:
        (H, W, 3) uint8 RGB image
    """
    if viewport:
        vp_start = max(0, min(sim.mario_col - 10, sim.width - 20))
        vp_end = min(vp_start + 20, sim.width)
    else:
        vp_start = 0
        vp_end = sim.width

    cols = vp_end - vp_start
    rows = MarioSimulator.GRID_H

    img = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)

    # Draw tiles
    for row in range(rows):
        for col_idx in range(cols):
            col = vp_start + col_idx
            tile = sim.grid[row, col]
            color = TILE_RENDER_COLOR.get(tile, (92, 148, 252))

            y0 = row * tile_size
            y1 = y0 + tile_size
            x0 = col_idx * tile_size
            x1 = x0 + tile_size

            img[y0:y1, x0:x1] = color

            # Add tile borders for ground/brick/question
            if tile in (Tile.GROUND, Tile.BRICK, Tile.QUESTION):
                # Darker border on bottom and right
                border_color = tuple(max(0, c - 40) for c in color)
                img[y1-1, x0:x1] = border_color
                img[y0:y1, x1-1] = border_color
                # Lighter top-left highlight
                highlight = tuple(min(255, c + 40) for c in color)
                img[y0, x0:x1] = highlight
                img[y0:y1, x0] = highlight

    # Draw Mario
    if vp_start <= sim.mario_col < vp_end:
        mx = (sim.mario_col - vp_start) * tile_size
        my = sim.mario_row * tile_size
        mario_color = ENTITY_RENDER_COLOR["mario"]
        # Mario is a filled tile with skin+red
        img[my+1:my+tile_size-1, mx+1:mx+tile_size-1] = mario_color
        # Skin face area (top portion)
        img[my+2:my+6, mx+4:mx+12] = (252, 188, 148)

    # Draw enemies
    for e in sim.enemies:
        if e.alive and vp_start <= e.col < vp_end:
            ex = (e.col - vp_start) * tile_size
            ey = e.row * tile_size
            if e.etype == EnemyType.GOOMBA:
                ecolor = ENTITY_RENDER_COLOR["goomba"]
            elif e.etype == EnemyType.TURTLE:
                ecolor = ENTITY_RENDER_COLOR["turtle"]
            elif e.etype == EnemyType.PIRANHA:
                ecolor = ENTITY_RENDER_COLOR["piranha"]
            else:
                ecolor = ENTITY_RENDER_COLOR["lakitu"]
            img[ey+2:ey+tile_size-2, ex+2:ex+tile_size-2] = ecolor

    return img


# ═══════════════════════════════════════════════════════════════
# IMAGE TO ASCII (the core converter)
# ═══════════════════════════════════════════════════════════════

def _color_distance(c1: Tuple[int, ...], c2: Tuple[int, ...]) -> float:
    """Euclidean distance between two RGB colors."""
    return sum((int(a) - int(b)) ** 2 for a, b in zip(c1, c2)) ** 0.5


def _classify_tile_by_color(
    tile_pixels: np.ndarray,
    threshold: float = 60.0,
    row: int = -1,
) -> Tile:
    """
    Classify a tile region by its dominant color.

    Uses center 50% of tile pixels to avoid border/highlight artifacts.
    Background scenery (clouds, bushes) is explicitly filtered to EMPTY.
    """
    h, w = tile_pixels.shape[:2]
    margin_h = max(1, h // 4)
    margin_w = max(1, w // 4)
    center = tile_pixels[margin_h:h-margin_h, margin_w:w-margin_w]
    if center.size == 0:
        center = tile_pixels

    mean_color = tuple(int(c) for c in center.mean(axis=(0, 1)))
    r, g, b = mean_color

    # Sky fast-path: high blue, low red
    if b > 180 and r < 120 and g < 180:
        return Tile.EMPTY

    # Very bright white → cloud scenery, not a gameplay tile
    if r > 230 and g > 230 and b > 230:
        return Tile.EMPTY

    # Background scenery check: match against known background colors
    for bg_color in BACKGROUND_COLORS:
        if _color_distance(mean_color, bg_color) < BACKGROUND_THRESHOLD:
            return Tile.EMPTY

    # Row-based background zone hints:
    # Rows 1-4 are sky/cloud zone — pipes don't appear here in W1-1
    if 0 < row <= 4:
        # In the sky rows, green almost certainly means bush or scenery
        if g > 100 and r < 80:
            return Tile.EMPTY

    best_tile = Tile.EMPTY
    best_dist = float("inf")

    for tile_type, ref_colors in TILE_PALETTE.items():
        for ref_color in ref_colors:
            dist = _color_distance(mean_color, ref_color)
            if dist < best_dist:
                best_dist = dist
                best_tile = tile_type

    if best_dist > threshold:
        return Tile.EMPTY

    return best_tile


def image_to_ascii(
    image: np.ndarray,
    grid_rows: int = 16,
    grid_cols: int = 20,
    detect_entities: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Convert a game screenshot to an ASCII tile grid.

    Args:
        image: (H, W, 3) uint8 RGB image
        grid_rows: number of tile rows (NES = 16)
        grid_cols: number of tile columns (NES viewport = 20)
        detect_entities: if True, also detect Mario and enemy positions

    Returns:
        grid: (grid_rows, grid_cols) uint8 array of Tile values
        entities: dict with 'mario_pos', 'enemies' lists
    """
    h, w = image.shape[:2]
    tile_h = h // grid_rows
    tile_w = w // grid_cols

    grid = np.full((grid_rows, grid_cols), Tile.EMPTY, dtype=np.uint8)
    entities = {"mario_pos": None, "enemies": []}

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = row * tile_h
            y1 = y0 + tile_h
            x0 = col * tile_w
            x1 = x0 + tile_w

            tile_pixels = image[y0:y1, x0:x1]
            tile_type = _classify_tile_by_color(tile_pixels, row=row)

            if tile_type == Tile.PLAYER:
                entities["mario_pos"] = (row, col)
                grid[row, col] = Tile.PLAYER
            else:
                grid[row, col] = tile_type

    # Detect entities by colored pixel clusters
    if detect_entities:
        _detect_entities(image, grid, entities, tile_h, tile_w,
                         grid_rows, grid_cols)

    return grid, entities


def _detect_entities(
    image: np.ndarray,
    grid: np.ndarray,
    entities: dict,
    tile_h: int,
    tile_w: int,
    grid_rows: int,
    grid_cols: int,
):
    """
    Detect Mario and enemies by looking for concentrated entity-colored pixels.
    More robust than per-tile classification for small sprites.
    """
    # Count red pixels (Mario) — tight: pure red only
    red_mask = (image[:, :, 0] > 200) & (image[:, :, 1] < 60) & (image[:, :, 2] < 60)

    # Count skin pixels (Mario's face/hands) — confirms it's Mario not a red enemy
    skin_mask = ((image[:, :, 0] > 220) & (image[:, :, 1] > 150) & (image[:, :, 1] < 220)
                 & (image[:, :, 2] > 120) & (image[:, :, 2] < 180))

    # Count brown pixels (Goomba) — tighter range to avoid ground
    brown_mask = ((image[:, :, 0] > 140) & (image[:, :, 0] < 190)
                  & (image[:, :, 1] > 60) & (image[:, :, 1] < 110)
                  & (image[:, :, 2] > 20) & (image[:, :, 2] < 70))

    # Count green pixels (Turtle) — exclude pipe greens (darker)
    green_mask = ((image[:, :, 1] > 140) & (image[:, :, 0] < 40)
                  & (image[:, :, 2] < 40))

    mario_candidates = []  # (red_ratio, row, col)

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0, y1 = row * tile_h, (row + 1) * tile_h
            x0, x1 = col * tile_w, (col + 1) * tile_w
            n_pixels = max(tile_h * tile_w, 1)

            # Mario detection: need BOTH red AND skin in same tile
            red_count = red_mask[y0:y1, x0:x1].sum()
            skin_count = skin_mask[y0:y1, x0:x1].sum()
            red_ratio = red_count / n_pixels
            skin_ratio = skin_count / n_pixels
            if red_ratio > 0.10 and skin_ratio > 0.05:
                mario_candidates.append((red_ratio + skin_ratio, row, col))

            # Goomba detection — exclude ground/brick rows
            brown_count = brown_mask[y0:y1, x0:x1].sum()
            if brown_count > n_pixels * 0.25:
                if grid[row, col] not in (Tile.GROUND, Tile.BRICK, Tile.QUESTION):
                    # Don't detect in bottom 3 rows (ground zone)
                    if row < grid_rows - 3:
                        entities["enemies"].append({
                            "type": "goomba", "row": row, "col": col
                        })

            # Turtle detection — must not be a pipe tile
            green_count = green_mask[y0:y1, x0:x1].sum()
            if green_count > n_pixels * 0.25:
                if grid[row, col] not in (Tile.PIPE_L, Tile.PIPE_R, Tile.FLAG):
                    entities["enemies"].append({
                        "type": "turtle", "row": row, "col": col
                    })

    # Pick the best Mario candidate
    if mario_candidates:
        mario_candidates.sort(reverse=True)
        _, mr, mc = mario_candidates[0]
        entities["mario_pos"] = (mr, mc)
        grid[mr, mc] = Tile.PLAYER


# ═══════════════════════════════════════════════════════════════
# GRID TO SIMULATOR (rebuilds a MarioSimulator from detected grid)
# ═══════════════════════════════════════════════════════════════

def grid_to_simulator(
    grid: np.ndarray,
    entities: Optional[Dict] = None,
) -> MarioSimulator:
    """
    Convert a detected tile grid + entities back to a MarioSimulator.

    This is the key function for sim-to-real:
      screenshot -> image_to_ascii() -> grid_to_simulator() -> agent.step()

    Args:
        grid: (16, 20) uint8 array of Tile values
        entities: dict with mario_pos, enemies

    Returns:
        MarioSimulator ready for agent interaction
    """
    enemies = []
    if entities and entities.get("enemies"):
        for e in entities["enemies"]:
            etype_map = {
                "goomba": EnemyType.GOOMBA,
                "turtle": EnemyType.TURTLE,
                "piranha": EnemyType.PIRANHA,
            }
            etype = etype_map.get(e["type"], EnemyType.GOOMBA)
            enemies.append(Enemy(
                etype=etype,
                row=e["row"],
                col=e["col"],
                direction=-1,
            ))

    sim = MarioSimulator(grid.copy(), enemies)
    return sim


# ═══════════════════════════════════════════════════════════════
# ROUNDTRIP SELF-TEST
# ═══════════════════════════════════════════════════════════════

def roundtrip_test(sim: MarioSimulator, tile_size: int = 16) -> Dict:
    """
    Test the converter by doing ASCII -> image -> ASCII.

    Returns dict with accuracy metrics.
    """
    # Step 1: Render to image
    img = ascii_to_image(sim, tile_size=tile_size, viewport=False)

    # Step 2: Convert back to ASCII
    recovered_grid, entities = image_to_ascii(
        img, grid_rows=sim.GRID_H, grid_cols=sim.width
    )

    # Step 3: Compare
    original = sim.grid.copy()
    # Put Mario back in original for comparison
    original[sim.mario_row, sim.mario_col] = Tile.PLAYER

    total = original.size
    matching = int(np.sum(original == recovered_grid))
    accuracy = matching / total

    # Per-tile accuracy
    tile_acc = {}
    for tile in Tile:
        mask = original == tile
        if mask.sum() > 0:
            correct = int(np.sum((original == tile) & (recovered_grid == tile)))
            tile_acc[TILE_CHAR.get(tile, '?')] = {
                "total": int(mask.sum()),
                "correct": correct,
                "accuracy": round(correct / mask.sum(), 2),
            }

    # Mario position detection
    mario_detected = entities.get("mario_pos") is not None
    mario_correct = False
    if mario_detected:
        mario_correct = (entities["mario_pos"] == (sim.mario_row, sim.mario_col))

    return {
        "total_tiles": total,
        "matching": matching,
        "accuracy": round(accuracy, 4),
        "tile_accuracy": tile_acc,
        "mario_detected": mario_detected,
        "mario_correct": mario_correct,
        "mario_pos_actual": (sim.mario_row, sim.mario_col),
        "mario_pos_detected": entities.get("mario_pos"),
        "enemies_detected": len(entities.get("enemies", [])),
        "image_shape": img.shape,
    }


def save_image(img: np.ndarray, path: str):
    """Save RGB image as PPM (no PIL dependency)."""
    h, w = img.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(img.tobytes())
