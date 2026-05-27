"""
world1_levels.py — Canonical World 1 level data from the VGLC dataset.

Converted from the Video Game Level Corpus (TheVGLC/TheVGLC on GitHub)
into MarioSimulator-compatible grids with enemies.

Levels included:
  1-1: Overworld   (212 cols) — standard platforming, goombas, pipes
  1-2: Underground (150 cols) — enclosed ceiling, coins, short pits
  1-3: Athletic    (150 cols) — elevated platforms, no ground, koopas

1-4 (Castle) is excluded for now — needs fire bar + lava tile support.

VGLC tile encoding → our Tile enum:
  '-' → EMPTY       'X' → GROUND     'S' → BRICK
  'Q' → QUESTION    '?' → QUESTION   'o' → COIN
  '<' → PIPE_L      '>' → PIPE_R     '[' → PIPE_L (lower)
  ']' → PIPE_R (lower)               'E' → enemy marker (stripped, placed as Enemy)

Usage:
    from src.games.mario.world1_levels import load_world1, get_random_world1

    levels = load_world1()      # {'1-1': sim, '1-2': sim, '1-3': sim}
    sim = get_random_world1()   # random choice, deep-copied
"""

from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Optional
import numpy as np

from .mario_simulator import (
    Enemy, EnemyType, MarioSimulator, Tile, SOLID_TILES,
)


# ══════════════════════════════════════════════════════════════════════
#  Raw VGLC data — 14 rows (VGLC uses 14-row format, we pad to 16)
# ══════════════════════════════════════════════════════════════════════

# Each string is one row, left-to-right = column 0..N
# Lines are from the VGLC "Processed" folder, rows top to bottom

_RAW_1_1 = [
    "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
    "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
    "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
    "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
    "----------------------------------------------------------------------------------E-----------------------------------------------------------------------------------------------------------------------",
    "----------------------Q---------------------------------------------------------SSSSSSSS---SSSQ--------------?-----------SSS----SQQS--------------------------------------------------------XX------------",
    "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------XXX------------",
    "-------------------------------------------------------------------------------E----------------------------------------------------------------------------------------------------------XXXX------------",
    "----------------------------------------------------------------S------------------------------------------------------------------------------------------------------------------------XXXXX------------",
    "----------------Q---S?SQS---------------------<>---------<>------------------S?S--------------S-----SS----Q--Q--Q-----S----------SS------X--X----------XX--X------------SSQS------------XXXXXX------------",
    "--------------------------------------<>------[]---------[]-----------------------------------------------------------------------------XX--XX--------XXX--XX--------------------------XXXXXXX------------",
    "----------------------------<>--------[]------[]---------[]----------------------------------------------------------------------------XXX--XXX------XXXX--XXX-----<>--------------<>-XXXXXXXX------------",
    "---------------------E------[]--------[]-E----[]-----E-E-[]------------------------------------E-E--------E-----------------EE-E-E----XXXX--XXXX----XXXXX--XXXX----[]---------EE---[]XXXXXXXXX--------X---",
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XXXXXXXXXXXXXXX---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
]

_RAW_1_2 = [
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------",
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------",
    "S-----SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS?SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS--------------------",
    "S-----------------------------------------------------SS--SSSSSS--SSSS------SSSS------------------------------------------------------------------------------",
    "S-----------------------------------------------------SS--SSSSSS--SSSS--E---SSSS------------------------------------------------------------XXX---------------",
    "S----------------------------------------oooo-------SS--------SS---S----SS----------oooooo-----------------------------------------------------------------XXX",
    "S-----------------------------o---------------------SS--------SS---S----SS------------------------------------------------------------------------------------",
    "S--------------------------------------S-SSSS-S-----SS--------SS---S----SS----------SSSSSS--------------------------------------------------------------------",
    "S----------------------------S---------SoS--SoS-----SS----ooooSS---So??-SS---E------SSSSSS--------------------E----------------------------------SSSSS??------",
    "S---------QQQQQ--------XXXX------------SSS--SSS-----SSSS--SSSSSS---SSS--SS--SSSS------------------------E----<>-------------------------XXX-------------------",
    "S--------------------XXXXXXXX--XX---------------------SS-------------------EE--------------------------<>----[]-----E-----SS-----------EXXX-XXX---------------",
    "S------------------XXXXXXXXXX--XXXX-------------------SS-----------------------------------------------[]----[]----<>-----SS---------EXXXXX-------------------",
    "S-------------EE-XXXXXXXXXXXXE-XXXX-------EE---------E------E--E--------------------------------E-EE---[]----[]--E-[]-----SS---------XXXXXX---------E---------",
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--XX--XXXXXXXXXXXX-------XXXXXXXX--XXX",
]

_RAW_1_3 = [
    "------------------------------------------------------------------------------------------------------------------------------------------------------",
    "------------------------------------------------------------------------------------------------------------------------------------------------------",
    "--------------------------------oo--------------------------------------------------------------------------------------------------------------------",
    "------------------------------------E--E--------------------------------------------------------------------------------------------------------------",
    "---------------------Eooo----------XXXXXXX-------------oooo----------E------------------oo--oo--------------------------------------------------------",
    "---------------------XXXXX-----------------------------XXXX----------E--E-------oo---------------------------E-----oo--------------------XX-----------",
    "---------------------------------------------oo------------------------XXXXXX------------------------E---------------------XXX-----------XX-----------",
    "---------------------------------------------------------------------------------------------------XXXXXXXX----------------------------XXXX-----------",
    "------------------------------XXXXX----------------------------------------------XXX---------------------------------------------------XXXX-----------",
    "-------------------XXXXXXXX--------------------------------------XXX---------------------XXX-------------------XXXX--XXXX------------XXXXXX-----------",
    "--------------------------------------------------XXXX?------------------------------------------------------------------------------XXXXXX-----------",
    "----------------------------o----------------------------------------------------------------XXXX------------------------------------XXXXXX-----------",
    "-------------XXXX----------XXX------------------------------------------------------------------------------ooo------------------E---XXXXXX--------X--",
    "XXXXXXXXXXX----------------------------------XXXX-----XXXXX-XXXXX-------------------------------------------XXX-------------XXXXXXXXXXXXXXXXXXXXXXXXXX",
]


# ══════════════════════════════════════════════════════════════════════
#  VGLC → MarioSimulator converter
# ══════════════════════════════════════════════════════════════════════

# Tile mapping
_VGLC_TILE = {
    '-': Tile.EMPTY,
    'X': Tile.GROUND,
    'S': Tile.BRICK,
    'Q': Tile.QUESTION,
    '?': Tile.QUESTION,
    'o': Tile.COIN,
    '<': Tile.PIPE_L,
    '>': Tile.PIPE_R,
    '[': Tile.PIPE_L,   # lower pipe section uses same tiles
    ']': Tile.PIPE_R,
    'E': Tile.EMPTY,    # Enemy marker — extracted separately
}


def _parse_vglc(
    raw_rows: List[str],
    level_name: str = "unknown",
    is_underground: bool = False,
) -> MarioSimulator:
    """
    Convert 14-row VGLC ASCII data into a 16-row MarioSimulator.

    VGLC uses 14 rows (no HUD). Our sim uses 16 rows.
    We pad 2 rows at the top (sky) and map row 13 = VGLC row 13 (ground).

    Enemy positions are extracted from 'E' markers and placed as
    Goomba or Turtle (alternating for variety).
    """
    # Determine width from the data
    width = max(len(row) for row in raw_rows)

    # Pad rows to consistent width
    padded = [row.ljust(width, '-') for row in raw_rows]

    # VGLC has 14 rows. Our grid has 16. Insert 2 empty rows at top.
    grid = np.full((16, width), Tile.EMPTY, dtype=np.uint8)
    enemies: List[Enemy] = []

    enemy_toggle = 0  # alternate goomba/turtle

    for vglc_row, row_str in enumerate(padded):
        # Map VGLC row → our grid row (+2 offset for top padding)
        grid_row = vglc_row + 2

        for col, ch in enumerate(row_str):
            if col >= width:
                break

            if ch == 'E':
                # Enemy marker — place enemy, tile is EMPTY
                grid[grid_row, col] = Tile.EMPTY
                etype = EnemyType.GOOMBA if enemy_toggle % 3 != 2 else EnemyType.TURTLE
                enemy_toggle += 1
                enemies.append(Enemy(
                    etype=etype,
                    row=grid_row,
                    col=col,
                    direction=-1,
                ))
            else:
                tile = _VGLC_TILE.get(ch, Tile.EMPTY)
                grid[grid_row, col] = tile

    # Underground levels: fill top 2 rows with BRICK (ceiling)
    if is_underground:
        grid[0, :] = Tile.BRICK
        grid[1, :] = Tile.BRICK
        grid[2, :] = Tile.BRICK  # row 2 too (maps to VGLC row 0 which has 'S')

    # Place player at start — find leftmost ground column
    player_row, player_col = 14, 2  # defaults
    for c in range(2, min(10, width)):
        # Look for ground below
        for r in range(14, 2, -1):
            if grid[r, c] in SOLID_TILES and r > 0 and grid[r - 1, c] == Tile.EMPTY:
                player_row = r - 1
                player_col = c
                break
        if player_row != 14:
            break
    grid[player_row, player_col] = Tile.PLAYER

    # Place flag near end
    flag_col = width - 3
    for c in range(width - 3, width - 10, -1):
        if grid[15, c] in SOLID_TILES or grid[14, c] in SOLID_TILES:
            flag_col = c
            break
    for fr in range(12, 15):
        if grid[fr, flag_col] == Tile.EMPTY:
            grid[fr, flag_col] = Tile.FLAG

    # Fix pits: columns with no ground in bottom rows → PIT
    for col in range(width):
        has_ground = False
        for r in range(13, 16):
            if grid[r, col] in SOLID_TILES:
                has_ground = True
                break
        if not has_ground:
            # Check it's not a pipe
            is_pipe_col = any(grid[r, col] in (Tile.PIPE_L, Tile.PIPE_R) for r in range(16))
            if not is_pipe_col:
                grid[15, col] = Tile.PIT

    sim = MarioSimulator(grid, enemies)
    return sim


# ══════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════

# Cached parsed levels (lazy)
_cache: Optional[Dict[str, MarioSimulator]] = None


def load_world1() -> Dict[str, MarioSimulator]:
    """
    Parse and return all World 1 canonical levels.

    Returns:
        dict: {'1-1': MarioSimulator, '1-2': ..., '1-3': ...}
    """
    global _cache
    if _cache is not None:
        return _cache

    _cache = {
        '1-1': _parse_vglc(_RAW_1_1, "World 1-1", is_underground=False),
        '1-2': _parse_vglc(_RAW_1_2, "World 1-2", is_underground=True),
        '1-3': _parse_vglc(_RAW_1_3, "World 1-3", is_underground=False),
    }
    return _cache


def get_random_world1(rng=None) -> MarioSimulator:
    """Return a deep-copied random World 1 sub-level."""
    levels = load_world1()
    if rng is None:
        rng = np.random.RandomState()
    key = rng.choice(list(levels.keys()))
    return deepcopy(levels[key])


def get_world1_level(name: str) -> MarioSimulator:
    """Return a deep-copied specific level like '1-1'."""
    levels = load_world1()
    if name not in levels:
        raise ValueError(f"Unknown level '{name}'. Available: {list(levels.keys())}")
    return deepcopy(levels[name])
