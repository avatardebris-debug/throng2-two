# Throng2 Master — Module Map

A cross-game universal AI agent built on the **Throng2** simulation-first paradigm.

---

## Architecture overview

```
Any environment
      │
      ▼
UniversalEncoder (src/encoder/universal_encoder.py)
  ASCII/pixel obs → z-vector (same shape, all games)
      │
      ├─→ MultiGameWorldModel (src/cell/world_model/)
      │     shared encoder + per-game heads
      │     dream_all_actions() → advisory policy
      │
      ├─→ MetaEncoder (src/encoder/meta_encoder.py)
      │     episode summaries → c_game challenge descriptor
      │     cosine similarity → warm-start transfer routing
      │
      └─→ CellDreamer (src/cell/dreamer.py)
            blend PPO actions with dream values
```

---

## Directory guide

| Path | What's in it |
|---|---|
| `src/encoder/` | UniversalEncoder, MetaEncoder, AsciiEncoder, `projections.py` |
| `docs/CANONICAL_PATH.md` | Which modules and scripts to use (vs legacy) |
| `src/cell/` | ThrongletCell (SNN+PPO), `world_model/` package, CellDreamer |
| `src/games/mario/` | MarioSimulator, MarioAdapter, zone curriculum, HPO |
| `src/games/mujoco/` | MuJoCoAdapter, MuJoCoActionDiscretizer (3 strategies) |
| `src/games/atari/` | AtariAdapter → 15×20 ASCII frames from ALE |
| `src/learning/` | DopamineSystem, STDPLearning, SpatialMemory, NeuromodulatorBridge |
| `src/compression/` | 6 z-space compression strategies |
| `examples/` | Training scripts (cross_game_training.py, mujoco_training.py) |
| `tests/` | Phase 1–4 unit tests (60 total, all passing) |

---

## Phases complete

| Phase | File(s) | Tests |
|---|---|---|
| 1A Zone curriculum | `mario_zone_curriculum.py` | 7 ✓ |
| 1B Difficulty spots | `mario_difficulty_analyzer.py` | 9 ✓ |
| 1C Bayesian HPO | `mario_hpo.py` | 7 ✓ |
| 2A Universal Encoder | `universal_encoder.py` | 11 ✓ |
| 2B/C/D World Model | `world_model.py`, `cross_game_training.py`, `dreamer.py` | (in P2) |
| 3A/B/C MuJoCo | `mujoco_adapter.py`, `mujoco_action_discretizer.py`, `mujoco_training.py` | 12 ✓ |
| 4A Meta-Encoder | `meta_encoder.py` | 14 ✓ |
| 4B Neuromodulators | `neuromodulator_bridge.py` | (in P4) |
| 4C Atari | `atari_adapter.py` | (in P4) |

---

## Quick start

```bash
# No dependencies (pure numpy):
python examples/cross_game_training.py --games mario cartpole

# MuJoCo fallback sim (no mujoco install needed):
python examples/mujoco_training.py --episodes 100

# All tests:
python tests/test_phase2.py
python tests/test_phase3.py
python tests/test_phase4.py
```
