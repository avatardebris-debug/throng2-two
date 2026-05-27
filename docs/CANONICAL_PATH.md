# Canonical training path

Use this stack for new work. Other directories are legacy experiments.

## Recommended entry points

| Goal | Command / module |
|------|------------------|
| Cross-game world model | `python examples/cross_game_training.py --games mario cartpole` (loop in `examples/cross_game/training_loop.py`) |
| MuJoCo training | `python examples/mujoco_training.py` |
| Single-cell RL agent | `src/cell/thronglet_cell.py` + `ThrongletCell` |
| Mario PPO / ICM | `make_mario_agent()` in `src/games/mario/mario_agent.py` |
| Phase regression tests | `python tests/test_phase2.py` (and phase 3–4) |

## Core modules

```
UniversalEncoder / EncoderRegistry   →  src/encoder/
MultiGameWorldModel / CellWorldModel →  src/cell/world_model/ (`WorldModelCore` → single or multi)
CellDreamer                          →  src/cell/dreamer.py
SimpleNumpyAgent (gym baselines)     →  src/learning/numpy_linear_agent.py
MetaEncoder (transfer routing)       →  src/encoder/meta_encoder.py
```

## Dreaming

- **ThrongletCell:** `CellDreamer.blend_action()` (advisory PPO blend).
- **Cross-game training:** `CellDreamer.guided_training_action()` (epsilon-decay WM override).

## Legacy (do not extend without reason)

Implementations live under `legacy/`; `src/core`, `src/event_based`, `src/compression`, and `src/integration` are **deprecation shims** (warn on import).

| Path | Notes |
|------|--------|
| `legacy/core/`, `legacy/event_based/` | Alternate Thronglet / phase pipelines |
| `legacy/compression/` | Compression ablations |
| `legacy/integration/compressed_brain.py` | Standalone experiment |
| `examples/01_*` … `examples/49_*` | Numbered demos; superseded by scripts above |
| `throng5 - Copy/`, `PufferLib-3.0 - Copy/` | Vendored reference trees |

See `archive/README.md` for migration notes.
