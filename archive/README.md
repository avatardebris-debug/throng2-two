# Archive / legacy code

This folder documents code that lives elsewhere in the repo but is **not** part of the maintained product path.

Nothing here is executed by the canonical training scripts. Prefer `docs/CANONICAL_PATH.md` before adding features to legacy trees.

## Locations

- **`legacy/core/`**, **`legacy/event_based/`**, **`legacy/compression/`**, **`legacy/integration/`** — implementations (quarantined from `src/`)
- **`src/core/`**, **`src/event_based/`**, etc. — thin **deprecation shims** that re-export from `legacy.*` (warn on package import)
- **`examples/01_*`–`49_*`** — Early numbered demos (compression, neuromodulators, etc.)

Run from repo root so `legacy` resolves as a top-level package (same as `src`).

## Safe to ignore for cross-game RL

If you are training `MultiGameWorldModel` + `EncoderRegistry`, you do not need to import from these paths.

## Future cleanup

A full move of legacy modules into `archive/` would shrink `src/` but break old example scripts. Do that only with an explicit deprecation pass and CI grep for imports.
