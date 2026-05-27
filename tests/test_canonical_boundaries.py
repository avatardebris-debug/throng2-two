"""
Guardrails: canonical modules must not import legacy experiment trees.

Run: python tests/test_canonical_boundaries.py
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_FILES = [
    ROOT / "examples" / "cross_game_training.py",
    ROOT / "examples" / "cross_game" / "episode.py",
    ROOT / "examples" / "cross_game" / "runners.py",
    ROOT / "examples" / "cross_game" / "training_loop.py",
    ROOT / "src" / "cell" / "thronglet_cell.py",
    ROOT / "src" / "cell" / "dreamer.py",
    ROOT / "src" / "cell" / "world_model" / "base.py",
    ROOT / "src" / "cell" / "world_model" / "multi.py",
    ROOT / "src" / "cell" / "world_model" / "core.py",
    ROOT / "src" / "cell" / "world_model" / "protocol.py",
    ROOT / "src" / "encoder" / "universal_encoder.py",
    ROOT / "src" / "games" / "mario" / "mario_agent.py",
    ROOT / "src" / "games" / "mario" / "mario_icm_agent.py",
    ROOT / "src" / "games" / "mario" / "backends" / "numpy_ppo.py",
]

FORBIDDEN_PREFIXES = (
    "src.core",
    "src.event_based",
    "src.compression",
    "src.integration.compressed_brain",
)


def _imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def test_canonical_paths_avoid_legacy_imports():
    violations: list[str] = []
    for path in CANONICAL_FILES:
        if not path.exists():
            continue
        for mod in _imports_in_file(path):
            for bad in FORBIDDEN_PREFIXES:
                if mod == bad or mod.startswith(bad + "."):
                    violations.append(f"{path.relative_to(ROOT)}: {mod}")
    assert not violations, "Legacy imports in canonical path:\n" + "\n".join(violations)


if __name__ == "__main__":
    test_canonical_paths_avoid_legacy_imports()
    print("test_canonical_boundaries: PASS")
