"""Move legacy src trees to legacy/ and install src/ re-export shims."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
PACKAGES = ("core", "event_based", "compression", "integration")

SHIM_HEADER = '''"""DEPRECATED shim — implementation moved to legacy.{pkg}.{mod}.
See docs/CANONICAL_PATH.md. New code must not import from src.{pkg}.
"""
from legacy.{pkg}.{mod} import *  # noqa: F403
'''

INIT_HEADER = '''"""DEPRECATED package shim — see docs/CANONICAL_PATH.md."""
import warnings

warnings.warn(
    "src.{pkg} is legacy; use the canonical path in docs/CANONICAL_PATH.md",
    DeprecationWarning,
    stacklevel=2,
)
'''


def main():
    LEGACY.mkdir(exist_ok=True)
    (LEGACY / "__init__.py").write_text(
        '"""Legacy experiment code (quarantined from src/)."""\n', encoding="utf-8"
    )

    for pkg in PACKAGES:
        src_pkg = ROOT / "src" / pkg
        dst_pkg = LEGACY / pkg
        if not src_pkg.exists():
            print(f"skip missing {src_pkg}")
            continue
        if dst_pkg.exists():
            shutil.rmtree(dst_pkg)
        shutil.move(str(src_pkg), str(dst_pkg))
        print(f"moved src/{pkg} -> legacy/{pkg}")

        shim_pkg = ROOT / "src" / pkg
        shim_pkg.mkdir(parents=True, exist_ok=True)
        (shim_pkg / "__init__.py").write_text(
            INIT_HEADER.format(pkg=pkg), encoding="utf-8"
        )

        for py in (LEGACY / pkg).rglob("*.py"):
            if py.name == "__init__.py":
                rel = py.relative_to(LEGACY / pkg)
                if rel.parts == ("__init__.py",):
                    init_path = LEGACY / pkg / "__init__.py"
                    if not init_path.read_text(encoding="utf-8").strip():
                        init_path.write_text("# legacy package\n", encoding="utf-8")
                continue
            rel = py.relative_to(LEGACY / pkg)
            shim_path = shim_pkg / rel
            shim_path.parent.mkdir(parents=True, exist_ok=True)
            mod = ".".join(rel.with_suffix("").parts)
            shim_path.write_text(SHIM_HEADER.format(pkg=pkg, mod=mod), encoding="utf-8")

        print(f"shims written under src/{pkg}/")

    print("done")


if __name__ == "__main__":
    main()
