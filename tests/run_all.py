"""
run_all.py — single entry point that runs pytest + any phase scripts.

Usage:
    python tests/run_all.py            # all pytest tests + phase scripts
    python tests/run_all.py --pytest-only
    python tests/run_all.py --phase-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

PHASE_SCRIPTS: list[str] = [
    "tests/test_canonical_boundaries.py",
    "tests/test_phase2.py",
    "tests/test_phase3.py",
    "tests/test_phase4.py",
]


def run_pytest() -> int:
    print("=" * 60)
    print("Running: pytest tests/ -q")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=ROOT,
    )
    return result.returncode


def run_phase_scripts() -> int:
    overall = 0
    for script in PHASE_SCRIPTS:
        script_path = ROOT / script
        if not script_path.exists():
            print(f"  [skip] {script} — not found")
            continue
        print("=" * 60)
        print(f"Running: python {script}")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"  [FAIL] {script} exited {result.returncode}")
            overall = result.returncode
        else:
            print(f"  [OK]   {script}")
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all tests for throng2.")
    parser.add_argument("--pytest-only", action="store_true")
    parser.add_argument("--phase-only", action="store_true")
    args = parser.parse_args()

    exit_codes: list[int] = []

    if not args.phase_only:
        exit_codes.append(run_pytest())

    if not args.pytest_only:
        exit_codes.append(run_phase_scripts())

    sys.exit(max(exit_codes) if exit_codes else 0)


if __name__ == "__main__":
    main()
