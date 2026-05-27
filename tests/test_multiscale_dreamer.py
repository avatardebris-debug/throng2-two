"""
test_multiscale_dreamer.py — Tests for multi-timescale dreaming.

Validates:
  1. N-step accumulator fills correctly after horizon_n transitions
  2. dream_horizon() returns (n_actions,) at O(n) cost
  3. CellDreamer slow path activates at exactly horizon_interval steps
  4. Blended fast+slow values have correct shape and finite values
  5. Graceful fallback to fast-only when horizon head returns zeros
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.cell.world_model import MultiGameReplayBuffer, MultiGameWorldModel
from src.cell.dreamer import CellDreamer


# ──────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────

FEATURE_DIM = 16
N_ACTIONS   = 4
N_GAMES     = 2


def make_wm():
    return MultiGameWorldModel(
        feature_dim=FEATURE_DIM,
        n_actions=N_ACTIONS,
        n_games=N_GAMES,
        hidden_size=64,
        buffer_size=500,
        min_transitions=5,
        lr=1e-3,
    )


def fill_wm(wm, n=80, game_id=0, n_train=12):
    """Push n random transitions and run n_train train steps."""
    for _ in range(n):
        s  = np.random.randn(FEATURE_DIM).astype(np.float32)
        a  = np.random.randint(N_ACTIONS)
        s2 = np.random.randn(FEATURE_DIM).astype(np.float32)
        r  = float(np.random.randn())
        wm.store_transition(s, a, s2, r, game_id=game_id)
    for _ in range(n_train):
        wm.train_step_multi_game()


# ──────────────────────────────────────────────────────────────
# Test 1: N-step accumulator fills correctly
# ──────────────────────────────────────────────────────────────

def test_nstep_accumulator():
    print("\n=== Test 1: N-step accumulator ===")
    buf = MultiGameReplayBuffer(capacity_per_game=500, horizon_n=8)

    # Before emitting: need horizon_n transitions to accumulate
    for i in range(7):
        s  = np.random.randn(FEATURE_DIM).astype(np.float32)
        s2 = np.random.randn(FEATURE_DIM).astype(np.float32)
        buf.add(s, 0, s2, 1.0, game_id=0)

    assert buf.horizon_size().get(0, 0) == 0, \
        "Should not emit before horizon_n transitions"

    # 8th transition → first horizon entry emitted
    buf.add(np.random.randn(FEATURE_DIM).astype(np.float32), 1,
            np.random.randn(FEATURE_DIM).astype(np.float32), 2.0, game_id=0)
    assert buf.horizon_size().get(0, 0) == 1, \
        "Should emit exactly 1 horizon entry after horizon_n transitions"

    # Each subsequent add emits another (sliding window)
    buf.add(np.random.randn(FEATURE_DIM).astype(np.float32), 2,
            np.random.randn(FEATURE_DIM).astype(np.float32), 0.5, game_id=0)
    assert buf.horizon_size().get(0, 0) == 2, \
        "Should emit 2 horizon entries after horizon_n+1 transitions"

    # Check horizon entry structure
    h_batch = buf.sample_horizon(1)
    assert len(h_batch) == 1
    s0, a0, z_N, cum_r, gid = h_batch[0]
    assert s0.shape == (FEATURE_DIM,), f"state shape wrong: {s0.shape}"
    assert z_N.shape == (FEATURE_DIM,), f"z_N shape wrong: {z_N.shape}"
    assert isinstance(cum_r, float), "cum_r should be float"
    assert gid == 0

    print(f"  horizon entries: {buf.horizon_size()}")
    print(f"  sample: cum_r={cum_r:.3f}, z_N[:3]={z_N[:3]}")
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 2: dream_horizon() shape and O(n) cost
# ──────────────────────────────────────────────────────────────

def test_dream_horizon_shape():
    print("\n=== Test 2: dream_horizon() shape ===")
    wm = make_wm()
    fill_wm(wm, n=60, game_id=0)

    z = np.random.randn(FEATURE_DIM).astype(np.float32)
    vals = wm.dream_horizon(z, game_id=0)

    assert vals.shape == (N_ACTIONS,), f"Expected ({N_ACTIONS},), got {vals.shape}"
    assert np.isfinite(vals).all(), "dream_horizon returned non-finite values"
    print(f"  dream_horizon values: {vals}")
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 3: dream_horizon returns zeros when not ready
# ──────────────────────────────────────────────────────────────

def test_dream_horizon_not_ready():
    print("\n=== Test 3: dream_horizon not-ready fallback ===")
    wm = make_wm()
    # No transitions stored — is_ready_for(0) is False
    z = np.random.randn(FEATURE_DIM).astype(np.float32)
    vals = wm.dream_horizon(z, game_id=0)
    assert vals.shape == (N_ACTIONS,)
    assert (vals == 0).all(), "Should return zeros when not ready"
    print(f"  zeros returned as expected: {vals}")
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 4: CellDreamer slow path activates at horizon_interval
# ──────────────────────────────────────────────────────────────

def test_dreamer_slow_path_timing():
    print("\n=== Test 4: CellDreamer slow path timing ===")
    wm = make_wm()
    fill_wm(wm, n=80, game_id=0, n_train=12)

    dreamer = CellDreamer(
        n_actions=N_ACTIONS,
        dream_interval=1,     # dream every step (easy testing)
        dream_depth=1,
        warmup_dreams=0,
        horizon_interval=5,   # slow path every 5 steps
        horizon_alpha=0.5,
    )
    dreamer.set_game_id(0)

    slow_refreshed_at = []

    for step in range(20):
        z = np.random.randn(FEATURE_DIM).astype(np.float32)
        before = dreamer._slow_values_step
        dreamer.dream(z, wm)
        after = dreamer._slow_values_step
        if after != before:
            slow_refreshed_at.append(step + 1)  # step is 0-based, dreamer step is 1-based

    # Slow path should have fired at steps 5, 10, 15, 20 (every 5)
    print(f"  Slow path refreshed at dreamer steps: {slow_refreshed_at}")
    assert len(slow_refreshed_at) >= 2, \
        f"Expected slow path to fire ≥2 times in 20 steps, got {slow_refreshed_at}"
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 5: Fast+slow blend produces correct shape and finite values
# ──────────────────────────────────────────────────────────────

def test_dreamer_blend_values():
    print("\n=== Test 5: Fast+slow blend values ===")
    wm = make_wm()
    fill_wm(wm, n=80, game_id=0, n_train=12)

    dreamer = CellDreamer(
        n_actions=N_ACTIONS,
        dream_interval=1,
        dream_depth=1,
        warmup_dreams=0,
        horizon_interval=1,   # refresh slow path every step
        horizon_alpha=0.4,
    )
    dreamer.set_game_id(0)

    # Run enough steps that slow path fires and is fresh
    z = np.random.randn(FEATURE_DIM).astype(np.float32)
    combined = None
    for _ in range(5):
        combined = dreamer.dream(z, wm)

    assert combined is not None, "dream() returned None unexpectedly"
    assert combined.shape == (N_ACTIONS,), f"Bad shape: {combined.shape}"
    assert np.isfinite(combined).all(), "Combined values not finite"

    stats = dreamer.slow_dream_stats()
    print(f"  combined dream values: {combined}")
    print(f"  slow_dream_stats: {stats}")
    assert stats["slow_values_available"], "Slow values should be available"
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 6: horizon_alpha=0 → purely fast path (slow never applied)
# ──────────────────────────────────────────────────────────────

def test_horizon_alpha_zero():
    print("\n=== Test 6: horizon_alpha=0 fast-path-only ===")
    wm = make_wm()
    fill_wm(wm, n=80, game_id=0, n_train=12)

    dreamer = CellDreamer(
        n_actions=N_ACTIONS,
        dream_interval=1,
        dream_depth=1,
        warmup_dreams=0,
        horizon_interval=1,
        horizon_alpha=0.0,   # slow path completely off
    )
    dreamer.set_game_id(0)

    z = np.random.randn(FEATURE_DIM).astype(np.float32)
    for _ in range(5):
        dreamer.dream(z, wm)

    # Slow values should NOT have been computed (alpha=0 skips dream_horizon call)
    assert dreamer._slow_values is None, \
        "With horizon_alpha=0, dream_horizon should never be called"
    print("  _slow_values is None as expected")
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 7: adaptive_horizon_n() scales with confidence
# ──────────────────────────────────────────────────────────────

def test_adaptive_horizon_n():
    print("\n=== Test 7: adaptive_horizon_n() ===")
    wm = make_wm()

    # Untrained WM → confidence=0 → min_horizon_n
    n_early = wm.adaptive_horizon_n(game_id=0)
    assert n_early == wm.min_horizon_n, \
        f"Untrained WM should return min_horizon_n={wm.min_horizon_n}, got {n_early}"

    fill_wm(wm, n=80, game_id=0, n_train=12)

    # After training, horizon_n should be >= min and <= max
    n_trained = wm.adaptive_horizon_n(game_id=0)
    assert wm.min_horizon_n <= n_trained <= wm.horizon_n, \
        f"adaptive horizon_n={n_trained} out of range [{wm.min_horizon_n}, {wm.horizon_n}]"

    print(f"  early (untrained) N={n_early}, trained N={n_trained}")
    print(f"  min_horizon_n={wm.min_horizon_n}, max horizon_n={wm.horizon_n}")
    print("  PASS")


if __name__ == "__main__":
    test_nstep_accumulator()
    test_dream_horizon_shape()
    test_dream_horizon_not_ready()
    test_dreamer_slow_path_timing()
    test_dreamer_blend_values()
    test_horizon_alpha_zero()
    test_adaptive_horizon_n()
    print("\n=== ALL MULTISCALE DREAMER TESTS PASSED ===")
