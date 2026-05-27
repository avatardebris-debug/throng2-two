"""
test_surprise_mode.py — Tests for SurpriseMap and DualModeEncoder.surprise_auto_mode()
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.encoder.dual_mode_encoder import SurpriseMap, DualModeEncoder


# ── Helpers ────────────────────────────────────────────────────

class FakeWorldModel:
    """Minimal mock with .surprise(game_id) -> float."""
    def __init__(self, value: float = 0.0):
        self.value = value
    def surprise(self, game_id: int = 0) -> float:
        return self.value


# ── Test 1: SurpriseMap — store, decay, lookup ──────────────────

def test_surprise_map_basic():
    print("\n=== Test 1: SurpriseMap store/decay/lookup ===")
    sm = SurpriseMap(resolution=4, decay=0.9)
    z = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)

    # First update
    sm.update(z, surprise=1.0)
    assert sm.n_cells == 1
    val = sm.predict_surprise(z)
    assert abs(val - 1.0) < 1e-6, f"Expected 1.0, got {val}"

    # Second update with same z → EMA decay
    sm.update(z, surprise=0.0)
    val2 = sm.predict_surprise(z)
    expected = 0.9 * 1.0 + 0.1 * 0.0   # = 0.9
    assert abs(val2 - expected) < 1e-6, f"Expected {expected}, got {val2}"

    # Unknown z → default
    z2 = np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32)
    assert sm.predict_surprise(z2, default=-1.0) == -1.0

    print(f"  SurpriseMap stats: {sm.stats()}")
    print("  PASS")


# ── Test 2: Rolling-window hysteresis — single spike does NOT trigger ──

def test_hysteresis_prevents_thrashing():
    print("\n=== Test 2: Hysteresis — single surprise spike does not trigger switch ===")
    enc = DualModeEncoder(
        game_name="cartpole",
        z_dim=32,
        initial_mode="fast",
    )
    obs = np.random.randn(4).astype(np.float32)

    # Window=3, so we need 3 consecutive high readings to trigger switch
    # Feed only ONE high reading — should stay in fast mode
    wm_high = FakeWorldModel(value=1.0)
    enc.surprise_auto_mode(wm_high, game_id=0, obs=obs,
                           threshold=0.5, window=3, use_map=False)
    assert enc.mode == "fast", f"Single spike should not switch mode, got {enc.mode!r}"

    # Now feed 3 consecutive high readings
    for _ in range(3):
        enc.surprise_auto_mode(wm_high, game_id=0, obs=obs,
                               threshold=0.5, window=3, use_map=False)
    # detail path requires torch — without torch, should stay "fast"
    if enc.detail_available:
        assert enc.mode == "detail", f"3 consecutive spikes should switch to detail"
    else:
        assert enc.mode == "fast", "detail not available, stays fast"

    print(f"  Mode after spikes: {enc.mode!r}, detail_available={enc.detail_available}")
    print("  PASS")


# ── Test 3: SurpriseMap pre-empts rolling window ───────────────

def test_surprise_map_preemption():
    print("\n=== Test 3: SurpriseMap pre-empts rolling window ===")

    enc = DualModeEncoder(game_name="cartpole", z_dim=32, initial_mode="fast")

    # Use a fixed non-zero obs so the encoder produces a defined z
    obs = np.ones(4, dtype=np.float32) * 0.5

    # Get the z the encoder will actually produce for this obs
    z_actual = enc._fast_enc.encode(obs)

    # Inject high surprise at this exact cell into the map
    enc._surprise_map.update(z_actual, surprise=1.0)
    assert enc._surprise_map.predict_surprise(z_actual) >= 0.9, \
        "SurpriseMap should have stored surprise for this z-cell"

    # World model reports LOW surprise -- map alone should trigger switch
    wm_low = FakeWorldModel(value=0.0)
    enc.surprise_auto_mode(wm_low, game_id=0, obs=obs,
                           threshold=0.5, use_map=True)

    if enc.detail_available:
        assert enc.mode == "detail", "Map pre-emption should have switched to detail"
    else:
        # Without torch detail path, the flag is set but mode stays "fast"
        assert enc._in_surprise_mode, \
            "Even without detail available, _in_surprise_mode should be True after map pre-emption"

    print(f"  Mode after map pre-emption: {enc.mode!r}, in_surprise_mode={enc._in_surprise_mode}")
    print("  PASS")


if __name__ == "__main__":
    test_surprise_map_basic()
    test_hysteresis_prevents_thrashing()
    test_surprise_map_preemption()
    print("\n=== ALL SURPRISE MODE TESTS PASSED ===")
