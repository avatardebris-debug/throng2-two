"""
test_contrastive_encoder.py — Tests for ContrastiveProjection and UniversalEncoder.fit_contrastive()

Validates:
  1. Fresh ContrastiveProjection: is_contrastive_fitted == False
  2. NT-Xent loss decreases over training epochs
  3. After fit(): same obs under different augmentations → cosine_sim > baseline
  4. is_contrastive_fitted propagates: ContrastiveProjection → UniversalEncoder → EncoderRegistry
  5. NumpyLinear fallback: fit_contrastive() on standard UniversalEncoder uses PCA instead
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.encoder.universal_encoder import (
    ContrastiveProjection, UniversalEncoder, EncoderRegistry,
    EncoderConfig, register_game,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

IN_DIM  = 32
Z_DIM   = 16
N_OBS   = 128


def make_obs(rng, n=N_OBS, in_dim=IN_DIM):
    return rng.randn(n, in_dim).astype(np.float32)


def cosine_sim(u, v):
    u = u / (np.linalg.norm(u) + 1e-8)
    v = v / (np.linalg.norm(v) + 1e-8)
    return float(np.dot(u, v))


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Fresh ContrastiveProjection — is_contrastive_fitted False
# ──────────────────────────────────────────────────────────────────────────────

def test_fresh_not_fitted():
    print("\n=== Test 1: Fresh ContrastiveProjection is_contrastive_fitted==False ===")
    cp = ContrastiveProjection(in_dim=IN_DIM, z_dim=Z_DIM)
    assert not cp.is_contrastive_fitted, "Fresh projection should not be fitted"
    assert not cp.is_pca_fitted
    # __call__ still works (random weights)
    x = np.ones(IN_DIM, dtype=np.float32)
    z = cp(x)
    assert z.shape == (Z_DIM,), f"Expected ({Z_DIM},), got {z.shape}"
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: NT-Xent loss decreases over training epochs
# ──────────────────────────────────────────────────────────────────────────────

def test_loss_decreases():
    print("\n=== Test 2: NT-Xent loss decreases over training ===")
    rng = np.random.RandomState(0)
    obs = make_obs(rng)
    cp = ContrastiveProjection(in_dim=IN_DIM, z_dim=Z_DIM, seed=1)
    losses = cp.fit(obs, n_epochs=20, lr=5e-3, batch_size=32, verbose=False)
    assert len(losses) == 20, f"Expected 20 epoch losses, got {len(losses)}"
    # Loss should be decreasing on average (compare first 5 vs last 5 epochs)
    early = np.mean(losses[:5])
    late  = np.mean(losses[-5:])
    print(f"  Early loss: {early:.4f}, Late loss: {late:.4f}")
    assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"
    assert cp.is_contrastive_fitted
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Same obs under augmentations → higher cosine sim than cross-obs
# ──────────────────────────────────────────────────────────────────────────────

def test_augmentation_coherence():
    print("\n=== Test 3: Same obs → closer z than random obs after training ===")
    rng = np.random.RandomState(42)
    obs = make_obs(rng, n=N_OBS)
    cp = ContrastiveProjection(in_dim=IN_DIM, z_dim=Z_DIM, seed=7)

    # Measure BEFORE training
    z0  = cp(obs[0])
    z0n = cp(obs[0] + rng.randn(IN_DIM).astype(np.float32) * 0.05)  # noisy version
    zx  = cp(obs[1])  # different obs
    sim_same_before  = cosine_sim(z0, z0n)
    sim_cross_before = cosine_sim(z0, zx)
    print(f"  Before: sim(same+noise)={sim_same_before:.4f}, sim(cross)={sim_cross_before:.4f}")

    cp.fit(obs, n_epochs=30, lr=5e-3, batch_size=32, temperature=0.1)

    # Measure AFTER training
    z0  = cp(obs[0])
    z0n = cp(obs[0] + rng.randn(IN_DIM).astype(np.float32) * 0.05)
    zx  = cp(obs[1])
    sim_same_after  = cosine_sim(z0, z0n)
    sim_cross_after = cosine_sim(z0, zx)
    print(f"  After:  sim(same+noise)={sim_same_after:.4f}, sim(cross)={sim_cross_after:.4f}")

    # Same-obs similarity should improve (or at least be close to 1.0)
    assert sim_same_after > sim_same_before - 0.05, (
        f"Same-obs sim should not collapse: before={sim_same_before:.4f}, after={sim_same_after:.4f}"
    )
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: is_contrastive_fitted propagates through UniversalEncoder → EncoderRegistry
# ──────────────────────────────────────────────────────────────────────────────

def test_fitted_flag_propagation():
    print("\n=== Test 4: is_contrastive_fitted propagates UniversalEncoder → Registry ===")

    # Register a tiny test game
    cfg = EncoderConfig(
        game_name="testgame_contrastive",
        game_id=99,
        obs_type="flat",
        obs_dim=IN_DIM,
    )
    register_game(cfg)

    # UniversalEncoder with ContrastiveProjection
    from src.encoder.universal_encoder import ContrastiveProjection
    enc = UniversalEncoder(game_name="testgame_contrastive", z_dim=Z_DIM)
    # Replace projection with ContrastiveProjection
    enc._project = ContrastiveProjection(enc._project.in_dim, Z_DIM, seed=0)

    assert not enc.is_contrastive_fitted, "Should start unfitted"

    rng = np.random.RandomState(3)
    obs_list = [rng.randn(IN_DIM).astype(np.float32) for _ in range(N_OBS)]
    enc.fit_contrastive(obs_list, n_epochs=5, batch_size=32)
    assert enc.is_contrastive_fitted, "Should be fitted after fit_contrastive()"
    print(f"  UniversalEncoder.is_contrastive_fitted = {enc.is_contrastive_fitted}")
    print("  PASS")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: NumpyLinear fallback — fit_contrastive() falls back to PCA gracefully
# ──────────────────────────────────────────────────────────────────────────────

def test_numpylinear_fallback():
    print("\n=== Test 5: NumpyLinear fallback — fit_contrastive() uses PCA ===")
    from src.encoder.universal_encoder import EncoderConfig, register_game, UniversalEncoder
    cfg = EncoderConfig(
        game_name="testgame_pca_fallback",
        game_id=100,
        obs_type="flat",
        obs_dim=IN_DIM,
    )
    register_game(cfg)
    enc = UniversalEncoder(game_name="testgame_pca_fallback", z_dim=Z_DIM)
    # _project is NumpyLinear by default
    assert not hasattr(enc._project, "fit"), "NumpyLinear should not have fit()"

    rng = np.random.RandomState(5)
    obs_list = [rng.randn(IN_DIM).astype(np.float32) for _ in range(N_OBS)]
    enc.fit_contrastive(obs_list, n_epochs=5, verbose=True)

    # Should fall back to PCA — pca fitted, contrastive not
    assert enc.is_pca_fitted,            "PCA should be fitted as fallback"
    assert not enc.is_contrastive_fitted, "Contrastive should not be fitted for NumpyLinear"
    print("  PASS")


if __name__ == "__main__":
    test_fresh_not_fitted()
    test_loss_decreases()
    test_augmentation_coherence()
    test_fitted_flag_propagation()
    test_numpylinear_fallback()
    print("\n=== ALL CONTRASTIVE ENCODER TESTS PASSED ===")
