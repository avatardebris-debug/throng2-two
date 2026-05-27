"""
test_meta_encoder_triplet.py — Tests for _TripletProjection and MetaEncoder.fit_projection()

Validates:
  1. Fresh MetaEncoder: projection_fitted == False
  2. After fit_projection() with 2 games: projection_fitted == True
  3. After fitting: same-game cosine distance < cross-game cosine distance
  4. fit_projection() with < 2 games: no-op (returns 0.0)
  5. Descriptors are finite and unit-normed after fitting
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.encoder.meta_encoder import MetaEncoder, EpisodeSummary, _TripletProjection


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

Z_DIM = 16
CHALLENGE_DIM = 8
N_EPS = 15   # Episodes per game (> min_episodes_per_game=5)


def make_summary(rng, bias: np.ndarray) -> np.ndarray:
    """Create a random EpisodeSummary-like vector with a game-specific bias."""
    in_dim = 2 * Z_DIM + 7
    raw = rng.randn(in_dim).astype(np.float32) * 0.2 + bias
    return raw


def feed_game(me: MetaEncoder, game: str, n: int, bias: np.ndarray, seed: int):
    """Feed n synthetic episode summaries into MetaEncoder for a game."""
    rng = np.random.RandomState(seed)
    for _ in range(n):
        s = make_summary(rng, bias)
        me.update(game, s)


# ──────────────────────────────────────────────────────────────
# Test 1: Fresh MetaEncoder — projection_fitted is False
# ──────────────────────────────────────────────────────────────

def test_fresh_not_fitted():
    print("\n=== Test 1: Fresh MetaEncoder projection_fitted==False ===")
    me = MetaEncoder(z_dim=Z_DIM, challenge_dim=CHALLENGE_DIM)
    assert not me.projection_fitted, "Fresh MetaEncoder should not be fitted"
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 2: After fit_projection() → projection_fitted is True
# ──────────────────────────────────────────────────────────────

def test_fitted_after_fit():
    print("\n=== Test 2: projection_fitted==True after fit_projection() ===")
    in_dim = 2 * Z_DIM + 7

    # Game A: summaries centred around +1
    bias_a = np.ones(in_dim, dtype=np.float32)
    # Game B: summaries centred around -1
    bias_b = -np.ones(in_dim, dtype=np.float32)

    me = MetaEncoder(z_dim=Z_DIM, challenge_dim=CHALLENGE_DIM)
    feed_game(me, "game_a", N_EPS, bias_a, seed=1)
    feed_game(me, "game_b", N_EPS, bias_b, seed=2)

    loss = me.fit_projection(min_episodes_per_game=5, n_epochs=20, verbose=True)
    assert me.projection_fitted, "Should be fitted after fit_projection()"
    assert isinstance(loss, float) and loss >= 0.0, f"Loss should be non-negative float, got {loss}"
    print(f"  final loss: {loss:.4f}")
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 3: Same-game distance < cross-game distance
# ──────────────────────────────────────────────────────────────

def test_triplet_clustering():
    print("\n=== Test 3: Same-game closer than cross-game after fitting ===")
    in_dim = 2 * Z_DIM + 7

    # Use clearly separated biases to make triplet objective achievable
    bias_a = np.zeros(in_dim, dtype=np.float32)
    bias_a[:in_dim//2] = 2.0   # Game A has positive first half
    bias_b = np.zeros(in_dim, dtype=np.float32)
    bias_b[in_dim//2:] = 2.0   # Game B has positive second half

    me = MetaEncoder(z_dim=Z_DIM, challenge_dim=CHALLENGE_DIM)
    rng_a = np.random.RandomState(10)
    rng_b = np.random.RandomState(20)

    summaries_a = [make_summary(rng_a, bias_a) for _ in range(N_EPS)]
    summaries_b = [make_summary(rng_b, bias_b) for _ in range(N_EPS)]

    for s in summaries_a:
        me.update("game_a", s)
    for s in summaries_b:
        me.update("game_b", s)

    # Encode a fresh same-game and cross-game pair
    def cos_dist(u, v):
        return 1.0 - float(np.dot(u, v))   # both unit-normed by encode_summary

    # Baseline: before fitting
    enc_a1 = me.encode_summary(summaries_a[0])
    enc_a2 = me.encode_summary(summaries_a[1])
    enc_b  = me.encode_summary(summaries_b[0])
    d_same_before   = cos_dist(enc_a1, enc_a2)
    d_cross_before  = cos_dist(enc_a1, enc_b)
    print(f"  Before: d(same)={d_same_before:.4f}, d(cross)={d_cross_before:.4f}")

    me.fit_projection(min_episodes_per_game=5, n_epochs=40, lr=0.03)

    enc_a1 = me.encode_summary(summaries_a[0])
    enc_a2 = me.encode_summary(summaries_a[1])
    enc_b  = me.encode_summary(summaries_b[0])
    d_same_after  = cos_dist(enc_a1, enc_a2)
    d_cross_after = cos_dist(enc_a1, enc_b)
    print(f"  After:  d(same)={d_same_after:.4f}, d(cross)={d_cross_after:.4f}")

    assert d_same_after < d_cross_after, (
        f"Triplet objective failed: d(same)={d_same_after:.4f} >= d(cross)={d_cross_after:.4f}"
    )
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 4: fit_projection() with < 2 games → no-op
# ──────────────────────────────────────────────────────────────

def test_single_game_noop():
    print("\n=== Test 4: fit_projection() with < 2 games is a no-op ===")
    in_dim = 2 * Z_DIM + 7
    me = MetaEncoder(z_dim=Z_DIM, challenge_dim=CHALLENGE_DIM)
    bias = np.ones(in_dim, dtype=np.float32)
    feed_game(me, "solo_game", N_EPS, bias, seed=5)

    loss = me.fit_projection(min_episodes_per_game=5, verbose=True)
    assert not me.projection_fitted, "Should not be fitted with only 1 game"
    assert loss == 0.0, f"Expected loss=0.0 for no-op, got {loss}"
    print(f"  loss={loss}, projection_fitted={me.projection_fitted}")
    print("  PASS")


# ──────────────────────────────────────────────────────────────
# Test 5: Descriptors are finite and unit-normed after fitting
# ──────────────────────────────────────────────────────────────

def test_descriptors_valid():
    print("\n=== Test 5: Descriptors finite and unit-normed after fitting ===")
    in_dim = 2 * Z_DIM + 7

    me = MetaEncoder(z_dim=Z_DIM, challenge_dim=CHALLENGE_DIM)
    rng = np.random.RandomState(99)

    for game, offset in [("g1", 1.0), ("g2", -1.0), ("g3", 0.5)]:
        bias = np.full(in_dim, offset, dtype=np.float32)
        for _ in range(N_EPS):
            me.update(game, make_summary(rng, bias))

    me.fit_projection(min_episodes_per_game=5)

    for game in ["g1", "g2", "g3"]:
        d = me.descriptor(game)
        assert d is not None, f"Descriptor for {game!r} is None"
        assert np.isfinite(d).all(), f"Non-finite descriptor for {game!r}"
        norm = np.linalg.norm(d)
        assert abs(norm - 1.0) < 1e-4, f"Descriptor for {game!r} not unit-normed: ||d||={norm:.4f}"
        print(f"  {game!r}: ||d||={norm:.4f} ✓")
    print("  PASS")


if __name__ == "__main__":
    test_fresh_not_fitted()
    test_fitted_after_fit()
    test_triplet_clustering()
    test_single_game_noop()
    test_descriptors_valid()
    print("\n=== ALL META ENCODER TRIPLET TESTS PASSED ===")
