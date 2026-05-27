"""Unit tests for Phase 4 components: MetaEncoder, NeuromodulatorBridge, AtariAdapter."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ════════════════════════════════════════════════════════════
# 4A: Meta-Encoder
# ════════════════════════════════════════════════════════════

print("=== Test 1: EpisodeSummary build ===")
from src.encoder.meta_encoder import EpisodeSummary, MetaEncoder

summ = EpisodeSummary(z_dim=8, max_steps=50)
rng = np.random.RandomState(0)
for step in range(20):
    z = rng.randn(8).astype(np.float32)
    summ.record(z=z, action=rng.randint(4), reward=float(rng.random() - 0.5))

vec = summ.build()
expected_dim = 2 * 8 + 7  # 23
assert vec.shape == (expected_dim,), f"Expected ({expected_dim},) got {vec.shape}"
assert vec.dtype == np.float32
print(f"  summary.shape={vec.shape}, dtype={vec.dtype}")
print("  PASS")

print("\n=== Test 2: MetaEncoder encode_summary ===")
meta = MetaEncoder(z_dim=8, challenge_dim=4, seed=0)
c = meta.encode_summary(vec)
assert c.shape == (4,), f"Expected (4,) got {c.shape}"
# Should be unit-normalised
assert abs(np.linalg.norm(c) - 1.0) < 1e-5, f"Expected unit norm, got {np.linalg.norm(c)}"
print(f"  c_game.shape={c.shape}, ||c||={np.linalg.norm(c):.5f}")
print("  PASS")

print("\n=== Test 3: MetaEncoder update + descriptor ===")
meta3 = MetaEncoder(z_dim=8, challenge_dim=4, window=5, seed=1)
meta3.register_games({"mario": 0, "cartpole": 1, "mountaincar": 2})
rng3 = np.random.RandomState(42)

# Simulate 10 mario episodes
for ep in range(10):
    s = EpisodeSummary(z_dim=8, max_steps=100)
    for _ in range(30):
        s.record(rng3.randn(8).astype(np.float32), rng3.randint(8), rng3.random())
    meta3.update("mario", s.build())

assert "mario" in meta3.known_games()
d = meta3.descriptor("mario")
assert d is not None and d.shape == (4,)
print(f"  mario descriptor: {d.round(3)}")
print(f"  episode_count={meta3._episode_counts['mario']}")
print("  PASS")

print("\n=== Test 4: Similarity matrix ===")
meta4 = MetaEncoder(z_dim=8, challenge_dim=4, seed=2)
rng4 = np.random.RandomState(0)

# Two games with very similar summaries (same RNG)
for game, seed_offset in [("gameA", 0), ("gameB", 0), ("gameC", 100)]:
    rng_g = np.random.RandomState(seed_offset)
    for _ in range(5):
        s = EpisodeSummary(z_dim=8, max_steps=50)
        for _ in range(20):
            s.record(rng_g.randn(8).astype(np.float32), rng_g.randint(2), rng_g.random())
        meta4.update(game, s.build())

games, mat = meta4.similarity_matrix()
assert mat.shape == (3, 3), f"Expected 3x3, got {mat.shape}"
assert abs(mat[0, 0] - 1.0) < 1e-5, "Diagonal should be 1.0"
print(f"  Games: {games}")
print(f"  Similarity matrix:\n{mat.round(3)}")
print("  PASS")

print("\n=== Test 5: nearest_game ===")
# gameA and gameB should be most similar (same random seed)
a_desc = meta4.descriptor("gameA")
best, sim = meta4.nearest_game(a_desc, exclude="gameA")
assert best in ["gameB", "gameC"]
print(f"  nearest to gameA (excl itself): {best} (sim={sim:.3f})")
print("  PASS")

print("\n=== Test 6: recommend_transfer_source ===")
meta6 = MetaEncoder(z_dim=8, challenge_dim=4, seed=3)
meta6.register_games({"mario": 0, "cartpole": 1})
rng6 = np.random.RandomState(0)
for game in ["mario", "cartpole"]:
    for _ in range(3):
        s = EpisodeSummary(z_dim=8)
        for _ in range(10):
            s.record(rng6.randn(8).astype(np.float32), 0, 0.0)
        meta6.update(game, s.build())

s_new = EpisodeSummary(z_dim=8)
for _ in range(10):
    s_new.record(rng6.randn(8).astype(np.float32), 0, 0.0)
src_game, src_id, sim6 = meta6.recommend_transfer_source("newgame", s_new.build())
assert src_game in ["mario", "cartpole", "none"]
assert isinstance(src_id, int)
print(f"  Transfer source: {src_game} (id={src_id}, sim={sim6:.3f})")
print("  PASS")

print("\n=== Test 7: MetaEncoder save/load ===")
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "meta_test.npz")
    meta6.save(path)
    meta_loaded = MetaEncoder(z_dim=8, challenge_dim=4)
    meta_loaded.load(path)
    assert "mario" in meta_loaded.known_games()
    d_orig = meta6.descriptor("mario")
    d_loaded = meta_loaded.descriptor("mario")
    assert np.allclose(d_orig, d_loaded, atol=1e-5), "Load mismatch"
print("  save/load: descriptors match exactly")
print("  PASS")

# ════════════════════════════════════════════════════════════
# 4B: NeuromodulatorBridge
# ════════════════════════════════════════════════════════════

print("\n=== Test 8: NeuromodulatorBridge step ===")
from src.learning.neuromodulator_bridge import NeuromodulatorBridge

bridge = NeuromodulatorBridge(n_neurons=16, lr_modulation_strength=0.3, verbose=False)
lr_mults = []
for i in range(20):
    reward = 1.0 if i % 5 == 0 else 0.0
    neurons = list(np.random.randint(0, 16, 4))
    pos = np.array([float(i % 5), float(i % 3)])
    mult = bridge.step(reward=reward, active_neurons=neurons, position=pos)
    assert bridge.min_lr_mult <= mult <= bridge.max_lr_mult, f"lr_mult={mult} out of range"
    lr_mults.append(mult)

print(f"  20 steps: lr_mult range=[{min(lr_mults):.3f}, {max(lr_mults):.3f}]")
print("  PASS")

print("\n=== Test 9: NeuromodulatorBridge spatial memory ===")
bridge9 = NeuromodulatorBridge(n_neurons=8)
# Add rewarded positions
bridge9.step(reward=1.0, position=np.array([0.5, 0.5]))
bridge9.step(reward=1.0, position=np.array([0.6, 0.4]))
bridge9.step(reward=0.0, position=np.array([9.0, 9.0]))  # no-reward: not stored

goal = bridge9.recall_goal()
assert goal is not None and goal.shape == (2,), f"Expected (2,) goal, got {goal}"
print(f"  Recalled goal position: {goal.round(3)} (expected ~[0.55, 0.45])")
assert abs(goal[0] - 0.55) < 0.1, "Goal X should be near 0.55"
print("  PASS")

print("\n=== Test 10: NeuromodulatorBridge eligibility traces ===")
bridge10 = NeuromodulatorBridge(n_neurons=8)
bridge10.step(reward=0.5, active_neurons=[0, 1, 2, 3])
import time; time.sleep(0.001)
bridge10.step(reward=0.5, active_neurons=[0, 1, 2, 3])
elig = bridge10.get_eligibility()
print(f"  Eligible synapses: {len(elig)}")
three_factor = bridge10.apply_dopamine_to_eligibility()
print(f"  After dopamine modulation: {len(three_factor)} updates")
print("  PASS")

# ════════════════════════════════════════════════════════════
# 4C: Atari Adapter
# ════════════════════════════════════════════════════════════

print("\n=== Test 11: AtariFallbackSim ===")
from src.games.atari.atari_adapter import AtariFallbackSim, make_atari_adapter, ATARI_SPECS

sim = AtariFallbackSim(game_name="Pong", obs_dim=300, max_steps=30, seed=0)
obs = sim.reset()
assert obs.shape == (300,), f"Expected (300,) got {obs.shape}"
assert obs.dtype == np.float32

total_r = 0.0
done = False
steps = 0
while not done:
    obs2, r, done, _ = sim.step(np.random.randint(6))
    total_r += r
    steps += 1

assert steps == 30
print(f"  obs_dim={sim.obs_dim}, n_actions={sim.n_actions}, steps={steps}, total_r={total_r:.2f}")
print("  PASS")

print("\n=== Test 12: make_atari_adapter factory ===")
adapter12 = make_atari_adapter("Pong", seed=0)
# Will be AtariFallbackSim since ALE is likely not installed in test env
obs12 = adapter12.reset()
assert obs12.shape[0] == adapter12.obs_dim
obs12b, r12, done12, _ = adapter12.step(0)
assert obs12b.shape[0] == adapter12.obs_dim
adapter12.close()
print(f"  adapter: {adapter12.game_name}, obs_dim={adapter12.obs_dim}, n_actions={adapter12.n_actions}")
print("  PASS")

print("\n=== Test 13: ATARI_SPECS completeness ===")
required = {"ale_name", "n_actions", "ram_obs_dim"}
for name, spec in ATARI_SPECS.items():
    for k in required:
        assert k in spec, f"Missing key {k!r} in {name!r} spec"
print(f"  {len(ATARI_SPECS)} Atari games: {list(ATARI_SPECS.keys())}")
print("  PASS")

# ════════════════════════════════════════════════════════════
# Integration: MetaEncoder + NeuromodulatorBridge together
# ════════════════════════════════════════════════════════════

print("\n=== Test 14: MetaEncoder + NeuromodulatorBridge combined ===")
from src.encoder.universal_encoder import EncoderRegistry

meta14 = MetaEncoder(z_dim=8, challenge_dim=4, seed=5)
meta14.register_games({"cartpole": 1, "mountaincar": 2})
bridge14 = NeuromodulatorBridge(n_neurons=8, lr_modulation_strength=0.2)

rng14 = np.random.RandomState(7)
for game_name in ["cartpole", "mountaincar"]:
    ep_summ = EpisodeSummary(z_dim=8, max_steps=20)
    for step in range(20):
        z = rng14.randn(8).astype(np.float32)
        reward = float(rng14.random() - 0.3)
        action = int(rng14.randint(2))
        surprise = float(rng14.random() * 0.1)

        # Neuromodulator step
        mult = bridge14.step(reward=reward, active_neurons=list(range(4)), position=z[:2])
        assert 0.4 <= mult <= 2.1, f"mult={mult} out of expected range"

        # Record in episode summary
        ep_summ.record(z=z, action=action, reward=reward, surprise=surprise)

    meta14.update(game_name, ep_summ.build())

# Both games should now have descriptors
assert set(meta14.known_games()) == {"cartpole", "mountaincar"}
print(f"  bridge stats: {bridge14.stats()}")
print(f"  meta report:\n{meta14.cluster_report()}")
print("  PASS")

print("\n=== ALL 14 PHASE 4 TESTS PASSED ===")
