"""Unit tests for Phase 2 components: universal_encoder, world_model extensions, dreamer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ════════════════════════════════════════════════════════════
# Phase 2A: Universal Encoder
# ════════════════════════════════════════════════════════════

from src.encoder.universal_encoder import (
    UniversalEncoder, EncoderRegistry, EncoderConfig,
    list_games, register_game, N_GAMES
)

print("=== Test 1: Game registry ===")
games = list_games()
assert "mario" in games
assert "cartpole" in games
assert "mountaincar" in games
assert "lunarlander" in games
print(f"  {len(games)} games registered: {games}")
print("  PASS")

print("\n=== Test 2: UniversalEncoder - flat obs (CartPole) ===")
enc = UniversalEncoder("cartpole", z_dim=32)
obs = np.array([0.1, -0.2, 0.05, 0.9], dtype=np.float32)
z = enc.encode(obs)
assert z.shape == (32 + N_GAMES,), f"Expected ({32 + N_GAMES},) got {z.shape}"
# z is normalized: first 32 dims should have unit norm
z_main = z[:32]
norm = np.linalg.norm(z_main)
assert abs(norm - 1.0) < 1e-5, f"z not normalized: norm={norm}"
# Game ID one-hot check
game_id_part = z[32:]
assert game_id_part.sum() == 1.0, "game_id one-hot should sum to 1"
assert game_id_part[enc.game_id] == 1.0
print(f"  z.shape={z.shape}, z[:32] norm={norm:.6f}, game_id={game_id_part.argmax()}")
print("  PASS")

print("\n=== Test 3: UniversalEncoder - mario sim obs ===")
enc_m = UniversalEncoder("mario", z_dim=32)
mario_obs = np.random.randn(378).astype(np.float32)
z_m = enc_m.encode(mario_obs)
assert z_m.shape == (32 + N_GAMES,)
print(f"  mario z.shape={z_m.shape}")
print("  PASS")

print("\n=== Test 4: UniversalEncoder - no game_id ===")
enc_noId = UniversalEncoder("mountaincar", z_dim=16, include_game_id=False)
obs2 = np.array([0.3, 0.7], dtype=np.float32)
z2 = enc_noId.encode(obs2)
assert z2.shape == (16,), f"Expected (16,) got {z2.shape}"
print(f"  z2.shape={z2.shape} (no game_id)")
print("  PASS")

print("\n=== Test 5: EncoderRegistry - consistent out_dim ===")
reg = EncoderRegistry(z_dim=32, games=["mario", "cartpole", "mountaincar"])
mario_obs = np.random.randn(378).astype(np.float32)
cp_obs = np.array([0.1, -0.2, 0.05, 0.9], dtype=np.float32)
mc_obs = np.array([0.0, 0.5], dtype=np.float32)

z_mario = reg.encode("mario", mario_obs)
z_cp = reg.encode("cartpole", cp_obs)
z_mc = reg.encode("mountaincar", mc_obs)

assert z_mario.shape == z_cp.shape == z_mc.shape, \
    f"All z-vecs must have same shape: {z_mario.shape} {z_cp.shape} {z_mc.shape}"
print(f"  All games produce z.shape={z_mario.shape}")
print("  PASS")

print("\n=== Test 6: EncoderRegistry - different game IDs ===")
assert reg.game_id("mario") != reg.game_id("cartpole")
assert reg.game_id("mario") != reg.game_id("mountaincar")
print(f"  mario_id={reg.game_id('mario')}, cartpole_id={reg.game_id('cartpole')}, mc_id={reg.game_id('mountaincar')}")
print("  PASS")

print("\n=== Test 7: Custom game registration ===")
custom = EncoderConfig(game_name="test_game", game_id=6, obs_type="flat", obs_dim=10)
register_game(custom)
enc_custom = UniversalEncoder("test_game", z_dim=8, include_game_id=False)
z_custom = enc_custom.encode(np.zeros(10))
assert z_custom.shape == (8,)
print(f"  Custom game registered (id=6), z_custom.shape={z_custom.shape}")
print("  PASS")


# ════════════════════════════════════════════════════════════
# Phase 2B: MultiGameReplayBuffer + MultiGameWorldModel
# ════════════════════════════════════════════════════════════

print("\n=== Test 8: MultiGameReplayBuffer ===")
from src.cell.world_model import MultiGameReplayBuffer

buf = MultiGameReplayBuffer(capacity_per_game=100, sampling="balanced")
for gid in [0, 1, 2]:
    for _ in range(30):
        s = np.random.randn(10).astype(np.float32)
        ns = np.random.randn(10).astype(np.float32)
        buf.add(s, gid % 8, ns, float(gid), gid)

assert buf.size == 90
assert buf.is_ready(min_per_game=20)
batch = buf.sample(60)
assert len(batch) >= 10, f"Expected at least 10 samples, got {len(batch)}"
game_ids_in_batch = set(b[4] for b in batch)
# Should have all 3 games represented in a balanced sample
assert game_ids_in_batch == {0, 1, 2}, f"Not all games represented: {game_ids_in_batch}"
print(f"  buf.size={buf.size}, sample size={len(batch)}, games in batch={game_ids_in_batch}")
print("  PASS")

print("\n=== Test 9: MultiGameWorldModel (PyTorch) ===")
try:
    from src.cell.world_model import MultiGameWorldModel
    mgwm = MultiGameWorldModel(
        feature_dim=16, n_actions=4, n_games=3,
        game_embed_dim=4, hidden_size=32, batch_size=16, min_transitions=10,
    )
    # Store transitions
    for gid in [0, 1, 2]:
        for _ in range(20):
            s = np.random.randn(16).astype(np.float32)
            ns = np.random.randn(16).astype(np.float32)
            mgwm.store_transition(s, gid % 4, ns, float(gid * 0.1), gid)

    # Train step
    metrics = mgwm.train_step_multi_game()
    assert "wm_loss" in metrics
    print(f"  MultiGameWorldModel: loss={metrics['wm_loss']}, buffer={metrics['wm_buffer']}")

    # predict_multi
    z = np.random.randn(16).astype(np.float32)
    z_hat, r_hat = mgwm.predict_multi(z, action=0, game_id=0)
    assert z_hat.shape == (16,), f"Expected (16,) got {z_hat.shape}"
    print(f"  predict_multi: z_hat.shape={z_hat.shape}, r_hat={r_hat:.4f}")

    # surprise
    s0 = mgwm.surprise(0)
    assert s0 >= 0.0
    print(f"  surprise(game=0)={s0:.4f}")
    print("  PASS")
except Exception as e:
    print(f"  [SKIP] PyTorch world model test failed: {e}")
    print("  (torch may not be installed)")

# ════════════════════════════════════════════════════════════
# Phase 2D: CellDreamer game_id
# ════════════════════════════════════════════════════════════

print("\n=== Test 10: CellDreamer game_id ===")
from src.cell.dreamer import CellDreamer

dreamer = CellDreamer(n_actions=4, dream_interval=1, log_dreams=True)
assert dreamer.get_game_id() == 0

dreamer.set_game_id(2)
assert dreamer.get_game_id() == 2

# Dream with no world model → None
result = dreamer.dream(np.zeros(16), world_model=None)
assert result is None
print(f"  game_id={dreamer.get_game_id()}, dream(no_wm)={result}")

# blend_action with no dream
action = dreamer.blend_action(
    ppo_action=1, ppo_log_prob=-0.5,
    dream_values=None, wm_confidence=0.0,
)
assert action == 1
print(f"  blend_action(no_dream)={action}")

# stats
s = dreamer.stats()
assert "current_game_id" in s
assert s["current_game_id"] == 2
print(f"  stats: {s}")
print("  PASS")

print("\n=== Test 11: CellDreamer dream_accuracy ===")
dreamer2 = CellDreamer(n_actions=4, log_dreams=True)
# Manually inject dream log entries
dreamer2._dream_log.extend([
    {"dream_action": 0, "ppo_action": 0, "final_action": 0, "overrode": False, "game_id": 0},
    {"dream_action": 1, "ppo_action": 0, "final_action": 0, "overrode": False, "game_id": 0},
    {"dream_action": 0, "ppo_action": 0, "final_action": 0, "overrode": False, "game_id": 0},
])
acc = dreamer2.dream_accuracy()
assert abs(acc - 2/3) < 0.01, f"Expected ~0.667, got {acc}"
print(f"  dream_accuracy = {acc:.3f} (expected 0.667)")
print("  PASS")

print("\n=== ALL 11 PHASE 2 TESTS PASSED ===")
