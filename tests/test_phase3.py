"""Unit tests for Phase 3 MuJoCo bridge components."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.games.mujoco.mujoco_adapter import (
    MuJoCoFallbackSim, make_mujoco_adapter, TASK_SPECS
)
from src.games.mujoco.mujoco_action_discretizer import MuJoCoActionDiscretizer

# ════════════════════════════════════════════════════════════
# 3A: MuJoCoFallbackSim
# ════════════════════════════════════════════════════════════

print("=== Test 1: FallbackSim reset ===")
sim = MuJoCoFallbackSim(n_joints=2, obs_dim=10, seed=0)
obs = sim.reset()
assert obs.shape == (10,), f"Expected (10,) got {obs.shape}"
assert obs.dtype == np.float32
print(f"  obs.shape={obs.shape}, dtype={obs.dtype}")
print("  PASS")

print("\n=== Test 2: FallbackSim step ===")
sim2 = MuJoCoFallbackSim(n_joints=2, obs_dim=10, seed=0)
obs2 = sim2.reset()
for action in range(9):  # all 9 ternary actions
    obs3, rew, done, info = sim2.step(action)
    assert obs3.shape == (10,)
    assert isinstance(rew, float)
    assert "distance" in info
print(f"  All 9 actions stepped. Final reward={rew:.4f}, distance={info['distance']:.4f}")
print("  PASS")

print("\n=== Test 3: FallbackSim episode to done ===")
sim3 = MuJoCoFallbackSim(n_joints=2, obs_dim=10, max_steps=20, seed=1)
obs3 = sim3.reset()
done3 = False
total_r = 0.0
steps3 = 0
while not done3:
    a = np.random.randint(9)
    obs3, r, done3, info3 = sim3.step(a)
    total_r += r
    steps3 += 1
assert steps3 == 20, f"Expected 20 steps, got {steps3}"
print(f"  Episode ended at step {steps3}, total_reward={total_r:.3f}")
print("  PASS")

print("\n=== Test 4: make_mujoco_adapter — fallback ===")
adapter = make_mujoco_adapter("Reacher-v4", seed=42)
# Will be MuJoCoFallbackSim if mujoco not installed
obs4 = adapter.reset()
assert obs4.shape[0] == adapter.obs_dim
obs4b, r4, done4, _ = adapter.step(0)
assert obs4b.shape[0] == adapter.obs_dim
print(f"  adapter={adapter.env_name}, obs_dim={adapter.obs_dim}, n_actions={adapter.n_actions}")
print("  PASS")

print("\n=== Test 5: TASK_SPECS completeness ===")
required_keys = {"n_joints", "action_dim", "pro_obs_dim"}
for env_name, spec in TASK_SPECS.items():
    for k in required_keys:
        assert k in spec, f"Missing key {k!r} in spec for {env_name!r}"
print(f"  {len(TASK_SPECS)} environments in TASK_SPECS: {list(TASK_SPECS.keys())}")
print("  PASS")

# ════════════════════════════════════════════════════════════
# 3B: MuJoCoActionDiscretizer
# ════════════════════════════════════════════════════════════

print("\n=== Test 6: Ternary discretizer (2 joints) ===")
disc = MuJoCoActionDiscretizer(n_joints=2, strategy="ternary", scale=1.0)
assert disc.n_actions == 9, f"Expected 9, got {disc.n_actions}"
for i in range(9):
    a = disc.decode(i)
    assert a.shape == (2,), f"Expected (2,) got {a.shape}"
    assert all(v in (-1.0, 0.0, 1.0) for v in a.tolist()), f"Non-ternary values: {a}"
print(f"  n_actions={disc.n_actions}, all actions in {{-1,0,+1}}")
print("  PASS")

print("\n=== Test 7: Ternary discretizer (6 joints — falls back to primitive) ===")
disc6 = MuJoCoActionDiscretizer(n_joints=6, strategy="ternary", scale=1.0)
# 3^6 = 729 — should fall to primitive automatically
print(f"  disc6.n_actions={disc6.n_actions} (primitive fallback for 6 joints: 3+2×6+4={3+2*6+4})")
assert disc6.n_actions < 729, "Should use primitive fallback for 6 joints"
print("  PASS")

print("\n=== Test 8: Primitive discretizer ===")
disc_prim = MuJoCoActionDiscretizer(n_joints=3, action_dim=3, strategy="primitive")
n = disc_prim.n_actions
assert n > 3, f"Expected >3 actions for primitives, got {n}"
a_freeze = disc_prim.decode(0)
assert np.all(a_freeze == 0), f"Action 0 (freeze) should be all zeros, got {a_freeze}"
print(f"  n_actions={n}, freeze action={a_freeze}")
print("  PASS")

print("\n=== Test 9: encode/decode round-trip ===")
disc_rt = MuJoCoActionDiscretizer(n_joints=2, strategy="ternary")
for idx in range(disc_rt.n_actions):
    cont = disc_rt.decode(idx)
    recovered = disc_rt.encode(cont)
    assert recovered == idx, f"Round-trip failed: {idx} → {cont} → {recovered}"
print(f"  All {disc_rt.n_actions} actions round-tripped correctly")
print("  PASS")

print("\n=== Test 10: K-means discretizer fit ===")
disc_km = MuJoCoActionDiscretizer(n_joints=3, action_dim=3, strategy="kmeans", k=8, seed=0)
# Generate random "demo" actions
demo_actions = np.random.randn(100, 3).astype(np.float32) * 0.5
disc_km.fit(demo_actions, n_iters=20)
assert disc_km.n_actions == 8, f"Expected 8 k-means actions, got {disc_km.n_actions}"
assert disc_km._is_fitted
# decode should be within demo action space
for i in range(8):
    a = disc_km.decode(i)
    assert a.shape == (3,)
print(f"  K-means fitted: {disc_km.n_actions} clusters, is_fitted={disc_km._is_fitted}")
print("  PASS")

# ════════════════════════════════════════════════════════════
# 3C: Training script smoke test
# ════════════════════════════════════════════════════════════

print("\n=== Test 11: mujoco_training.train() smoke test ===")
from examples.mujoco_training import train, SimpleQAgent

result = train(
    env_name="Reacher-v4",
    views=["xy", "xz", "yz"],
    n_episodes=5,
    max_steps=20,
    use_visual=False,
    verbose=False,
    seed=0,
)
assert "final_success_rate" in result
assert "final_avg_reward" in result
assert "obs_dim" in result
print(f"  5 episodes: SR={result['final_success_rate']:.1f}%, "
      f"avg_r={result['final_avg_reward']:+.3f}, obs_dim={result['obs_dim']}")
print("  PASS")

print("\n=== Test 12: run_ablation() smoke test ===")
from examples.mujoco_training import run_ablation

ablation_results = run_ablation(
    env_name="Reacher-v4",
    n_episodes=3,
    max_steps=10,
    seed=0,
    verbose=False,
)
assert set(ablation_results.keys()) == {"xy_only", "xy_xz", "all_views"}
for k, v in ablation_results.items():
    assert "obs_dim" in v
    assert "final_success_rate" in v
print(f"  Ablation: " + " | ".join(f"{k}: obs_dim={v['obs_dim']}" for k,v in ablation_results.items()))
print("  PASS")

print("\n=== ALL 12 PHASE 3 TESTS PASSED ===")
