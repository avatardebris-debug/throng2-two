"""Unit tests for mario_hpo.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.games.mario.mario_hpo import (
    MarioParameterSpace, MarioHPOObjective, agent_from_config,
)

ps = MarioParameterSpace()

# --- Test 1: Parameter space round-trip ---
print("=== Test 1: Parameter space round-trip ===")
default = ps.get_default_config()
arr = ps.to_array(default)
recovered = ps.from_array(arr)
for name in default:
    param_def = next(p for p in ps.PARAMS if p[0] == name)
    ptype = param_def[3]
    if ptype == "int":
        assert int(default[name]) == int(recovered[name]), f"{name}: {default[name]} != {recovered[name]}"
    else:
        # Allow 1% tolerance for log-scale
        assert abs(default[name] - recovered[name]) / max(abs(default[name]), 1e-9) < 0.02, \
            f"{name}: {default[name]} != {recovered[name]}"
print(f"  Default config ({ps.count_parameters()} params) round-trips correctly")
print("  PASS")

# --- Test 2: Random sampling stays in bounds ---
print("\n=== Test 2: Random sampling bounds ===")
for _ in range(20):
    cfg = ps.sample_random()
    for name, low, high, ptype, _, _ in ps.PARAMS:
        v = cfg[name]
        assert low <= v <= high * 1.001, f"{name}={v} out of [{low},{high}]"
print(f"  20 random configs all within bounds")
print("  PASS")

# --- Test 3: Array length matches param count ---
print("\n=== Test 3: Array dimension ===")
arr = ps.to_array(ps.sample_random())
assert len(arr) == ps.count_parameters(), f"Array len {len(arr)} != {ps.count_parameters()}"
print(f"  Array dim = {len(arr)}")
print("  PASS")

# --- Test 4: Objective runs and returns scalar ---
print("\n=== Test 4: Objective function ===")
obj = MarioHPOObjective(tier=2, eval_episodes=5, max_steps=80, seed=42, verbose=True)
config = ps.get_default_config()
score = obj(config)
print(f"  score = {score:.4f}")
assert isinstance(score, float), f"Expected float, got {type(score)}"
assert len(obj.history) == 1
print("  PASS")

# --- Test 5: Score is finite and reasonable ---
print("\n=== Test 5: Score sanity ===")
assert np.isfinite(score), "Score should be finite"
# Score can be negative (dying fast) to >6 (perfect wins), just check it's real
assert -5.0 < score < 10.0, f"Score {score} out of reasonable range"
print(f"  Score {score:.4f} is finite and in [-5, 10]")
print("  PASS")

# --- Test 6: agent_from_config builds an agent ---
print("\n=== Test 6: agent_from_config ===")
agent = agent_from_config(config)
import numpy as np
dummy_obs = np.zeros(378, dtype=np.float32)
action = agent.step(dummy_obs)
assert 0 <= action < 8, f"Invalid action {action}"
print(f"  Agent built, first action = {action}")
print("  PASS")

# --- Test 7: pretty() produces readable output ---
print("\n=== Test 7: pretty() output ===")
s = ps.pretty(config)
print(s)
assert "lr=" in s
assert "gamma=" in s
print("  PASS")

print("\n=== ALL 7 TESTS PASSED ===")
