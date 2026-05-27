import sys; sys.path.insert(0, '.')
import numpy as np, time

print('=== VectorizedImaginedEnv benchmark ===')

from src.cell.vec_imagined_env import VectorizedImaginedEnv, benchmark_vs_real
from src.cell.world_model import CellWorldModel

# Correctness check
wm = CellWorldModel(feature_dim=16, n_actions=8, hidden_size=64, min_transitions=2, batch_size=4)
states  = np.random.randn(12, 16).astype('float32')
actions = np.random.randint(0, 8, 12)

ns_batch, rew_batch = wm.predict_batch(states, actions)
ns_single, _ = wm.predict(states[0], int(actions[0]))
err = float(np.mean(abs(ns_batch[0] - ns_single)))
print(f'predict_batch[0] vs predict(): diff={err:.7f}  (pass if <1e-5)')
assert err < 1e-5

# SPS table
print('\nN_envs   imagined_sps   ms/step')
for n in [1, 4, 12, 32, 64, 128]:
    r = benchmark_vs_real(n_envs=n, z_dim=16, n_actions=8, n_steps=500)
    print(f"{n:6d}   {r['imagined_sps']:>12,}   {r['ms_per_step']:>7.3f}ms")

# Real sim reference
from src.games.mario.mario_simulator import MarioSimulator
from src.games.mario.level_sampler import LevelSampler
sampler = LevelSampler(seed=42)
sim = MarioSimulator(sampler.sample()[0])
obs = sim.reset()
t0 = time.perf_counter()
for _ in range(1000):
    obs, r, done, _ = sim.step(1)
    if done: obs = sim.reset()
real_sps = int(1000 / (time.perf_counter() - t0))

print(f'\nReal ASCII sim (1 env):   {real_sps:>10,} sps')
r12  = benchmark_vs_real(n_envs=12,  z_dim=16, n_actions=8, n_steps=1000)
r64  = benchmark_vs_real(n_envs=64,  z_dim=16, n_actions=8, n_steps=1000)
r256 = benchmark_vs_real(n_envs=256, z_dim=16, n_actions=8, n_steps=500)
print(f'WM imagined (12 envs):    {r12["imagined_sps"]:>10,} sps')
print(f'WM imagined (64 envs):    {r64["imagined_sps"]:>10,} sps')
print(f'WM imagined (256 envs):   {r256["imagined_sps"]:>10,} sps')
print('\nDONE')
