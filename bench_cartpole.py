"""
CartPole Benchmark — CalibrationScheduler wired into the imagination loop.

Three agents, same wall-clock budget:
  Agent A: Real CartPole only (standard RL baseline)
  Agent B: Imagination only, NO calibration (frozen WM)  
  Agent C: Imagination WITH CalibrationScheduler (WM updates from reality)

This proves:
  1. Speed advantage of imagination (sps comparison)
  2. CalibrationScheduler keeps the WM honest (B vs C quality)
  3. Overall: can imagination + calibration beat real-only?
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import time
import gymnasium as gym

from src.cell.world_model import CellWorldModel
from src.cell.vec_imagined_env import VectorizedImaginedEnv
from src.learning.calibration_scheduler import SyncCalibrator
from src.learning.consequence_cascade import ConsequenceCascadeBuffer

Z_DIM = 4  # CartPole obs is 4-dim (cart_pos, cart_vel, pole_angle, pole_vel)
N_ACT = 2
N_ENVS = 32

# ═══════════════════════════════════════════════════════
#  Numpy Q-network (same for all agents)
# ═══════════════════════════════════════════════════════

class NumpyQNet:
    def __init__(self, z_dim=4, n_act=2, hidden=64, lr=0.003):
        self.n_act = n_act; self.lr = lr; self.gamma = 0.99; self.eps = 0.10
        s = np.sqrt(2/(z_dim+hidden))
        self.W1 = np.random.randn(hidden, z_dim).astype('f') * s
        self.b1 = np.zeros(hidden, dtype='f')
        s2 = np.sqrt(2/(hidden+n_act))
        self.W2 = np.random.randn(n_act, hidden).astype('f') * s2
        self.b2 = np.zeros(n_act, dtype='f')

    def _fwd(self, o):
        h = np.maximum(0, o @ self.W1.T + self.b1)
        return h @ self.W2.T + self.b2

    def act(self, o):
        if np.random.rand() < self.eps: return np.random.randint(self.n_act)
        return int(np.argmax(self._fwd(o)))

    def act_batch(self, o):
        m = np.random.rand(len(o)) < self.eps
        q = self._fwd(o); a = np.argmax(q, 1)
        a[m] = np.random.randint(0, self.n_act, m.sum()); return a

    def update(self, obs, act, rew, nobs, done):
        N = len(obs)
        if N == 0: return
        h = np.maximum(0, obs @ self.W1.T + self.b1)
        q = h @ self.W2.T + self.b2
        hn = np.maximum(0, nobs @ self.W1.T + self.b1)
        qn = hn @ self.W2.T + self.b2
        tgt = rew + self.gamma * np.max(qn, 1) * (1 - done.astype('f'))
        td = tgt - q[np.arange(N), act.astype(int)]
        dq = np.zeros_like(q)
        dq[np.arange(N), act.astype(int)] = -td * (2.0/N)
        dW2 = dq.T @ h; db2 = dq.sum(0)
        dh = dq @ self.W2; dh *= (h > 0).astype('f')
        dW1 = dh.T @ obs; db1 = dh.sum(0)
        self.W2 -= np.clip(self.lr*dW2, -0.05, 0.05)
        self.b2 -= np.clip(self.lr*db2, -0.05, 0.05)
        self.W1 -= np.clip(self.lr*dW1, -0.05, 0.05)
        self.b1 -= np.clip(self.lr*db1, -0.05, 0.05)

def evaluate(agent, n_ep=20):
    """Evaluate on REAL CartPole (no delay, no imagination)."""
    env = gym.make('CartPole-v1')
    total_r = 0
    for _ in range(n_ep):
        obs, _ = env.reset()
        ep_r = 0
        for _ in range(500):
            a = agent.act(obs.astype('f'))
            obs, r, term, trunc, _ = env.step(a)
            ep_r += r
            if term or trunc: break
        total_r += ep_r
    env.close()
    return total_r / n_ep

# ═══════════════════════════════════════════════════════
#  Phase 0: Pre-calibrate a WM on CartPole
# ═══════════════════════════════════════════════════════
print("=" * 65)
print("  CartPole Benchmark: Real vs Imagination vs Imagination+Cal")
print("=" * 65)

print("\n--- Phase 0: WM Calibration (shared, not timed) ---")
wm = CellWorldModel(feature_dim=Z_DIM, n_actions=N_ACT,
                     hidden_size=64, min_transitions=10, batch_size=16)
cascade = ConsequenceCascadeBuffer(capacity=10000, min_magnitude=5.0)

env_cal = gym.make('CartPole-v1')
obs_cal, _ = env_cal.reset()
obs_cal = obs_cal.astype('f')
cal_steps = 0
cal_episodes = 0

cascade.begin_episode()
while cal_steps < 2000:
    a = np.random.randint(N_ACT)
    nobs, r, term, trunc, _ = env_cal.step(a)
    nobs = nobs.astype('f')
    done = term or trunc
    wm.store_transition(obs_cal, a, nobs, r)
    wm.train_step()
    cascade.store_step(obs_cal, a, nobs, r)
    cal_steps += 1

    if done:
        # Mark the terminal step as consequential (pole fell = important)
        cascade.mark_consequential(
            step=len(cascade._current_episode)-1,
            magnitude=5.0, tag="terminal"
        )
        cascade.end_episode()
        cascade.begin_episode()
        obs_cal, _ = env_cal.reset()
        obs_cal = obs_cal.astype('f')
        cal_episodes += 1
    else:
        obs_cal = nobs

cascade.end_episode()
env_cal.close()

# Extra WM training on consequence-weighted data
for _ in range(100):
    batch = cascade.sample_as_tuples(min(16, len(cascade)))
    for s, a, ns, r in batch:
        wm.store_transition(s, a, ns, r)
    wm.train_step()

# Test WM accuracy
env_test = gym.make('CartPole-v1')
obs_t, _ = env_test.reset()
errs = []
for _ in range(200):
    a = np.random.randint(N_ACT)
    real_n, real_r, term, trunc, _ = env_test.step(a)
    pred_n, pred_r = wm.predict(obs_t.astype('f'), a)
    errs.append(np.mean(np.abs(pred_n - real_n.astype('f'))))
    obs_t = real_n if not (term or trunc) else env_test.reset()[0]
env_test.close()

print(f"  Calibration: {cal_steps} steps, {cal_episodes} episodes")
print(f"  WM confidence: {wm.confidence:.3f}")
print(f"  WM state prediction error: {np.mean(errs):.4f}")
print(f"  Cascade: {len(cascade)} transitions, {cascade.stats()['n_consequences']} consequences")

# Clone WM for agent C (so B and C start from same weights)
import copy
wm_b = copy.deepcopy(wm)  # Agent B: frozen
wm_c = copy.deepcopy(wm)  # Agent C: calibration-updated

TIME_BUDGET = 5.0
print(f"\n--- Race: {TIME_BUDGET}s wall-clock budget ---")

# ═══════════════════════════════════════════════════════
#  Agent A: Real CartPole only
# ═══════════════════════════════════════════════════════
print(f"\n  Agent A: Real CartPole")
agent_a = NumpyQNet()
env_a = gym.make('CartPole-v1')
obs_a, _ = env_a.reset()
obs_a = obs_a.astype('f')
real_steps = 0
buf_o, buf_a, buf_r, buf_no, buf_d = [], [], [], [], []
ckpts_a = []

t0 = time.perf_counter()
while time.perf_counter() - t0 < TIME_BUDGET:
    a = agent_a.act(obs_a)
    no, r, term, trunc, _ = env_a.step(a)
    no = no.astype('f')
    done = term or trunc
    buf_o.append(obs_a); buf_a.append(a); buf_r.append(r)
    buf_no.append(no); buf_d.append(done)
    obs_a = no if not done else env_a.reset()[0].astype('f')
    real_steps += 1

    if len(buf_o) >= 64:
        agent_a.update(np.array(buf_o), np.array(buf_a), np.array(buf_r),
                       np.array(buf_no), np.array(buf_d))
        buf_o, buf_a, buf_r, buf_no, buf_d = [], [], [], [], []

    el = time.perf_counter() - t0
    if len(ckpts_a) < int(el / 1.0) + 1:
        ckpts_a.append((el, real_steps, evaluate(agent_a, 10)))

elapsed_a = time.perf_counter() - t0
env_a.close()
final_a = evaluate(agent_a, 30)
sps_a = real_steps / elapsed_a
print(f"    {real_steps:,} steps, {sps_a:,.0f} sps, avg_reward: {final_a:.0f}/500")

# ═══════════════════════════════════════════════════════
#  Agent B: Imagination only, NO calibration (frozen WM)
# ═══════════════════════════════════════════════════════
print(f"\n  Agent B: Imagination (frozen WM, no calibration)")
agent_b = NumpyQNet()
vec_b = VectorizedImaginedEnv(wm_b, n_envs=N_ENVS, z_dim=Z_DIM)
inits_b = np.random.randn(N_ENVS, Z_DIM).astype('f') * 0.05
vec_b.reset(inits_b)

# Seed with real CartPole initial states
env_seed = gym.make('CartPole-v1')
for i in range(N_ENVS):
    inits_b[i] = env_seed.reset()[0].astype('f')
env_seed.close()

imag_obs_b = vec_b.reset(inits_b)
imag_steps_b = 0
ep_c_b = np.zeros(N_ENVS, dtype=int)
ckpts_b = []

t0 = time.perf_counter()
while time.perf_counter() - t0 < TIME_BUDGET:
    actions = agent_b.act_batch(imag_obs_b)
    nobs, rews, dones = vec_b.step(actions)
    resets = (ep_c_b >= 499)
    agent_b.update(imag_obs_b, actions, rews, nobs, resets.astype('f'))
    imag_obs_b = nobs
    imag_steps_b += N_ENVS
    ep_c_b += 1

    for j in np.where(resets)[0]:
        imag_obs_b[j] = np.random.randn(Z_DIM).astype('f') * 0.05
        ep_c_b[j] = 0

    el = time.perf_counter() - t0
    if len(ckpts_b) < int(el / 1.0) + 1:
        ckpts_b.append((el, imag_steps_b, evaluate(agent_b, 10)))

elapsed_b = time.perf_counter() - t0
final_b = evaluate(agent_b, 30)
sps_b = imag_steps_b / elapsed_b
print(f"    {imag_steps_b:,} steps, {sps_b:,.0f} sps, avg_reward: {final_b:.0f}/500")

# ═══════════════════════════════════════════════════════
#  Agent C: Imagination WITH CalibrationScheduler
# ═══════════════════════════════════════════════════════
print(f"\n  Agent C: Imagination + CalibrationScheduler")
agent_c = NumpyQNet()
vec_c = VectorizedImaginedEnv(wm_c, n_envs=N_ENVS, z_dim=Z_DIM)

env_seed2 = gym.make('CartPole-v1')
inits_c = np.zeros((N_ENVS, Z_DIM), dtype='f')
for i in range(N_ENVS):
    inits_c[i] = env_seed2.reset()[0].astype('f')
env_seed2.close()

imag_obs_c = vec_c.reset(inits_c)

# Wire up SyncCalibrator (shares the wm_c object — updates propagate)
cal_env = gym.make('CartPole-v1')

class GymAdapter:
    """Adapts gymnasium API to old-style for calibrator."""
    def __init__(self, env):
        self._env = env
    def reset(self):
        obs, _ = self._env.reset()
        return obs.astype('f')
    def step(self, action):
        obs, r, term, trunc, info = self._env.step(action)
        return obs.astype('f'), float(r), term or trunc, info

calibrator = SyncCalibrator(
    world_model=wm_c,
    real_env=GymAdapter(cal_env),
    encode_fn=lambda o: o.astype('f'),
    n_actions=N_ACT,
    calibration_steps=64,
    tier1_threshold=0.15,
    fast_retrain_steps=20,
    fast_retrain_lr_mult=2.0,
)

imag_steps_c = 0
ep_c_c = np.zeros(N_ENVS, dtype=int)
n_calibrations = 0
ckpts_c = []
CALIBRATE_EVERY = 300  # calibrate every 300 imagination steps

t0 = time.perf_counter()
while time.perf_counter() - t0 < TIME_BUDGET:
    actions = agent_c.act_batch(imag_obs_c)
    nobs, rews, dones = vec_c.step(actions)
    resets = (ep_c_c >= 499)
    agent_c.update(imag_obs_c, actions, rews, nobs, resets.astype('f'))
    imag_obs_c = nobs
    imag_steps_c += N_ENVS
    ep_c_c += 1

    for j in np.where(resets)[0]:
        imag_obs_c[j] = np.random.randn(Z_DIM).astype('f') * 0.05
        ep_c_c[j] = 0

    # Calibration: periodically validate WM against real env
    if imag_steps_c % (CALIBRATE_EVERY * N_ENVS) < N_ENVS:
        cal_result = calibrator.calibrate()
        n_calibrations += 1

    el = time.perf_counter() - t0
    if len(ckpts_c) < int(el / 1.0) + 1:
        ckpts_c.append((el, imag_steps_c, evaluate(agent_c, 10)))

elapsed_c = time.perf_counter() - t0
cal_env.close()
final_c = evaluate(agent_c, 30)
sps_c = imag_steps_c / elapsed_c
print(f"    {imag_steps_c:,} steps, {sps_c:,.0f} sps, avg_reward: {final_c:.0f}/500")
print(f"    Calibrations: {n_calibrations}, WM confidence: {wm_c.confidence:.3f}")
print(f"    Calibrator stats: {calibrator.stats()}")

# ═══════════════════════════════════════════════════════
#  Final comparison
# ═══════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  RESULTS ({TIME_BUDGET}s)        Real     Imag(frozen)  Imag+Cal")
print(f"  Steps:            {real_steps:>10,}   {imag_steps_b:>10,}  {imag_steps_c:>10,}")
print(f"  SPS:              {sps_a:>10,.0f}   {sps_b:>10,.0f}  {sps_c:>10,.0f}")
print(f"  Avg reward:       {final_a:>10.0f}   {final_b:>10.0f}  {final_c:>10.0f}")
print(f"  vs Real speed:    {'1.0x':>10s}   {imag_steps_b/max(real_steps,1):.1f}x        {imag_steps_c/max(real_steps,1):.1f}x")

print(f"\n  Learning Curve (avg reward on REAL CartPole):")
print(f"  Time    Real   Frozen  +Cal")
for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
    ra = rb = rc = 0
    for ct, cs, cw in ckpts_a:
        if ct >= t: ra = cw; break
    else:
        if ckpts_a: ra = ckpts_a[-1][2]
    for ct, cs, cw in ckpts_b:
        if ct >= t: rb = cw; break
    else:
        if ckpts_b: rb = ckpts_b[-1][2]
    for ct, cs, cw in ckpts_c:
        if ct >= t: rc = cw; break
    else:
        if ckpts_c: rc = ckpts_c[-1][2]
    print(f"  {t:.0f}s    {ra:5.0f}   {rb:5.0f}   {rc:5.0f}")

best = max([(final_a, "Real"), (final_b, "Imag(frozen)"), (final_c, "Imag+Cal")])
print(f"\n  Winner: {best[1]} ({best[0]:.0f}/500)")
print(f"{'='*65}")
