"""
Imagination Demo v4 — With Consequence-Aware WM Training

Root cause from v3: WM never learned the +10 goal reward because
only 1 of 500 calibration transitions reached the goal.

Fix: Use ConsequenceCascadeBuffer to heavily weight the rare
goal-reaching transitions during WM training.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import time

from src.cell.world_model import CellWorldModel
from src.cell.vec_imagined_env import VectorizedImaginedEnv
from src.learning.consequence_cascade import ConsequenceCascadeBuffer

Z=8; N_ACT=4; GRID=10; GOAL=(8,8)

class GridEnv:
    def __init__(self, delay_ms=0):
        self.x=self.y=0; self._d=delay_ms/1000
    def reset(self):
        self.x=np.random.randint(GRID); self.y=np.random.randint(GRID)
        return self._obs()
    def step(self,a):
        if self._d>0: time.sleep(self._d)
        if a==0 and self.y>0: self.y-=1
        elif a==1 and self.y<GRID-1: self.y+=1
        elif a==2 and self.x>0: self.x-=1
        elif a==3 and self.x<GRID-1: self.x+=1
        d=abs(self.x-GOAL[0])+abs(self.y-GOAL[1])
        done=d==0; r=-d/GRID+(10.0 if done else 0.0)
        return self._obs(), r, done, {}
    def _obs(self):
        o=np.zeros(Z,dtype='f')
        o[0]=self.x/GRID; o[1]=self.y/GRID
        o[2]=abs(self.x-GOAL[0])/GRID; o[3]=abs(self.y-GOAL[1])/GRID
        return o

class NumpyQNet:
    def __init__(self, z_dim, n_act, hidden=32, lr=0.005):
        self.n_act=n_act; self.lr=lr; self.gamma=0.95; self.eps=0.15
        s=np.sqrt(2/(z_dim+hidden))
        self.W1=np.random.randn(hidden,z_dim).astype('f')*s
        self.b1=np.zeros(hidden,dtype='f')
        s2=np.sqrt(2/(hidden+n_act))
        self.W2=np.random.randn(n_act,hidden).astype('f')*s2
        self.b2=np.zeros(n_act,dtype='f')
    def _fwd(self,o):
        h=np.maximum(0,o@self.W1.T+self.b1); return h@self.W2.T+self.b2
    def act(self,o):
        if np.random.rand()<self.eps: return np.random.randint(self.n_act)
        return int(np.argmax(self._fwd(o)))
    def act_batch(self,o):
        m=np.random.rand(len(o))<self.eps; q=self._fwd(o); a=np.argmax(q,1)
        a[m]=np.random.randint(0,self.n_act,m.sum()); return a
    def update(self,obs,act,rew,nobs,done):
        N=len(obs)
        h=np.maximum(0,obs@self.W1.T+self.b1); q=h@self.W2.T+self.b2
        hn=np.maximum(0,nobs@self.W1.T+self.b1); qn=hn@self.W2.T+self.b2
        tgt=rew+self.gamma*np.max(qn,1)*(1-done.astype('f'))
        td=tgt-q[np.arange(N),act.astype(int)]
        dq=np.zeros_like(q); dq[np.arange(N),act.astype(int)]=-td*(2.0/N)
        dW2=dq.T@h; db2=dq.sum(0); dh=dq@self.W2; dh*=(h>0).astype('f')
        dW1=dh.T@obs; db1=dh.sum(0)
        self.W2-=np.clip(self.lr*dW2,-0.1,0.1); self.b2-=np.clip(self.lr*db2,-0.1,0.1)
        self.W1-=np.clip(self.lr*dW1,-0.1,0.1); self.b1-=np.clip(self.lr*db1,-0.1,0.1)

def evaluate(agent, n_ep=50):
    env=GridEnv(); wins=0
    for _ in range(n_ep):
        o=env.reset()
        for _ in range(50):
            a=agent.act(o); o,r,d,_=env.step(a)
            if d: wins+=1; break
    return wins/n_ep

print("="*65)
print("  PART 1: WM Training — Standard vs Consequence-Weighted")
print("="*65)

# --- Standard WM (500 random steps) ---
wm_std = CellWorldModel(feature_dim=Z, n_actions=N_ACT, hidden_size=32,
                         min_transitions=5, batch_size=8)
env = GridEnv()
obs = env.reset()
n_goals_seen = 0
for i in range(500):
    a = np.random.randint(N_ACT)
    nobs, r, done, _ = env.step(a)
    wm_std.store_transition(obs, a, nobs, r)
    wm_std.train_step()
    if r > 5: n_goals_seen += 1
    obs = nobs if not done else env.reset()

# Test goal reward prediction
near_goal = np.zeros(Z, dtype='f')
near_goal[0]=7/GRID; near_goal[1]=8/GRID; near_goal[2]=1/GRID; near_goal[3]=0
_, std_pred_r = wm_std.predict(near_goal, 3)  # right → goal
print(f"\n  Standard WM (500 steps, {n_goals_seen} goal transitions seen):")
print(f"    Goal reward prediction: {std_pred_r:.3f}  (target: 10.0)")

# --- Consequence-weighted WM ---
wm_cw = CellWorldModel(feature_dim=Z, n_actions=N_ACT, hidden_size=32,
                        min_transitions=5, batch_size=8)
cascade = ConsequenceCascadeBuffer(capacity=5000, decay=0.995, min_magnitude=2.0)

env = GridEnv()
obs = env.reset()
n_goals_seen_cw = 0

cascade.begin_episode()
for i in range(500):
    a = np.random.randint(N_ACT)
    nobs, r, done, _ = env.step(a)
    wm_cw.store_transition(obs, a, nobs, r)
    wm_cw.train_step()
    cascade.store_step(obs, a, nobs, r)
    if r > 5:
        n_goals_seen_cw += 1
        cascade.mark_consequential(len(cascade._current_episode)-1, magnitude=r)
    if done:
        cascade.end_episode()
        cascade.begin_episode()
        obs = env.reset()
    else:
        obs = nobs
cascade.end_episode()

# Extra WM training on consequence-weighted transitions
print(f"\n  Consequence-weighted WM (500 steps, {n_goals_seen_cw} goal transitions):")
print(f"    Cascade buffer: {len(cascade)} transitions, {cascade.stats()['n_consequences']} consequences")

# Replay consequential transitions into WM heavily
for _ in range(200):
    batch = cascade.sample_as_tuples(min(16, len(cascade)))
    for s, a, ns, r in batch:
        wm_cw.store_transition(s, a, ns, r)
    wm_cw.train_step()

_, cw_pred_r = wm_cw.predict(near_goal, 3)
print(f"    Goal reward prediction: {cw_pred_r:.3f}  (target: 10.0)")
print(f"    Improvement: {abs(10.0-std_pred_r):.3f} → {abs(10.0-cw_pred_r):.3f}")

# ═══════════════════════════════════════════════════════
#  PART 2: Race with Consequence-Weighted WM
# ═══════════════════════════════════════════════════════
DELAY_MS = 0.5
TIME_BUDGET = 3.0
N_ENVS = 64

print(f"\n{'='*65}")
print(f"  PART 2: Learning Race (env delay={DELAY_MS}ms, budget={TIME_BUDGET}s)")
print(f"{'='*65}")

# Agent A: Real env
print(f"\n--- Agent A: Real Environment ---")
agent_a = NumpyQNet(Z, N_ACT)
envs_a = [GridEnv(delay_ms=DELAY_MS) for _ in range(12)]
obs_a = [e.reset() for e in envs_a]
real_steps = 0
buf_o,buf_a,buf_r,buf_no,buf_d = [],[],[],[],[]
ckpts_a = []

t0 = time.perf_counter()
while time.perf_counter()-t0 < TIME_BUDGET:
    for i in range(12):
        a = agent_a.act(obs_a[i])
        no,r,d,_ = envs_a[i].step(a)
        buf_o.append(obs_a[i]); buf_a.append(a); buf_r.append(r)
        buf_no.append(no); buf_d.append(d)
        obs_a[i] = no if not d else envs_a[i].reset()
        real_steps += 1
    if len(buf_o) >= 32:
        agent_a.update(np.array(buf_o),np.array(buf_a),np.array(buf_r),
                      np.array(buf_no),np.array(buf_d))
        buf_o,buf_a,buf_r,buf_no,buf_d = [],[],[],[],[]
    el = time.perf_counter()-t0
    if len(ckpts_a) < int(el/0.5)+1:
        ckpts_a.append((el, real_steps, evaluate(agent_a, 20)))

elapsed_a = time.perf_counter()-t0
final_a = evaluate(agent_a, 100)
print(f"  {real_steps:,} steps, {real_steps/elapsed_a:,.0f} sps, {final_a*100:.0f}% win")

# Agent B: Imagination with consequence-weighted WM
print(f"\n--- Agent B: Imagination (consequence-weighted WM) ---")
agent_b = NumpyQNet(Z, N_ACT)

vec = VectorizedImaginedEnv(wm_cw, n_envs=N_ENVS, z_dim=Z)
inits = np.zeros((N_ENVS, Z), dtype='f')
for i in range(N_ENVS):
    inits[i] = GridEnv().reset()
imag_obs = vec.reset(inits)
imag_steps = 0
ep_c = np.zeros(N_ENVS, dtype=int)
ckpts_b = []

t0 = time.perf_counter()
while time.perf_counter()-t0 < TIME_BUDGET:
    actions = agent_b.act_batch(imag_obs)
    nobs, rews, dones = vec.step(actions)
    resets = (dones | (ep_c >= 49))
    agent_b.update(imag_obs, actions, rews, nobs, resets.astype('f'))
    imag_obs = nobs
    imag_steps += N_ENVS
    ep_c += 1
    for j in np.where(resets)[0]:
        imag_obs[j] = GridEnv().reset()
        ep_c[j] = 0
    el = time.perf_counter()-t0
    if len(ckpts_b) < int(el/0.5)+1:
        ckpts_b.append((el, imag_steps, evaluate(agent_b, 20)))

elapsed_b = time.perf_counter()-t0
final_b = evaluate(agent_b, 100)
print(f"  {imag_steps:,} steps, {imag_steps/elapsed_b:,.0f} sps, {final_b*100:.0f}% win")

# Results
ratio = imag_steps / max(real_steps, 1)
print(f"\n{'='*65}")
print(f"  RESULTS ({TIME_BUDGET}s)         Real        Imagined")
print(f"  Steps:                {real_steps:>10,}      {imag_steps:>10,}")
print(f"  SPS:                  {real_steps/elapsed_a:>10,.0f}      {imag_steps/elapsed_b:>10,.0f}")
print(f"  Experience:           {'1.0x':>10s}      {ratio:.1f}x")
print(f"  Win rate:             {final_a*100:>9.0f}%      {final_b*100:>9.0f}%")

print(f"\n  Learning Curve:")
print(f"  Time    R-win  I-win   R-steps    I-steps")
for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    wa=sa=wb=sb=0
    for ct,cs,cw in ckpts_a:
        if ct>=t: wa,sa=cw,cs; break
    else:
        if ckpts_a: wa,sa=ckpts_a[-1][2],ckpts_a[-1][1]
    for ct,cs,cw in ckpts_b:
        if ct>=t: wb,sb=cw,cs; break
    else:
        if ckpts_b: wb,sb=ckpts_b[-1][2],ckpts_b[-1][1]
    print(f"  {t:.1f}s   {wa*100:5.0f}%  {wb*100:5.0f}%   {sa:>8,}   {sb:>8,}")

winner = "IMAGINATION" if final_b > final_a else ("REAL" if final_a > final_b else "TIE")
print(f"\n  Winner: {winner}")
print(f"  Imagination: {ratio:.1f}x experience in same wall-clock time")
print(f"{'='*65}")
