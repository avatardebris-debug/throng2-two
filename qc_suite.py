"""
Quality Control Suite — Full validation of the Imagination Engine.

Tests every module built today:
  1. SurpriseClassifier: Type 1/2 detection, per-entity tracking
  2. WorldModel upgrades: measure_surprise, predict_batch, fast_retrain, info_gain
  3. ImaginedSim: lifecycle, pattern interrupt handling
  4. VectorizedImaginedEnv: batched stepping, correctness
  5. ExplorationController: mode transitions, budget awareness
  6. CalibrationScheduler: sync calibrator, tier escalation
  7. ConsequenceCascadeBuffer: backward weighting, sampling
  8. UncertaintySeeker: entity lifecycle, precursor graph
  9. MultiResolutionEncoder: all input paths, batch encoding
  10. Integration: full pipeline end-to-end
  11. Speed benchmarks: predict_batch scaling, VecImaginedEnv SPS
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import time
import traceback

Z = 16
N_ACT = 4
PASS = 0
FAIL = 0
ERRORS = []

def test(name, fn):
    global PASS, FAIL, ERRORS
    try:
        fn()
        PASS += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()

# ═════════════════════════════════════════════════════════
#  1. SurpriseClassifier
# ═════════════════════════════════════════════════════════
print("\n=== 1. SurpriseClassifier ===")
from src.cell.surprise_classifier import SurpriseClassifier, SurpriseResult

def test_clf_type1():
    clf = SurpriseClassifier()
    s = np.zeros(Z, dtype='f')
    # Warm up rolling history with small errors
    for _ in range(20):
        clf.classify(s + np.random.randn(Z)*0.01, s + np.random.randn(Z)*0.01, s)
    # Small error → parametric
    r = clf.classify(s, s + np.random.randn(Z)*0.03, s, 'player')
    assert r.is_parametric, f"Expected parametric, got {r.type}"

def test_clf_type2():
    clf = SurpriseClassifier(interrupt_abs=0.3)
    s = np.zeros(Z, dtype='f')
    for _ in range(20):
        clf.classify(s + np.random.randn(Z)*0.01, s + np.random.randn(Z)*0.01, s)
    # Massive coherent shift → structural / interrupt
    actual = s.copy()
    actual[:Z] += 3.0
    r = clf.classify(s, actual, s, 'wall')
    assert r.is_structural or r.is_pattern_interrupt, f"Expected structural, got {r.type}"

def test_clf_entity_tracking():
    clf = SurpriseClassifier()
    s = np.zeros(Z, dtype='f')
    for tag in ['player', 'wall']:
        for _ in range(10):
            clf.classify(s, s + np.random.randn(Z)*0.05, s, tag)
    errs = clf.per_entity_avg_error()
    assert 'player' in errs and 'wall' in errs
    assert clf.worst_understood_entity() is not None

def test_clf_stats():
    clf = SurpriseClassifier()
    s = np.zeros(Z, dtype='f')
    clf.classify(s, s+0.01, s)
    st = clf.stats()
    assert 'n_parametric' in st
    assert 'n_structural' in st

test("Type 1 (parametric)", test_clf_type1)
test("Type 2 (structural)", test_clf_type2)
test("Per-entity tracking", test_clf_entity_tracking)
test("Stats output", test_clf_stats)

# ═════════════════════════════════════════════════════════
#  2. WorldModel upgrades
# ═════════════════════════════════════════════════════════
print("\n=== 2. WorldModel upgrades ===")
from src.cell.world_model import CellWorldModel

def make_wm(n=50):
    wm = CellWorldModel(feature_dim=Z, n_actions=N_ACT, hidden_size=32,
                        min_transitions=5, batch_size=8)
    for i in range(n):
        s = np.random.randn(Z).astype('f') * 0.1
        ns = s + np.random.randn(Z).astype('f') * 0.05
        wm.store_transition(s, i % N_ACT, ns, 0.1)
        wm.train_step()
    return wm

def test_measure_surprise():
    wm = make_wm()
    s = np.random.randn(Z).astype('f')
    ns = s + np.random.randn(Z).astype('f') * 0.1
    r = wm.measure_surprise(s, 0, ns, 'player')
    assert isinstance(r, SurpriseResult)
    assert hasattr(r, 'total')

def test_predict_batch_correctness():
    wm = make_wm()
    states = np.random.randn(8, Z).astype('f') * 0.1
    actions = np.random.randint(0, N_ACT, 8)
    ns_batch, rew_batch = wm.predict_batch(states, actions)
    assert ns_batch.shape == (8, Z)
    assert rew_batch.shape == (8,)
    # Compare to single predict
    ns_single, rew_single = wm.predict(states[0], int(actions[0]))
    diff = float(np.mean(np.abs(ns_batch[0] - ns_single)))
    assert diff < 1e-5, f"Batch/single diff: {diff}"

def test_sim_reset_step():
    wm = make_wm()
    s = np.random.randn(Z).astype('f')
    wm.sim_reset(s)
    ns, rew, surp = wm.step(0)
    assert ns.shape == (Z,)
    # Step with real_next
    real = np.random.randn(Z).astype('f') * 0.1
    ns2, rew2, surp2 = wm.step(1, real_next=real)
    assert wm.sim2real_accuracy >= 0

def test_information_gain():
    wm = make_wm()
    s = np.random.randn(Z).astype('f')
    ig = wm.information_gain(s, 0, n_samples=3)
    assert isinstance(ig, float) and ig >= 0
    igs = wm.information_gains_all_actions(s, n_samples=3)
    assert igs.shape == (N_ACT,)

def test_fast_retrain():
    wm = make_wm()
    s = np.random.randn(Z).astype('f')
    ns = s + np.random.randn(Z).astype('f') * 0.1
    transitions = [(s, 0, ns, 1.0)] * 3
    result = wm.fast_retrain(transitions, lr_multiplier=2.0, n_steps=5)
    assert 'fast_retrain_loss' in result
    assert result['n_steps'] == 5

def test_wm_stats():
    wm = make_wm()
    st = wm.stats()
    assert 'sim2real_accuracy' in st
    assert 'n_parametric' in st
    assert 'per_entity_confidence' in st

test("measure_surprise", test_measure_surprise)
test("predict_batch correctness", test_predict_batch_correctness)
test("sim_reset + step", test_sim_reset_step)
test("information_gain", test_information_gain)
test("fast_retrain", test_fast_retrain)
test("WM stats", test_wm_stats)

# ═════════════════════════════════════════════════════════
#  3. ImaginedSim
# ═════════════════════════════════════════════════════════
print("\n=== 3. ImaginedSim ===")
from src.cell.imagined_sim import ImaginedSim, PatternInterruptHandler

class FakeEnv:
    def reset(self): return np.zeros(Z, dtype='f')
    def step(self, a): return np.random.randn(Z).astype('f')*0.05, 0.1, False, {}

def test_imagined_sim_lifecycle():
    wm = make_wm()
    sim = ImaginedSim(wm, FakeEnv(), confidence_threshold=0.01, sim2real_trust_threshold=0.0)
    obs = sim.reset()
    assert obs.shape == (Z,)
    for i in range(10):
        obs, r, done, info = sim.step(i % N_ACT)
    assert 'imagined' in info
    assert sim.sim_state in ['active', 'inactive', 'fallback', 'reconfiguring']

def test_pattern_interrupt_handler():
    wm = make_wm()
    handler = PatternInterruptHandler(wm, fast_retrain_steps=3)
    s = np.random.randn(Z).astype('f')
    pred = s + 0.01
    actual = s + 2.0  # massive displacement
    clf = SurpriseClassifier()
    surprise = clf.classify(pred, actual, s)
    result = handler.handle(s, 0, pred, actual, 1.0, surprise)
    assert result.shape == (Z,)
    assert handler.n_handled == 1

test("ImaginedSim lifecycle", test_imagined_sim_lifecycle)
test("PatternInterruptHandler", test_pattern_interrupt_handler)

# ═════════════════════════════════════════════════════════
#  4. VectorizedImaginedEnv
# ═════════════════════════════════════════════════════════
print("\n=== 4. VectorizedImaginedEnv ===")
from src.cell.vec_imagined_env import VectorizedImaginedEnv

def test_vec_reset_step():
    wm = make_wm()
    vec = VectorizedImaginedEnv(wm, n_envs=8, z_dim=Z)
    init = np.random.randn(8, Z).astype('f') * 0.1
    obs = vec.reset(init)
    assert obs.shape == (8, Z)
    actions = np.random.randint(0, N_ACT, 8)
    obs2, rew, done = vec.step(actions)
    assert obs2.shape == (8, Z)
    assert rew.shape == (8,)
    assert done.shape == (8,)

def test_vec_broadcast_reset():
    wm = make_wm()
    vec = VectorizedImaginedEnv(wm, n_envs=4, z_dim=Z)
    obs = vec.reset(np.zeros(Z, dtype='f'))  # 1D broadcast
    assert obs.shape == (4, Z)

def test_vec_status():
    wm = make_wm()
    vec = VectorizedImaginedEnv(wm, n_envs=4, z_dim=Z)
    vec.reset(np.zeros((4, Z), dtype='f'))
    st = vec.status()
    assert st['imagined'] == True
    assert st['active'] == 4

test("VecImaginedEnv reset/step", test_vec_reset_step)
test("VecImaginedEnv broadcast reset", test_vec_broadcast_reset)
test("VecImaginedEnv status", test_vec_status)

# ═════════════════════════════════════════════════════════
#  5. ExplorationController
# ═════════════════════════════════════════════════════════
print("\n=== 5. ExplorationController ===")
from src.learning.exploration_controller import ExplorationController, PLAYGROUND, TEST, RECALIBRATE

def test_exploration_modes():
    wm = make_wm()
    ctrl = ExplorationController(wm, n_actions=N_ACT,
                                 overall_confidence_min=0.01)
    m = ctrl.mode(step_budget_remaining=100)
    assert m in [PLAYGROUND, TEST, RECALIBRATE]

def test_budget_forced_test():
    wm = make_wm()
    ctrl = ExplorationController(wm, n_actions=N_ACT,
                                 overall_confidence_min=0.01,
                                 critical_budget_fraction=0.15)
    ctrl.set_budget(100)
    ctrl.update_budget(10)  # 10% < 15% critical
    m = ctrl.mode(step_budget_remaining=10)
    assert m == TEST, f"Expected test, got {m}"

def test_recalibrate():
    wm = make_wm()
    ctrl = ExplorationController(wm, n_actions=N_ACT,
                                 overall_confidence_min=0.01,
                                 recalibrate_steps=5)
    m = ctrl.mode(structural_surprise=True, interrupt_entity='wall')
    assert m == RECALIBRATE
    for _ in range(10):
        m = ctrl.mode()
    assert m != RECALIBRATE  # countdown expired

def test_playground_action():
    wm = make_wm()
    ctrl = ExplorationController(wm, n_actions=N_ACT)
    s = np.random.randn(Z).astype('f')
    a = ctrl.select_action_playground(s)
    assert 0 <= a < N_ACT

test("Mode transitions", test_exploration_modes)
test("Budget forced TEST", test_budget_forced_test)
test("Recalibrate countdown", test_recalibrate)
test("Playground action selection", test_playground_action)

# ═════════════════════════════════════════════════════════
#  6. CalibrationScheduler (sync only for QC)
# ═════════════════════════════════════════════════════════
print("\n=== 6. CalibrationScheduler ===")
from src.learning.calibration_scheduler import SyncCalibrator

def test_sync_calibrator():
    wm = make_wm(100)
    cal = SyncCalibrator(wm, FakeEnv(), lambda o: o.astype('f'),
                         N_ACT, calibration_steps=16, tier1_threshold=0.15)
    result = cal.calibrate()
    assert hasattr(result, 'tier1_surprise')
    assert hasattr(result, 'retrained')
    assert result.n_steps == 16

def test_calibrator_stats():
    wm = make_wm(100)
    cal = SyncCalibrator(wm, FakeEnv(), lambda o: o.astype('f'),
                         N_ACT, calibration_steps=16)
    cal.calibrate()
    st = cal.stats()
    assert 'n_calibrations' in st
    assert st['n_calibrations'] == 1

test("SyncCalibrator run", test_sync_calibrator)
test("SyncCalibrator stats", test_calibrator_stats)

# ═════════════════════════════════════════════════════════
#  7. ConsequenceCascadeBuffer
# ═════════════════════════════════════════════════════════
print("\n=== 7. ConsequenceCascadeBuffer ===")
from src.learning.consequence_cascade import ConsequenceCascadeBuffer

def test_cascade_basic():
    cc = ConsequenceCascadeBuffer(capacity=1000, min_magnitude=0.5)
    cc.begin_episode()
    for i in range(50):
        cc.store_step(np.zeros(Z), i%N_ACT, np.zeros(Z), 0.0)
    cc.mark_consequential(step=49, magnitude=5.0)
    cc.end_episode()
    assert len(cc) > 0
    assert cc.stats()['n_consequences'] >= 1

def test_cascade_sampling():
    cc = ConsequenceCascadeBuffer(capacity=1000, min_magnitude=0.5)
    cc.begin_episode()
    for i in range(100):
        cc.store_step(np.random.randn(Z).astype('f'), i%N_ACT,
                     np.random.randn(Z).astype('f'), 10.0 if i==99 else 0.0)
    cc.mark_consequential(99, 10.0)
    cc.end_episode()
    batch = cc.sample(32)
    assert len(batch) == 32
    # High-weight samples should dominate
    avg_w = np.mean([t.weight for t in batch])
    assert avg_w > cc.base_weight

def test_cascade_auto_detect():
    cc = ConsequenceCascadeBuffer(capacity=1000, min_magnitude=2.0)
    cc.begin_episode()
    for i in range(20):
        cc.store_step(np.zeros(Z), 0, np.zeros(Z), 5.0 if i==10 else 0.0)
    cc.end_episode(auto_detect_consequences=True)
    assert cc.stats()['n_consequences'] >= 1

test("Cascade basic", test_cascade_basic)
test("Cascade sampling bias", test_cascade_sampling)
test("Cascade auto-detect", test_cascade_auto_detect)

# ═════════════════════════════════════════════════════════
#  8. UncertaintySeeker + PrecursorGraph
# ═════════════════════════════════════════════════════════
print("\n=== 8. UncertaintySeeker ===")
from src.learning.uncertainty_seeker import UncertaintySeeker, PrecursorGraph, EntityStatus

def test_precursor_graph():
    g = PrecursorGraph(min_edge_strength=0.05)
    for _ in range(10):
        g.observe_confidence_change('key', 'door', 0.15)
        g.observe_confidence_change('switch', 'wall', 0.10)
    g.rebuild()
    assert g.strongest_precursor('door') == 'key'
    assert g.strongest_precursor('wall') == 'switch'
    assert g.has_precursors('door')

def test_seeker_entity_lifecycle():
    wm = make_wm()
    seeker = UncertaintySeeker(wm, N_ACT, understood_threshold=0.10, max_investigation=5)
    seeker.register_entity('player')
    seeker.register_entity('door', is_consequential=True)
    target = seeker.next_target()
    assert target is not None
    # Interact
    s = np.random.randn(Z).astype('f')
    for i in range(6):
        ns = s + np.random.randn(Z).astype('f') * 0.02
        seeker.observe(s, i%N_ACT, ns, 'player')
        s = ns
    # Player should be transitioning toward investigated/understood
    rec = seeker._entities['player']
    assert rec.n_interactions == 6

def test_seeker_priorities():
    wm = make_wm()
    seeker = UncertaintySeeker(wm, N_ACT)
    seeker.register_entity('a')
    seeker.register_entity('b', is_consequential=True)
    p = seeker.get_exploration_priorities()
    assert 'a' in p and 'b' in p
    assert p['b'] >= p['a']  # consequential gets boost

test("PrecursorGraph", test_precursor_graph)
test("Entity lifecycle", test_seeker_entity_lifecycle)
test("Exploration priorities", test_seeker_priorities)

# ═════════════════════════════════════════════════════════
#  9. MultiResolutionEncoder
# ═════════════════════════════════════════════════════════
print("\n=== 9. MultiResolutionEncoder ===")
from src.encoder.multi_resolution_encoder import MultiResolutionEncoder

def test_multires_structured():
    enc = MultiResolutionEncoder()
    g = np.random.randint(0, 10, (15, 20)).astype('f')
    z = enc.encode_structured(g, (7, 10), [('enemy', 5, 12)])
    assert z.shape == (enc.z_dim,)

def test_multires_flat():
    enc = MultiResolutionEncoder()
    obs = np.random.rand(378).astype('f')
    z = enc.encode_flat_obs(obs)
    assert z.shape == (enc.z_dim,)

def test_multires_batch():
    enc = MultiResolutionEncoder()
    obs = np.random.rand(12, 378).astype('f')
    z = enc.encode_flat_obs_batch(obs)
    assert z.shape == (12, enc.z_dim)

def test_multires_edges():
    enc = MultiResolutionEncoder()
    g = np.random.randint(0, 10, (15, 20)).astype('f')
    z1 = enc.encode_structured(g, (0, 0), None)
    z2 = enc.encode_structured(g, (14, 19), None)
    assert z1.shape == z2.shape == (enc.z_dim,)
    # Different positions should give different focal encodings
    assert not np.allclose(z1, z2)

def test_multires_resolution():
    enc = MultiResolutionEncoder()
    r = enc.describe_resolution()
    assert r['total_z_dim'] == enc.z_dim
    assert '6.1' in r['focal']['effective_resolution']

test("Structured encoding", test_multires_structured)
test("Flat obs encoding", test_multires_flat)
test("Batch encoding", test_multires_batch)
test("Edge positions", test_multires_edges)
test("Resolution description", test_multires_resolution)

# ═════════════════════════════════════════════════════════
#  10. Integration: full pipeline
# ═════════════════════════════════════════════════════════
print("\n=== 10. Integration ===")

def test_full_pipeline():
    """End-to-end: encode → WM → imagine → calibrate → explore."""
    # Setup
    enc = MultiResolutionEncoder(z_global_dim=8, z_focal_dim=4, z_entity_dim=4)
    z_dim = enc.z_dim  # 16
    wm = CellWorldModel(feature_dim=z_dim, n_actions=N_ACT,
                        hidden_size=32, min_transitions=5, batch_size=8)

    # Seed WM
    for i in range(30):
        obs = np.random.rand(378).astype('f')
        z = enc.encode_flat_obs(obs)
        obs2 = np.random.rand(378).astype('f')
        z2 = enc.encode_flat_obs(obs2)
        wm.store_transition(z, i%N_ACT, z2, 0.1)
        wm.train_step()

    # Imagined envs
    vec = VectorizedImaginedEnv(wm, n_envs=8, z_dim=z_dim)
    init = np.random.randn(8, z_dim).astype('f') * 0.1
    obs = vec.reset(init)

    # Exploration controller
    ctrl = ExplorationController(wm, N_ACT, overall_confidence_min=0.01)

    # Cascade buffer
    cascade = ConsequenceCascadeBuffer(capacity=5000)

    # 100 imagined steps
    cascade.begin_episode()
    for step in range(100):
        mode = ctrl.mode()
        if mode == PLAYGROUND:
            actions = np.array([ctrl.select_action_playground(obs[0])] * 8)
        else:
            actions = np.random.randint(0, N_ACT, 8)
        obs, rew, done = vec.step(actions)
        cascade.store_step(obs[0], int(actions[0]), obs[0], float(rew[0]))
    cascade.end_episode()

    assert vec.stats()['n_steps'] == 800  # 8 envs × 100 steps
    assert len(cascade) > 0

test("Full pipeline", test_full_pipeline)

# ═════════════════════════════════════════════════════════
#  11. Speed benchmarks
# ═════════════════════════════════════════════════════════
print("\n=== 11. Speed benchmarks ===")

def bench_predict_batch():
    wm = make_wm()
    results = {}
    for N in [1, 4, 12, 32, 64]:
        states = np.random.randn(N, Z).astype('f') * 0.1
        actions = np.random.randint(0, N_ACT, N)
        # Warmup
        for _ in range(3):
            wm.predict_batch(states, actions)
        # Timed
        t0 = time.perf_counter()
        n_iters = 200
        for _ in range(n_iters):
            wm.predict_batch(states, actions)
        elapsed = time.perf_counter() - t0
        sps = n_iters * N / elapsed
        results[N] = sps
    return results

def bench_vec_imagined():
    results = {}
    for N in [4, 12, 32, 64]:
        wm = make_wm()
        vec = VectorizedImaginedEnv(wm, n_envs=N, z_dim=Z)
        init = np.random.randn(N, Z).astype('f') * 0.1
        vec.reset(init)
        # Warmup
        for _ in range(5):
            vec.step(np.random.randint(0, N_ACT, N))
        # Timed
        n_steps = 500
        t0 = time.perf_counter()
        for _ in range(n_steps):
            vec.step(np.random.randint(0, N_ACT, N))
        elapsed = time.perf_counter() - t0
        sps = n_steps * N / elapsed
        results[N] = sps
    return results

def bench_multires():
    enc = MultiResolutionEncoder()
    obs = np.random.rand(12, 378).astype('f')
    # Warmup
    enc.encode_flat_obs_batch(obs)
    # Timed
    t0 = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        enc.encode_flat_obs_batch(obs)
    elapsed = time.perf_counter() - t0
    return n_iters * 12 / elapsed

print("\n  predict_batch SPS:")
pb = bench_predict_batch()
for n, sps in pb.items():
    print(f"    N={n:3d}  →  {sps:>10,.0f} sps")

print("\n  VectorizedImaginedEnv SPS:")
vi = bench_vec_imagined()
for n, sps in vi.items():
    print(f"    N={n:3d}  →  {sps:>10,.0f} sps")

print(f"\n  MultiResolutionEncoder:")
mr_sps = bench_multires()
print(f"    batch(12)  →  {mr_sps:>10,.0f} encodes/sec")

# ═════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"\n  FAILURES:")
    for name, err in ERRORS:
        print(f"    {name}: {err}")
print(f"{'='*60}")
