"""Test CalibrationScheduler using SyncCalibrator."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import numpy as np
    import time

    from src.cell.world_model import CellWorldModel
    from src.cell.vec_imagined_env import VectorizedImaginedEnv
    from src.learning.calibration_scheduler import SyncCalibrator, CalibrationScheduler

    Z_DIM = 16
    N_ACTIONS = 4
    N_ENVS = 32

    # ── Fake real env (simple dynamics: next = state + action_encoding * 0.1)
    class FakeRealEnv:
        def __init__(self):
            self.state = np.zeros(Z_DIM, dtype=np.float32)

        def reset(self):
            self.state = np.random.randn(Z_DIM).astype(np.float32) * 0.1
            return self.state.copy()

        def step(self, action):
            delta = np.zeros(Z_DIM, dtype=np.float32)
            delta[action % Z_DIM] = 0.1
            self.state = self.state + delta + np.random.randn(Z_DIM).astype(np.float32) * 0.01
            return self.state.copy(), 0.1, False, {}

    # Identity encoder (fake env already returns z)
    encode_fn = lambda obs: np.asarray(obs, dtype=np.float32)

    print("=== Test 1: SyncCalibrator ===")
    wm = CellWorldModel(feature_dim=Z_DIM, n_actions=N_ACTIONS,
                        hidden_size=32, min_transitions=5, batch_size=8)

    # Seed WM with some transitions first
    env = FakeRealEnv()
    obs = env.reset()
    for i in range(50):
        a = i % N_ACTIONS
        new_obs, r, _, _ = env.step(a)
        wm.store_transition(encode_fn(obs), a, encode_fn(new_obs), r)
        wm.train_step()
        obs = new_obs

    print(f"  WM ready: {wm.is_ready}, confidence: {wm.confidence:.3f}")

    cal = SyncCalibrator(
        world_model=wm,
        real_env=FakeRealEnv(),
        encode_fn=encode_fn,
        n_actions=N_ACTIONS,
        calibration_steps=32,
        tier1_threshold=0.15,
        fast_retrain_steps=10,
    )

    # Run 3 calibration episodes
    for i in range(3):
        result = cal.calibrate()
        print(f"  Cal {i+1}: tier1_surprise={result.tier1_surprise:.4f}, "
              f"retrained={result.retrained}, tier={result.tier_reached}, "
              f"conf: {result.wm_confidence_before:.3f} -> {result.wm_confidence_after:.3f}")

    print(f"  Calibrator stats: {cal.stats()}")
    print()

    # ── Test 2: Full imagination + calibration flow
    print("=== Test 2: VectorizedImaginedEnv + SyncCalibrator ===")

    wm2 = CellWorldModel(feature_dim=Z_DIM, n_actions=N_ACTIONS,
                         hidden_size=32, min_transitions=5, batch_size=8)

    # Seed WM
    env2 = FakeRealEnv()
    obs2 = env2.reset()
    for i in range(100):
        a = i % N_ACTIONS
        new_obs, r, _, _ = env2.step(a)
        wm2.store_transition(encode_fn(obs2), a, encode_fn(new_obs), r)
        wm2.train_step()
        obs2 = new_obs

    # Set up imagined envs
    vec = VectorizedImaginedEnv(wm2, n_envs=N_ENVS, z_dim=Z_DIM)
    init = np.random.randn(N_ENVS, Z_DIM).astype(np.float32) * 0.1
    vec.reset(init)

    # Set up sync calibrator
    cal2 = SyncCalibrator(wm2, FakeRealEnv(), encode_fn, N_ACTIONS,
                          calibration_steps=32, tier1_threshold=0.15)

    # Training loop: imagination + periodic calibration
    TOTAL_STEPS = 1000
    CALIBRATE_EVERY = 200

    t0 = time.perf_counter()
    for step in range(TOTAL_STEPS):
        actions = np.random.randint(0, N_ACTIONS, N_ENVS)
        vec.step(actions)

        # Periodic calibration (non-blocking in async version)
        if (step + 1) % CALIBRATE_EVERY == 0:
            result = cal2.calibrate()
            print(f"  Step {(step+1)*N_ENVS:>6,}: cal tier1={result.tier1_surprise:.4f} "
                  f"retrained={result.retrained} wm_conf={result.wm_confidence_after:.3f}")

    elapsed = time.perf_counter() - t0
    total_imagined = TOTAL_STEPS * N_ENVS
    sps = total_imagined / elapsed

    print(f"\n  Imagined throughput: {sps:,.0f} sps ({N_ENVS} envs × {TOTAL_STEPS} steps)")
    print(f"  Calibration cost: {cal2.stats()['n_calibrations']} episodes "
          f"× 32 real steps = {cal2.stats()['n_calibrations'] * 32} total real steps")
    print(f"  Real/imagined ratio: {cal2.stats()['n_calibrations'] * 32 / total_imagined:.4f} "
          f"({cal2.stats()['n_calibrations'] * 32 / total_imagined * 100:.2f}%)")
    print(f"  Cal stats: {cal2.stats()}")
    print()

    # ── Test 3: Async CalibrationScheduler (brief smoke test)
    print("=== Test 3: Async CalibrationScheduler (3s smoke test) ===")

    wm3 = CellWorldModel(feature_dim=Z_DIM, n_actions=N_ACTIONS,
                         hidden_size=32, min_transitions=5, batch_size=8)
    # Seed
    env3 = FakeRealEnv()
    obs3 = env3.reset()
    for i in range(50):
        a = i % N_ACTIONS
        no, r, _, _ = env3.step(a)
        wm3.store_transition(encode_fn(obs3), a, encode_fn(no), r)
        wm3.train_step()
        obs3 = no

    scheduler = CalibrationScheduler(
        world_model=wm3,
        real_env_fn=FakeRealEnv,
        encode_fn=encode_fn,
        n_actions=N_ACTIONS,
        calibration_steps=32,
        calibration_interval_s=0.5,
        tier1_threshold=0.15,
    )

    scheduler.start()
    time.sleep(3.0)  # Let background calibrations run
    scheduler.stop()

    results = scheduler.drain_updates()
    print(f"  Background calibrations completed: {len(results)}")
    for i, r in enumerate(results):
        print(f"    Cal {i+1}: tier1={r.tier1_surprise:.4f} retrained={r.retrained}")
    print(f"  Scheduler stats: {scheduler.stats()}")
    print()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
