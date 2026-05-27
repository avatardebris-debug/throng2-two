"""
throng_vec_env.py — PufferLib-inspired async vectorized environment runner.

Distilled from PufferLib 3.0 (puffer.ai/blog) for Python 3.13.

Key features:
  1. N worker processes, shared-memory zero-copy buffers
  2. Async send/recv — GPU policy and CPU envs overlap
  3. Auto-scaling probe via psutil
  4. Dynamic scaling: pause/resume workers (free CPU) or prune (free RAM too)
  5. ResourceMonitor background thread — auto-adjusts under pressure

Usage
-----
    from src.learning.throng_vec_env import ThrongVecEnv, probe_resources

    n = probe_resources(obs_dim=378)
    vec = ThrongVecEnv(lambda: MarioSim(), n_envs=n, obs_dim=378,
                       auto_scale=True)   # ← turn on the monitor
    obs = vec.reset()
    while training:
        vec.send(policy(obs))
        obs, rew, done = vec.recv()     # active_n may shrink/grow here
"""

from __future__ import annotations

import logging
import os
import time
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import psutil

_log = logging.getLogger(__name__)

try:
    import cloudpickle as _pickle_mod
except ImportError:
    import pickle as _pickle_mod  # type: ignore


# ══════════════════════════════════════════════════════════════════════
#  Resource probe
# ══════════════════════════════════════════════════════════════════════

def probe_resources(
    obs_dim: int,
    n_actions: int = 8,
    target_mem_util: float = 0.70,
    target_cpu_util: float = 0.80,
    min_envs: int = 2,
    max_envs: int = 2048,
    verbose: bool = True,
) -> int:
    """Return safe n_envs for this machine given obs_dim."""
    mem = psutil.virtual_memory()
    free_ram_gb = mem.available / 1e9
    total_ram_gb = mem.total / 1e9
    n_cores = psutil.cpu_count(logical=True) or 4

    bytes_per_env = (obs_dim * 4 + n_actions * 4 + 64) + 5 * 1024 * 1024
    n_from_ram = int(free_ram_gb * 1e9 * target_mem_util / bytes_per_env)
    n_from_cpu = int(n_cores * 3 * target_cpu_util)

    n_from_vram = max_envs
    try:
        import torch
        if torch.cuda.is_available():
            free_vram, _ = torch.cuda.mem_get_info()
            n_from_vram = int(free_vram * target_mem_util / (1 * 1024 * 1024))
    except ImportError:
        pass

    n = max(min_envs, min(n_from_ram, n_from_cpu, n_from_vram, max_envs))

    if verbose:
        print(f"[ThrongVecEnv] Resource probe:")
        print(f"  RAM: {free_ram_gb:.1f}/{total_ram_gb:.1f} GB free → {n_from_ram} envs")
        print(f"  CPU: {n_cores} cores → {n_from_cpu} envs")
        if n_from_vram < max_envs:
            print(f"  VRAM cap: {n_from_vram} envs")
        print(f"  → Recommended: {n} envs")
    return n


# ══════════════════════════════════════════════════════════════════════
#  Worker process
# ══════════════════════════════════════════════════════════════════════

_CMD_STEP  = 0
_CMD_RESET = 1
_CMD_CLOSE = 2
_CMD_PAUSE = 3   # worker blocks until CMD_RESUME, uses ~0 CPU
_CMD_RESUME = 4


def _worker_fn(
    worker_id: int,
    env_fn_bytes: bytes,
    obs_shm_name: str,
    rew_shm_name: str,
    don_shm_name: str,
    obs_dim: int,
    cmd_q: mp.Queue,
    ack_q: mp.Queue,
):
    """Worker process: one env, reads/writes shared memory."""
    try:
        import cloudpickle as _pm
    except ImportError:
        import pickle as _pm  # type: ignore

    env_fn = _pm.loads(env_fn_bytes)

    obs_shm = shared_memory.SharedMemory(name=obs_shm_name)
    rew_shm = shared_memory.SharedMemory(name=rew_shm_name)
    don_shm = shared_memory.SharedMemory(name=don_shm_name)

    obs_buf = np.ndarray((obs_dim,), dtype=np.float32, buffer=obs_shm.buf)
    rew_buf = np.ndarray((1,),       dtype=np.float32, buffer=rew_shm.buf)
    don_buf = np.ndarray((1,),       dtype=np.uint8,   buffer=don_shm.buf)

    env = env_fn()
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs_buf[:] = np.asarray(obs, dtype=np.float32)
    rew_buf[0] = 0.0
    don_buf[0] = 0
    ack_q.put(True)

    while True:
        try:
            cmd, payload = cmd_q.get()
        except Exception:
            break

        if cmd == _CMD_CLOSE:
            break

        elif cmd == _CMD_PAUSE:
            # Block cheaply until resumed — uses almost no CPU
            ack_q.put(True)
            while True:
                resume_cmd, _ = cmd_q.get()
                if resume_cmd == _CMD_RESUME:
                    ack_q.put(True)
                    break
                elif resume_cmd == _CMD_CLOSE:
                    env.close()
                    obs_shm.close(); rew_shm.close(); don_shm.close()
                    return

        elif cmd == _CMD_RESET:
            try:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs_buf[:] = np.asarray(obs, dtype=np.float32)
                rew_buf[0] = 0.0
                don_buf[0] = 0
                ack_q.put(True)
            except Exception as e:
                ack_q.put(e)

        elif cmd == _CMD_STEP:
            action = payload
            try:
                result = env.step(action)
                if len(result) == 4:
                    obs, reward, done, _ = result
                else:
                    obs, reward, done, trunc, _ = result
                    done = done or trunc
                obs_buf[:] = np.asarray(obs, dtype=np.float32)
                rew_buf[0] = float(reward)
                don_buf[0] = int(bool(done))
                ack_q.put(True)
            except Exception as e:
                ack_q.put(e)

    try:
        env.close()
    except Exception as _e:
        import sys
        print(f"[throng_vec_env worker {worker_id}] env.close() error: {_e}", file=sys.stderr)
    obs_shm.close(); rew_shm.close(); don_shm.close()


# ══════════════════════════════════════════════════════════════════════
#  ResourceMonitor — background thread
# ══════════════════════════════════════════════════════════════════════

class ResourceMonitor(threading.Thread):
    """
    Watches RAM and CPU every `interval` seconds.
    Pauses workers when pressure is high; resumes when it drops.

    Thresholds:
      ram_high / cpu_high / vram_high  → pause one worker (repeat each interval)
      ram_low  / cpu_low  / vram_low   → resume one paused worker (ramp up gently)

    Never goes below min_active workers.
    Works identically on laptop, cloud CPU instance, and GPU cloud instance.
    """

    def __init__(
        self,
        vec: "ThrongVecEnv",
        interval: float = 2.0,
        ram_high:  float = 0.88,
        ram_low:   float = 0.78,
        cpu_high:  float = 0.90,
        cpu_low:   float = 0.75,
        vram_high: float = 0.90,   # GPU memory high watermark (cloud GPU)
        vram_low:  float = 0.75,   # GPU memory low watermark
        min_active: int = 2,
        verbose: bool = True,
    ):
        super().__init__(daemon=True)
        self.vec        = vec
        self.interval   = interval
        self.ram_high   = ram_high
        self.ram_low    = ram_low
        self.cpu_high   = cpu_high
        self.cpu_low    = cpu_low
        self.vram_high  = vram_high
        self.vram_low   = vram_low
        self.min_active = min_active
        self.verbose    = verbose
        self._stop_evt  = threading.Event()

        # Detect GPU once at init
        self._has_gpu = False
        try:
            import torch
            self._has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

    def _vram_util(self) -> float:
        """Returns VRAM utilization 0-1, or 0.0 if no GPU."""
        if not self._has_gpu:
            return 0.0
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            return (total - free) / total if total > 0 else 0.0
        except Exception:
            return 0.0

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.wait(timeout=self.interval):
            # CRITICAL: Don't touch workers while a step cycle is in progress.
            # Pausing mid-step steals the step-ack from recv(), causing a timeout.
            if self.vec._stepping:
                continue

            ram  = psutil.virtual_memory().percent / 100.0
            cpu  = psutil.cpu_percent(interval=None) / 100.0
            vram = self._vram_util()

            pressure = (
                ram  > self.ram_high  or
                cpu  > self.cpu_high  or
                vram > self.vram_high
            )
            relief = (
                ram  < self.ram_low  and
                cpu  < self.cpu_low  and
                vram < self.vram_low
            )

            if pressure and self.vec.active_count > self.min_active:
                # Double-check stepping flag after acquiring intent
                if self.vec._stepping:
                    continue
                self.vec.pause_last()
                if self.verbose:
                    reasons = []
                    if ram  > self.ram_high:  reasons.append(f"RAM={ram*100:.0f}%")
                    if cpu  > self.cpu_high:  reasons.append(f"CPU={cpu*100:.0f}%")
                    if vram > self.vram_high: reasons.append(f"VRAM={vram*100:.0f}%")
                    print(f"  [Monitor] ↓ Paused — active={self.vec.active_count}"
                          f" ({', '.join(reasons)})")

            elif relief and self.vec.paused_count > 0:
                if self.vec._stepping:
                    continue
                self.vec.resume_one()
                if self.verbose:
                    print(f"  [Monitor] ↑ Resumed — active={self.vec.active_count}"
                          f"  RAM={ram*100:.0f}%  CPU={cpu*100:.0f}%"
                          + (f"  VRAM={vram*100:.0f}%" if self._has_gpu else ""))


# ══════════════════════════════════════════════════════════════════════
#  ThrongVecEnv
# ══════════════════════════════════════════════════════════════════════

class ThrongVecEnv:
    """
    Async vectorized env runner with dynamic pause/resume/prune scaling.

    Dynamic scaling API:
        vec.pause_last()     — pause highest-index active worker (CPU→0)
        vec.resume_one()     — resume one paused worker
        vec.pause_worker(i)  — pause specific worker
        vec.resume_worker(i) — resume specific worker
        vec.prune_worker(i)  — terminate worker + free its shared memory

    With auto_scale=True, a ResourceMonitor thread handles this automatically.

    Active count affects batch size transparently:
        vec.active_count     — number of currently active workers
        obs shape from recv() is (active_count, obs_dim)
    """

    def __init__(
        self,
        env_fn: Callable,
        n_envs: int,
        obs_dim: int,
        n_actions: int = 8,
        timeout: float = 60.0,
        auto_scale: bool = False,
        monitor_kwargs: Optional[dict] = None,
    ):
        self.n_envs    = n_envs
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.timeout   = timeout
        self._pending  = False
        self._stepping = False  # True during send/recv — blocks monitor

        # Track which workers are active vs paused vs pruned
        self._active:  Set[int] = set(range(n_envs))  # stepping
        self._paused:  Set[int] = set()                # blocked cheaply
        self._pruned:  Set[int] = set()                # terminated + freed
        self._lock = threading.Lock()

        # Shared memory
        obs_bytes = obs_dim * np.dtype(np.float32).itemsize
        rew_bytes = np.dtype(np.float32).itemsize
        don_bytes = np.dtype(np.uint8).itemsize

        self._obs_shms: List[Optional[shared_memory.SharedMemory]] = []
        self._rew_shms: List[Optional[shared_memory.SharedMemory]] = []
        self._don_shms: List[Optional[shared_memory.SharedMemory]] = []

        for _ in range(n_envs):
            self._obs_shms.append(shared_memory.SharedMemory(create=True, size=obs_bytes))
            self._rew_shms.append(shared_memory.SharedMemory(create=True, size=rew_bytes))
            self._don_shms.append(shared_memory.SharedMemory(create=True, size=don_bytes))

        self._obs_np = [
            np.ndarray((obs_dim,), dtype=np.float32, buffer=shm.buf)
            for shm in self._obs_shms
        ]
        self._rew_np = [
            np.ndarray((1,), dtype=np.float32, buffer=shm.buf)
            for shm in self._rew_shms
        ]
        self._don_np = [
            np.ndarray((1,), dtype=np.uint8, buffer=shm.buf)
            for shm in self._don_shms
        ]

        env_fn_bytes = _pickle_mod.dumps(env_fn)

        ctx = mp.get_context("spawn")
        self._cmd_qs: List[Optional[mp.Queue]] = [ctx.Queue(maxsize=4) for _ in range(n_envs)]
        self._ack_qs: List[Optional[mp.Queue]] = [ctx.Queue(maxsize=4) for _ in range(n_envs)]
        self._procs:  List[Optional[mp.Process]] = []

        for i in range(n_envs):
            p = ctx.Process(
                target=_worker_fn,
                args=(
                    i, env_fn_bytes,
                    self._obs_shms[i].name,
                    self._rew_shms[i].name,
                    self._don_shms[i].name,
                    obs_dim,
                    self._cmd_qs[i],
                    self._ack_qs[i],
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

        for i, ack_q in enumerate(self._ack_qs):
            result = ack_q.get(timeout=timeout)
            if isinstance(result, Exception):
                raise RuntimeError(f"Worker {i} init failed: {result}") from result

        # Start monitor if requested
        self._monitor: Optional[ResourceMonitor] = None
        if auto_scale:
            kw = monitor_kwargs or {}
            self._monitor = ResourceMonitor(self, **kw)
            self._monitor.start()

    # ── properties ────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def paused_count(self) -> int:
        return len(self._paused)

    def _active_sorted(self) -> List[int]:
        return sorted(self._active)

    # ── dynamic scaling ───────────────────────────────────────────────

    def pause_worker(self, i: int) -> None:
        """Pause worker i — it blocks with ~0 CPU. RAM stays allocated."""
        with self._lock:
            if i not in self._active:
                return
            self._cmd_qs[i].put((_CMD_PAUSE, None))
            self._ack_qs[i].get(timeout=self.timeout)  # wait for it to block
            self._active.discard(i)
            self._paused.add(i)

    def resume_worker(self, i: int) -> None:
        """Resume a paused worker."""
        with self._lock:
            if i not in self._paused:
                return
            self._cmd_qs[i].put((_CMD_RESUME, None))
            self._ack_qs[i].get(timeout=self.timeout)
            self._paused.discard(i)
            self._active.add(i)

    def pause_last(self) -> None:
        """Pause the highest-index active worker."""
        active = self._active_sorted()
        if active:
            self.pause_worker(active[-1])

    def resume_one(self) -> None:
        """Resume the lowest-index paused worker."""
        paused = sorted(self._paused)
        if paused:
            self.resume_worker(paused[0])

    def prune_worker(self, i: int) -> None:
        """
        Terminate worker i and free its shared memory.
        More aggressive than pause — frees RAM too.
        Cannot be undone without creating a new ThrongVecEnv.
        """
        with self._lock:
            # Pause first if active
            if i in self._active:
                self._cmd_qs[i].put((_CMD_CLOSE, None))
                self._active.discard(i)
            elif i in self._paused:
                self._cmd_qs[i].put((_CMD_CLOSE, None))
                self._paused.discard(i)
            else:
                return  # already pruned

            # Terminate process
            p = self._procs[i]
            if p is not None:
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()
                self._procs[i] = None

            # Free shared memory
            for shm_list in (self._obs_shms, self._rew_shms, self._don_shms):
                shm = shm_list[i]
                if shm is not None:
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:
                        _log.debug("Could not free shared memory for worker %d", i, exc_info=True)
                    shm_list[i] = None

            self._pruned.add(i)

    def status(self) -> dict:
        """Return current scaling status."""
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=None)
        return {
            "active":  self.active_count,
            "paused":  self.paused_count,
            "pruned":  len(self._pruned),
            "ram_pct": ram,
            "cpu_pct": cpu,
        }

    # ── internal send/recv (active workers only) ──────────────────────

    def _send_active(self, cmd: int, payloads: list) -> None:
        active = self._active_sorted()
        for idx, i in enumerate(active):
            self._cmd_qs[i].put((cmd, payloads[idx]))

    def _recv_active(self) -> None:
        for i in self._active_sorted():
            result = self._ack_qs[i].get(timeout=self.timeout)
            if isinstance(result, Exception):
                raise RuntimeError(f"Worker {i} error: {result}") from result

    def _read_active_buffers(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = self._active_sorted()
        obs  = np.stack([self._obs_np[i].copy() for i in active])
        rew  = np.array([self._rew_np[i][0] for i in active], dtype=np.float32)
        done = np.array([bool(self._don_np[i][0]) for i in active])
        return obs, rew, done

    # ── public API ────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset all active workers. Returns obs (active_n, obs_dim)."""
        self._stepping = True
        try:
            n = self.active_count
            self._send_active(_CMD_RESET, [None] * n)
            self._recv_active()
            obs, _, _ = self._read_active_buffers()
            self._pending = False
            return obs
        finally:
            self._stepping = False

    def send(self, actions: np.ndarray) -> None:
        """
        Async: dispatch actions to active workers, return immediately.
        actions must have shape (active_count,).
        """
        assert not self._pending, "Call recv() before the next send()"
        self._stepping = True  # Block monitor from pausing during cycle
        self._send_active(_CMD_STEP, list(actions))
        self._pending = True

    def recv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Block until active workers finish. Auto-resets done envs.
        Returns (obs, rewards, dones) — shape (active_count, ...).
        """
        assert self._pending, "Call send() before recv()"
        try:
            self._recv_active()
            obs, rew, done = self._read_active_buffers()
            self._pending = False

            active = self._active_sorted()
            for arr_i, worker_i in enumerate(active):
                if done[arr_i]:
                    self._cmd_qs[worker_i].put((_CMD_RESET, None))
            for arr_i, worker_i in enumerate(active):
                if done[arr_i]:
                    result = self._ack_qs[worker_i].get(timeout=self.timeout)
                    if isinstance(result, Exception):
                        raise RuntimeError(f"Worker {worker_i} reset error: {result}") from result
                    obs[arr_i] = self._obs_np[worker_i].copy()

            return obs, rew, done
        finally:
            self._stepping = False  # Allow monitor to act again

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synchronous step (send + recv)."""
        self.send(actions)
        return self.recv()

    def close(self) -> None:
        """Shut down monitor, terminate all workers, free shared memory."""
        if self._monitor is not None:
            self._monitor.stop()

        for i in list(self._active | self._paused):
            try:
                self._cmd_qs[i].put((_CMD_CLOSE, None))
            except Exception:
                pass
        for p in self._procs:
            if p is not None:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
        for shm in self._obs_shms + self._rew_shms + self._don_shms:
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    _log.debug("Could not release shared memory segment during close", exc_info=True)

    def benchmark(self, n_steps: int = 500) -> float:
        """Returns steps/sec over n_steps synchronous steps."""
        self.reset()
        t0 = time.time()
        for _ in range(n_steps):
            actions = np.random.randint(0, self.n_actions, size=self.active_count)
            self.step(actions)
        return (n_steps * self.active_count) / (time.time() - t0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ══════════════════════════════════════════════════════════════════════
#  Async training loop
# ══════════════════════════════════════════════════════════════════════

def run_async_loop(
    vec: ThrongVecEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    n_steps: int = 100_000,
    on_batch: Optional[Callable] = None,
    verbose: bool = True,
):
    """
    PufferLib-style async loop: GPU policy runs while CPU envs step.

    policy_fn receives obs of shape (active_count, obs_dim) — size may vary
    if auto_scale is on. Return actions of shape (active_count,).
    """
    obs = vec.reset()
    actions = policy_fn(obs)
    vec.send(actions)

    step = 0
    t0 = time.time()
    log_every = max(1000, n_steps // 20)

    while step < n_steps:
        # Next-batch policy runs WHILE previous envs are stepping
        obs, rewards, dones = vec.recv()
        step += vec.active_count

        if on_batch is not None:
            on_batch(obs, actions, rewards, dones, step)

        actions = policy_fn(obs)  # GPU runs here — envs busy next send()

        if verbose and step % log_every < vec.active_count:
            sps = step / (time.time() - t0)
            st  = vec.status()
            print(f"  [VecEnv] step={step:,}  sps={sps:,.0f}  "
                  f"active={st['active']} paused={st['paused']}  "
                  f"RAM={st['ram_pct']:.0f}%  CPU={st['cpu_pct']:.0f}%")

        vec.send(actions)

    vec.recv()  # drain
