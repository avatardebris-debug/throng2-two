"""
mujoco_adapter.py — Triple orthographic ASCII adapter for MuJoCo environments.

Converts MuJoCo's high-dimensional continuous state into a compact observation
using three orthographic 2D projections + proprioception:

    XY view (top-down)   — footprint, lateral position, ground contact
    XZ view (front)      — height, vertical reach, jumping/standing
    YZ view (side)       — depth, approach angle, forward velocity

Each view is encoded to a 15×15 ASCII density grid (225 values).
Proprioceptive features (joint positions, velocities, contacts) are appended.

Observation layout:
    [0:225]      XY grid  (top-down)
    [225:450]    XZ grid  (front view)
    [450:675]    YZ grid  (side view)
    [675:675+P]  proprioception  (P varies: Reacher≈10, HalfCheetah≈23)

Total obs_dim = 675 + P

Design goals:
  1. No GPU required — works on CPU-only machines
  2. Graceful degradation — when mujoco is not installed, returns RAM-based obs
  3. Compatible with UniversalEncoder / MultiGameWorldModel pipeline
  4. Ablation-ready: `views=["xy"]` / `["xy","xz"]` / `["xy","xz","yz"]`

Supported environments (ordered by difficulty):
    Reacher-v4       — 2 joints, reach target (XY effectively sufficient)
    Pusher-v4        — 7 joints, push object to target (needs all 3 views)
    HalfCheetah-v4   — 6 joints, locomotion (XZ most informative)

Usage:
    adapter = MuJoCoAdapter("Reacher-v4")
    obs = adapter.reset()          # (675 + P,) float32
    obs2, rew, done, info = adapter.step(action_idx)

    # Ablation
    adapter_xy = MuJoCoAdapter("Reacher-v4", views=["xy"])
    obs_dim_xy = adapter_xy.obs_dim    # 225 + P
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Graceful mujoco import ─────────────────────────────────────
_MUJOCO_AVAILABLE = True
try:
    import gymnasium as gym
    # probe for mujoco
    import mujoco  # noqa: F401
except ImportError:
    _MUJOCO_AVAILABLE = False

# ── ASCII encoder ──────────────────────────────────────────────
try:
    from src.encoder.ascii_encoder import AsciiEncoder
except ImportError:
    from encoder.ascii_encoder import AsciiEncoder  # type: ignore


# ═══════════════════════════════════════════════════════════════
# TASK SPEC — per-environment metadata
# ═══════════════════════════════════════════════════════════════

TASK_SPECS: Dict[str, Dict] = {
    "Reacher-v4": {
        "n_joints": 2,
        "action_dim": 2,
        "pro_obs_dim": 10,   # gymnasium RAM obs dim
        "camera_distance": 0.4,
        "camera_elevation": 20,
        "description": "2-joint arm: reach a target point",
        "difficulty": 1,
    },
    "Pusher-v4": {
        "n_joints": 7,
        "action_dim": 7,
        "pro_obs_dim": 23,
        "camera_distance": 1.5,
        "camera_elevation": 20,
        "description": "7-joint arm: push object to target",
        "difficulty": 2,
    },
    "HalfCheetah-v4": {
        "n_joints": 6,
        "action_dim": 6,
        "pro_obs_dim": 17,
        "camera_distance": 3.0,
        "camera_elevation": 5,
        "description": "6-joint cheetah-like locomotion",
        "difficulty": 3,
    },
    "Ant-v4": {
        "n_joints": 8,
        "action_dim": 8,
        "pro_obs_dim": 27,
        "camera_distance": 4.0,
        "camera_elevation": 30,
        "description": "Ant quadruped locomotion",
        "difficulty": 4,
    },
    "Hopper-v4": {
        "n_joints": 3,
        "action_dim": 3,
        "pro_obs_dim": 11,
        "camera_distance": 2.5,
        "camera_elevation": 5,
        "description": "1-legged hopper locomotion",
        "difficulty": 2,
    },
}

# Default grid size for all three views
GRID_SIZE = 15  # 15×15 = 225 per view


# ═══════════════════════════════════════════════════════════════
# TRIPLE-VIEW RENDERER
# ═══════════════════════════════════════════════════════════════

class TripleViewRenderer:
    """
    Renders three orthographic views of a MuJoCo environment and
    encodes each to a 15×15 ASCII density grid.

    Views:
        XY (top-down):   camera above, looking down
        XZ (front):      camera in front, looking back along Y
        YZ (side):       camera to the right, looking left along X

    Each rendered image is (H×W×3) uint8 → AsciiEncoder → (15×15,) float32
    """

    def __init__(
        self,
        env_name: str,
        grid_size: int = GRID_SIZE,
        render_size: int = 64,
        views: Optional[List[str]] = None,
    ):
        self.env_name = env_name
        self.grid_size = grid_size
        self.render_size = render_size
        self.views = views or ["xy", "xz", "yz"]
        self._spec = TASK_SPECS.get(env_name, {})

        # One AsciiEncoder shared across all views (stateless)
        self._enc = AsciiEncoder(rows=grid_size, cols=grid_size, color_channels=False)

        # Camera positions for orthographic-like views
        # gymnasium uses: distance, azimuth, elevation, lookat
        self._cam_params = {
            "xy": dict(distance=self._spec.get("camera_distance", 1.5),
                       azimuth=0, elevation=90),      # looking straight down
            "xz": dict(distance=self._spec.get("camera_distance", 1.5),
                       azimuth=0, elevation=0),       # looking from front
            "yz": dict(distance=self._spec.get("camera_distance", 1.5),
                       azimuth=90, elevation=0),      # looking from right side
        }

    def render_views(self, env) -> Dict[str, np.ndarray]:
        """
        Render the active views and encode to ASCII grids.

        Args:
            env: Active gymnasium environment with render_mode='rgb_array'

        Returns:
            dict mapping view name → (grid_size×grid_size,) float32 array
        """
        grids = {}
        for view in self.views:
            try:
                frame = self._render_view(env, view)
                grid = self._enc.encode(frame)                        # (G, G) int
                grids[view] = grid.flatten().astype(np.float32) / 9.0  # → [0,1]
            except Exception:
                # If camera manipulation fails, return zeros for this view
                grids[view] = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        return grids

    def _render_view(self, env, view: str) -> np.ndarray:
        """Render one view. Falls back to default render if camera unavailable."""
        cam = self._cam_params[view]
        # Try to set camera via env.unwrapped.model if mujoco is available
        # Try to set camera via mujoco_renderer if available
        try:
            viewer = env.unwrapped.mujoco_renderer
            if viewer is not None and hasattr(viewer, 'viewer'):
                viewer.viewer.cam.distance = cam["distance"]
                viewer.viewer.cam.azimuth = cam["azimuth"]
                viewer.viewer.cam.elevation = cam["elevation"]
        except Exception:
            pass

        frame = env.render()
        if frame is None:
            return np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)
        # Resize to standard render_size if needed
        if frame.shape[:2] != (self.render_size, self.render_size):
            frame = self._resize(frame, self.render_size)
        return frame

    def _resize(self, frame: np.ndarray, size: int) -> np.ndarray:
        """Simple nearest-neighbor resize without cv2/PIL."""
        h, w = frame.shape[:2]
        row_idx = (np.arange(size) * h / size).astype(int)
        col_idx = (np.arange(size) * w / size).astype(int)
        return frame[np.ix_(row_idx, col_idx)]

    @property
    def view_dim(self) -> int:
        """Dimension contributed by visual views (all views combined)."""
        return len(self.views) * self.grid_size * self.grid_size


# ═══════════════════════════════════════════════════════════════
# MUJOCO ADAPTER
# ═══════════════════════════════════════════════════════════════

class MuJoCoAdapter:
    """
    Triple orthographic ASCII adapter for MuJoCo environments.

    When mujoco is installed:
        obs = [XY_grid | XZ_grid | YZ_grid | proprioception]
        obs_dim = len(views) × 225 + pro_obs_dim

    When mujoco / gymnasium-mujoco is NOT installed:
        obs = proprioception only (RAM-based features from gymnasium)
        obs_dim = pro_obs_dim
        Graceful degradation — the adapter still works, just without visual views.

    This keeps the system testable and trainable even without a MuJoCo license.

    Action space:
        Uses MuJoCoActionDiscretizer to map integer actions → continuous torques.
        Default: per-joint ternary {-1, 0, +1} (3^N_joints options).
    """

    def __init__(
        self,
        env_name: str = "Reacher-v4",
        views: Optional[List[str]] = None,
        grid_size: int = GRID_SIZE,
        render_size: int = 64,
        use_visual: bool = True,
        discretizer=None,          # MuJoCoActionDiscretizer (injected)
        seed: int = 42,
    ):
        """
        Args:
            env_name: MuJoCo gymnasium environment name.
            views: Which orthographic views to use. Default: ["xy", "xz", "yz"].
                   Ablation options: ["xy"], ["xy", "xz"].
            grid_size: ASCII grid resolution per view (default 15 → 225 per view).
            render_size: Pixel resolution for rendering before ASCII encode (64×64).
            use_visual: If False, use proprioception only (useful when no GPU).
            discretizer: MuJoCoActionDiscretizer instance. If None, created automatically.
            seed: Random seed.
        """
        self.env_name = env_name
        self.views = views or ["xy", "xz", "yz"]
        self.grid_size = grid_size
        self.render_size = render_size
        self.seed = seed
        self._spec = TASK_SPECS.get(env_name, {
            "n_joints": 2,
            "action_dim": 2,
            "pro_obs_dim": 10,
            "camera_distance": 1.5,
            "camera_elevation": 20,
            "description": "Unknown MuJoCo task",
        })

        # Can we actually use visual rendering?
        self.use_visual = use_visual and _MUJOCO_AVAILABLE

        # Claim the discretizer now (avoids circular import at module level)
        self._discretizer = discretizer

        # Gymnasium environment
        self._env = None
        self._env_name_actual = env_name

        # Renderer (lazy init when env is ready)
        self._renderer: Optional[TripleViewRenderer] = None

        # Proprioception dim (from task spec, or detected at first reset)
        self._pro_dim = self._spec.get("pro_obs_dim", 10)

        # Compute obs_dim
        view_dim = len(self.views) * grid_size * grid_size if self.use_visual else 0
        self._obs_dim = view_dim + self._pro_dim

        # Episode tracking
        self._step_count = 0
        self._total_reward = 0.0
        self._total_episodes = 0

    @property
    def obs_dim(self) -> int:
        """Total observation dimension."""
        return self._obs_dim

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        if self._discretizer is not None:
            return self._discretizer.n_actions
        # Default: ternary per joint
        n_joints = self._spec.get("n_joints", 2)
        return 3 ** n_joints

    @property
    def is_visual(self) -> bool:
        """True if visual rendering is active."""
        return self.use_visual

    def _make_env(self):
        """Create the gymnasium environment."""
        if not _MUJOCO_AVAILABLE:
            raise RuntimeError(
                "gymnasium with mujoco not installed. "
                "Install with: pip install gymnasium[mujoco]"
            )
        render_mode = "rgb_array" if self.use_visual else None
        self._env = gym.make(self._env_name_actual, render_mode=render_mode)

        if self.use_visual and self._renderer is None:
            self._renderer = TripleViewRenderer(
                env_name=self.env_name,
                grid_size=self.grid_size,
                render_size=self.render_size,
                views=self.views,
            )

    def _lazy_discretizer(self):
        """Import and create default discretizer on first use."""
        if self._discretizer is None:
            try:
                from src.games.mujoco.mujoco_action_discretizer import MuJoCoActionDiscretizer
            except ImportError:
                from mujoco_action_discretizer import MuJoCoActionDiscretizer  # type: ignore
            n_joints = self._spec.get("n_joints", 2)
            action_dim = self._spec.get("action_dim", n_joints)
            self._discretizer = MuJoCoActionDiscretizer(
                n_joints=n_joints,
                action_dim=action_dim,
                strategy="ternary",
            )

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment and return initial observation.

        Returns:
            obs: (obs_dim,) float32 array
        """
        if self._env is None:
            self._make_env()
        self._lazy_discretizer()

        seed = seed if seed is not None else self.seed + self._total_episodes
        ram_obs, _ = self._env.reset(seed=seed)
        ram_obs = np.asarray(ram_obs, dtype=np.float32).flatten()

        # Update pro_dim from actual obs (handles env-specific variations)
        if len(ram_obs) != self._pro_dim:
            self._pro_dim = len(ram_obs)
            view_dim = len(self.views) * self.grid_size * self.grid_size if self.use_visual else 0
            self._obs_dim = view_dim + self._pro_dim

        self._step_count = 0
        self._total_reward = 0.0
        self._total_episodes += 1

        return self._build_obs(ram_obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute a discrete action.

        Args:
            action: Integer action index (mapped to continuous torques by discretizer).

        Returns:
            (next_obs, reward, done, info)
        """
        if self._env is None:
            raise RuntimeError("Call reset() before step()")

        # Map discrete → continuous
        continuous_action = self._discretizer.decode(action)
        ram_obs, reward, terminated, truncated, info = self._env.step(continuous_action)
        ram_obs = np.asarray(ram_obs, dtype=np.float32).flatten()

        done = terminated or truncated
        self._step_count += 1
        self._total_reward += float(reward)

        obs = self._build_obs(ram_obs)
        return obs, float(reward), done, info

    def _build_obs(self, ram_obs: np.ndarray) -> np.ndarray:
        """Combine visual views + proprioception into one flat obs vector."""
        parts = []

        if self.use_visual and self._renderer is not None:
            view_grids = self._renderer.render_views(self._env)
            for v in self.views:
                parts.append(view_grids.get(v, np.zeros(self.grid_size ** 2, dtype=np.float32)))

        # Normalize proprioception: clip to [-10, 10], divide by 10
        pro = np.clip(ram_obs, -10.0, 10.0) / 10.0
        parts.append(pro)

        return np.concatenate(parts, axis=0).astype(np.float32)

    def close(self):
        """Clean up the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def stats(self) -> Dict[str, Any]:
        """Episode statistics."""
        return {
            "env": self.env_name,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "views": self.views,
            "visual": self.use_visual,
            "total_episodes": self._total_episodes,
            "step_count": self._step_count,
            "total_reward": round(self._total_reward, 3),
        }

    def __repr__(self) -> str:
        return (
            f"MuJoCoAdapter({self.env_name!r}, views={self.views}, "
            f"obs_dim={self.obs_dim}, n_actions={self.n_actions}, "
            f"visual={self.use_visual})"
        )


# ═══════════════════════════════════════════════════════════════
# NUMPY FALLBACK — no mujoco installed
# ═══════════════════════════════════════════════════════════════

class MuJoCoFallbackSim:
    """
    Pure numpy dummy simulator that mimics MuJoCoAdapter's interface
    without requiring gymnasium or mujoco.

    Used for unit testing and CI environments without mujoco.

    Simulates Reacher-like dynamics:
        State: joint angles (2) + joint velocities (2) + target (2) + fingertip XY (2) + distance (2) = 10
        Action: ternary torques on 2 joints
        Reward: -distance to target (shaped)
    """

    def __init__(
        self,
        n_joints: int = 2,
        obs_dim: int = 10,
        max_steps: int = 50,
        seed: int = 42,
    ):
        self.n_joints = n_joints
        self._obs_dim = obs_dim
        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.n_actions = 3 ** n_joints
        self.views = ["xy", "xz", "yz"]
        self.use_visual = False
        self.env_name = f"FallbackReacher(n_joints={n_joints})"

        self._rng = np.random.RandomState(seed)
        self._state: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        self._step_count = 0
        self._total_episodes = 0
        self._total_reward = 0.0

        # Ternary action decoding
        self._action_table = self._build_action_table()

    def _build_action_table(self) -> np.ndarray:
        """Build ternary action table: n_actions × n_joints, values in {-1, 0, +1}."""
        n = self.n_actions
        table = np.zeros((n, self.n_joints), dtype=np.float32)
        for i in range(n):
            code = i
            for j in range(self.n_joints):
                table[i, j] = float(code % 3) - 1.0   # 0→-1, 1→0, 2→+1
                code //= 3
        return table

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Random joint angles and velocities
        angles = self._rng.uniform(-np.pi, np.pi, self.n_joints).astype(np.float32)
        vels = np.zeros(self.n_joints, dtype=np.float32)
        # Random target
        self._target = self._rng.uniform(-0.2, 0.2, 2).astype(np.float32)
        # Fingertip position (simple forward kinematic for 2-joint arm)
        fingertip = self._forward_kinematics(angles)
        dist_vec = self._target - fingertip

        self._state = np.concatenate([angles, vels, self._target, fingertip, dist_vec])
        self._step_count = 0
        self._total_reward = 0.0
        self._total_episodes += 1
        return self._state.copy()

    def _forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """
        Compute fingertip XY for a serial N-joint arm (link length 0.1 each).
        Projects all joints into XY plane (sufficient for a reaching task).
        """
        L = 0.1
        x, y = 0.0, 0.0
        cum_angle = 0.0
        for angle in angles:
            cum_angle += angle
            x += L * np.cos(cum_angle)
            y += L * np.sin(cum_angle)
        return np.array([x, y], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        torques = self._action_table[action]   # (n_joints,)

        angles = self._state[:self.n_joints]
        vels = self._state[self.n_joints:2 * self.n_joints]

        # Simple Euler integration
        dt = 0.05
        vels = np.clip(vels + torques * dt, -1.0, 1.0)
        angles = angles + vels * dt

        fingertip = self._forward_kinematics(angles)
        dist_vec = self._target - fingertip
        distance = float(np.linalg.norm(dist_vec))

        # Shaped reward: negative distance + success bonus
        reward = -distance
        if distance < 0.01:
            reward += 1.0

        self._state = np.concatenate([
            angles, vels, self._target, fingertip, dist_vec
        ])
        self._step_count += 1
        self._total_reward += reward
        done = self._step_count >= self.max_steps

        info = {"distance": distance, "success": distance < 0.01}
        return self._state.copy(), reward, done, info

    def stats(self) -> Dict[str, Any]:
        return {
            "env": self.env_name,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "views": self.views,
            "visual": False,
            "total_episodes": self._total_episodes,
            "step_count": self._step_count,
            "total_reward": round(self._total_reward, 3),
        }

    def close(self):
        """No-op — FallbackSim has no resources to release."""
        pass


def make_mujoco_adapter(
    env_name: str = "Reacher-v4",
    views: Optional[List[str]] = None,
    use_visual: bool = True,
    seed: int = 42,
) -> "MuJoCoAdapter | MuJoCoFallbackSim":
    """
    Factory that returns a MuJoCoAdapter if mujoco is available,
    otherwise a MuJoCoFallbackSim.

    This is the recommended entry point for training scripts:
        adapter = make_mujoco_adapter("Reacher-v4")
        obs = adapter.reset()
    """
    if not _MUJOCO_AVAILABLE:
        spec = TASK_SPECS.get(env_name, {})
        n_joints = spec.get("n_joints", 2)
        obs_dim = spec.get("pro_obs_dim", 10)
        return MuJoCoFallbackSim(n_joints=n_joints, obs_dim=obs_dim, seed=seed)
    return MuJoCoAdapter(
        env_name=env_name,
        views=views,
        use_visual=use_visual,
        seed=seed,
    )
