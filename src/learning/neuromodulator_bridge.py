"""
neuromodulator_bridge.py — Integration layer between biological neuromodulators
and the ThrongletCell training loop.

This module wires together the existing (but previously unused) modules:
    src/learning/dopamine.py    → DopamineSystem  (RPE, lr modulation)
    src/learning/stdp.py        → STDPLearning     (eligibility traces)
    src/learning/spatial_memory.py → SpatialMemory (reward location recall)

into a single ThrongletCell-compatible callback object.

Design principle:
    The bridge is ADVISORY: it computes modulated learning rates and
    eligibility-weighted updates, but doesn't replace the existing PPO/DQN
    gradient flow. Instead it:
        1. Returns a `lr_multiplier` that the cell's optimizer can apply.
        2. Maintains spatial memory so the cell can navigate toward previously
           rewarded regions.
        3. Logs neuromodulator state for diagnostics/tensorboard.

Usage (inside ThrongletCell.train_step):

    bridge = NeuromodulatorBridge(n_neurons=64)

    # After each step:
    lr_mult = bridge.step(
        reward=r,
        active_neurons=snn_output_indices,  # which SNN neurons fired
        position=z[:2],                     # first 2 dims of z as "position"
    )
    # Modify learning rate:
    for group in optimizer.param_groups:
        group['lr'] = base_lr * lr_mult

    # Query spatial memory:
    target_pos = bridge.recall_goal()    # (2,) or None

Compatible with: CellWorldModel, MarioICMAgent, SimpleNumpyAgent.
No PyTorch dependency — pure numpy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.learning.dopamine import DopamineSystem
from src.learning.stdp import STDPLearning
from src.learning.spatial_memory import SpatialMemory


# ═══════════════════════════════════════════════════════════════
# NEUROMODULATOR BRIDGE
# ═══════════════════════════════════════════════════════════════

class NeuromodulatorBridge:
    """
    Wraps DopamineSystem + STDPLearning + SpatialMemory into one interface.

    Key outputs:
        lr_multiplier  — scale the optimizer's learning rate this step
        eligibility    — {(pre, post): weight} STDP updates (optional)
        goal_position  — spatial memory recall (where reward was found)

    The bridge is designed to be non-intrusive: if the cell doesn't use
    lr_multiplier or eligibility, nothing breaks.
    """

    def __init__(
        self,
        n_neurons: int = 64,
        dopamine_lr: float = 0.1,
        stdp_tau: float = 0.020,
        memory_decay: float = 0.95,
        lr_modulation_strength: float = 0.3,
        min_lr_mult: float = 0.5,
        max_lr_mult: float = 2.0,
        verbose: bool = False,
    ):
        """
        Args:
            n_neurons: Number of SNN neurons to track for STDP.
            dopamine_lr: How fast expected reward updates (0.1 = TD(0.1)).
            stdp_tau: STDP time constant in seconds.
            memory_decay: Spatial memory decay per step (0.95 = lose 5%/step).
            lr_modulation_strength: How much RPE multiplies lr (0 = no effect).
            min_lr_mult: Minimum lr multiplier (clamp).
            max_lr_mult: Maximum lr multiplier (clamp).
            verbose: Print debug info.
        """
        self.n_neurons = n_neurons
        self.lr_modulation_strength = lr_modulation_strength
        self.min_lr_mult = min_lr_mult
        self.max_lr_mult = max_lr_mult
        self.verbose = verbose

        # Simulated timestep: each call to step() advances sim time by SIM_DT seconds.
        # This makes STDP windows (tau=20ms) fire correctly regardless of wall-clock speed.
        self.SIM_DT = 0.020   # 20 ms per sim step (matches STDP tau defaults)

        # Suppress stdout from legacy print() statements in these modules
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            self._dopamine = DopamineSystem(
                baseline=0.0,
                learning_rate=dopamine_lr,
            )
            self._stdp = STDPLearning(
                tau_plus=stdp_tau,
                tau_minus=stdp_tau,
                A_plus=0.01,
                A_minus=0.01,
            )
            self._memory = SpatialMemory(decay=memory_decay)

        # Running stats
        self._step_count = 0
        self._sim_time = 0.0         # simulated clock (seconds)
        self._rpe_history: List[float] = []
        self._lr_history: List[float] = []

    # ── MAIN STEP ──────────────────────────────────────────────

    def step(
        self,
        reward: float,
        active_neurons: Optional[List[int]] = None,
        position: Optional[np.ndarray] = None,
    ) -> float:
        """
        Process one environment step through all three neuromodulators.

        Args:
            reward: Reward received this step.
            active_neurons: List of SNN neuron indices that fired.
                            If None, STDP is skipped.
            position: 2D position vector (e.g., z[:2]) for spatial memory.
                      If None, spatial memory update is skipped.

        Returns:
            lr_multiplier: Float in [min_lr_mult, max_lr_mult].
                           Multiply the base learning rate by this value.
        """
        self._step_count += 1
        self._sim_time += self.SIM_DT   # advance simulated clock

        # 1. Dopamine (RPE)
        rpe = self._dopamine.compute_rpe(reward)
        self._rpe_history.append(rpe)

        # 2. lr modulation from dopamine
        base_mult = 1.0 + self.lr_modulation_strength * np.tanh(rpe)
        lr_mult = float(np.clip(base_mult, self.min_lr_mult, self.max_lr_mult))
        self._lr_history.append(lr_mult)

        # 3. STDP eligibility update — use simulated time so windows fire correctly
        if active_neurons is not None and len(active_neurons) >= 2:
            # Only pass neurons that are valid indices
            valid = [int(n) % self.n_neurons for n in active_neurons]
            self._stdp.update_eligibility(valid, self._sim_time)

        # 4. Spatial memory
        if position is not None and reward > 0:
            pos = np.asarray(position, dtype=np.float32).flatten()[:2]
            if len(pos) < 2:
                pos = np.pad(pos, (0, 2 - len(pos)))
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                self._memory.add(pos, reward)

        # Periodic decay of spatial memory
        if self._step_count % 100 == 0:
            self._memory.decay_memory()

        if self.verbose:
            print(f"  Bridge step={self._step_count}: "
                  f"rpe={rpe:+.3f}, lr_mult={lr_mult:.3f}, "
                  f"dopamine={self._dopamine.level:.3f}")

        return lr_mult

    # ── ACCESSORS ──────────────────────────────────────────────

    def lr_multiplier(self, reward: float) -> float:
        """
        Compute lr multiplier without updating state.
        Useful for one-shot queries outside the main loop.
        """
        rpe = reward - self._dopamine.expected_reward
        mult = 1.0 + self.lr_modulation_strength * np.tanh(rpe)
        return float(np.clip(mult, self.min_lr_mult, self.max_lr_mult))

    def get_eligibility(self) -> Dict[Tuple[int, int], float]:
        """
        Return the current STDP eligibility traces.

        Returns:
            Dict mapping (pre_neuron, post_neuron) → weight_change
            Empty dict if no SNN output was provided this step.
        """
        return self._stdp.get_updates()

    def apply_dopamine_to_eligibility(self) -> Dict[Tuple[int, int], float]:
        """
        Apply dopamine to STDP eligibility (three-factor rule).

        Three-factor: dw = STDP × dopamine_level
        Returns modulated weight updates ready to apply.
        """
        eligibility = self._stdp.get_updates()
        dopamine_level = self._dopamine.level
        modulated = {
            syn: dw * dopamine_level
            for syn, dw in eligibility.items()
        }
        return modulated

    def recall_goal(self) -> Optional[np.ndarray]:
        """
        Recall the most reward-rich spatial position (or None if no memory).

        Returns:
            (2,) float32 array, or None
        """
        return self._memory.recall()

    def memory_confidence(self) -> float:
        """Return spatial memory confidence (0–1)."""
        return self._memory.confidence()

    def reset_episode(self):
        """Reset per-episode state (keep long-term memory and STDP history)."""
        self._dopamine.reset()
        self._stdp.clear_eligibility()
        # Do NOT clear spatial memory — it persists across episodes

    def reset_all(self):
        """Full reset including spatial memory."""
        self._dopamine.reset()
        self._stdp.clear_eligibility()
        self._stdp.spike_times.clear()   # also clear spike history to avoid cross-episode STDP
        self._memory.clear()
        self._step_count = 0
        self._sim_time = 0.0
        self._rpe_history = []
        self._lr_history = []

    # ── DIAGNOSTICS ────────────────────────────────────────────

    def stats(self) -> Dict:
        """Neuromodulator statistics."""
        rpe_arr = np.array(self._rpe_history[-100:]) if self._rpe_history else np.array([0.0])
        lr_arr = np.array(self._lr_history[-100:]) if self._lr_history else np.array([1.0])
        return {
            "step_count": self._step_count,
            "dopamine_level": round(float(self._dopamine.level), 4),
            "expected_reward": round(float(self._dopamine.expected_reward), 4),
            "rpe_mean": round(float(rpe_arr.mean()), 4),
            "rpe_std": round(float(rpe_arr.std()), 4),
            "lr_mult_mean": round(float(lr_arr.mean()), 4),
            "memory_locations": len(self._memory.locations),
            "memory_confidence": round(self._memory.confidence(), 3),
            "stdp_eligible_synapses": len(self._stdp.get_updates()),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"NeuromodulatorBridge("
            f"steps={s['step_count']}, "
            f"dopamine={s['dopamine_level']:+.3f}, "
            f"memory={s['memory_locations']}loc, "
            f"lr_mult={s['lr_mult_mean']:.3f})"
        )
