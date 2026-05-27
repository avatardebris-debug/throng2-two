"""Shared greedy dream rollout for single- and multi-game world models."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

PredictStep = Callable[[np.ndarray, int], Tuple[np.ndarray, float]]


def greedy_action_values(
    predict_step: PredictStep,
    state: np.ndarray,
    n_actions: int,
    depth: int = 1,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    Greedy rollout: for each first action, roll depth-1 steps with myopic best follow-up.

    predict_step(state, action) -> (next_state, reward)
    """
    action_values = np.zeros(n_actions, dtype=np.float64)
    for a in range(n_actions):
        next_feat, r = predict_step(state, a)
        total_return = r
        feat = next_feat
        for d in range(1, depth):
            best_r = -float("inf")
            best_a = 0
            for a2 in range(n_actions):
                _, pr = predict_step(feat, a2)
                if pr > best_r:
                    best_r = pr
                    best_a = a2
            feat, r = predict_step(feat, best_a)
            total_return += (gamma**d) * r
        action_values[a] = total_return
    return action_values.astype(np.float32)
