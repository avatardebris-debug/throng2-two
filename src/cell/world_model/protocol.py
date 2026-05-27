"""Typed helpers for game-conditioned world models (dreaming + prediction)."""
from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class GameConditionedWorldModel(Protocol):
    """Minimum contract for cross-game dreaming and guided training."""

    @property
    def is_ready(self) -> bool: ...

    @property
    def n_actions(self) -> int: ...

    def is_ready_for(self, game_id: int) -> bool: ...

    def dream_all_actions(
        self,
        features: np.ndarray,
        *,
        depth: int = 1,
        game_id: int = 0,
    ) -> np.ndarray: ...


@runtime_checkable
class HorizonDreaming(Protocol):
    """Multi-game slow path (N-step horizon head)."""

    def dream_horizon(self, features: np.ndarray, game_id: int = 0) -> np.ndarray: ...

    def adaptive_horizon_n(self, game_id: int = 0) -> int: ...


@runtime_checkable
class MultiStepPredictor(Protocol):
    """Per-game next-state prediction."""

    def predict_multi(
        self,
        features: np.ndarray,
        action: int,
        game_id: int,
    ) -> Tuple[np.ndarray, float]: ...


def is_ready_for_game(world_model: Optional[object], game_id: int) -> bool:
    if world_model is None:
        return False
    if isinstance(world_model, GameConditionedWorldModel):
        return world_model.is_ready_for(game_id)
    return bool(getattr(world_model, "is_ready", False))


def dream_all_actions_for_game(
    world_model: object,
    features: np.ndarray,
    *,
    depth: int = 1,
    game_id: int = 0,
) -> np.ndarray:
    return world_model.dream_all_actions(features, depth=depth, game_id=game_id)


def has_horizon_dreaming(world_model: object) -> bool:
    return isinstance(world_model, HorizonDreaming)


def prediction_error(
    world_model: object,
    z: np.ndarray,
    action: int,
    next_z: np.ndarray,
    game_id: int,
) -> float:
    """Mean |predicted_z - actual_z| for meta-encoder surprise."""
    if not is_ready_for_game(world_model, game_id):
        return 0.0
    if isinstance(world_model, MultiStepPredictor):
        pred_z, _ = world_model.predict_multi(z, action, game_id)
    else:
        pred_z, _, _ = world_model.predict(z, action)
    return float(np.mean(np.abs(pred_z - next_z)))
