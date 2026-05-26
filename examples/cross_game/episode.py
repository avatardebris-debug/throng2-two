"""Single-episode rollout for cross-game training."""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.cell.dreamer import CellDreamer
from src.cell.world_model.protocol import (
    dream_all_actions_for_game,
    is_ready_for_game,
    prediction_error,
)
from src.encoder.universal_encoder import EncoderRegistry

if TYPE_CHECKING:
    from src.cell.world_model import MultiGameWorldModel
    from src.encoder.dual_mode_encoder import DualModeEncoder
    from src.encoder.meta_encoder import EpisodeSummary, MetaEncoder

try:
    from src.encoder.meta_encoder import EpisodeSummary
except ImportError:
    EpisodeSummary = None


def run_episode(
    runner,
    agent,
    encoder: EncoderRegistry,
    game_name: str,
    world_model: Optional["MultiGameWorldModel"],
    max_steps: int = 500,
    dual_encoders: Optional[Dict[str, "DualModeEncoder"]] = None,
    meta_encoder: Optional["MetaEncoder"] = None,
    z_dim: int = 32,
    verbose: bool = False,
) -> Dict:
    """Run one episode; collect z-seq and world-model transitions."""
    obs = runner.reset()
    agent.reset()
    total_reward = 0.0
    steps = 0
    transitions = []
    z_seq = []

    game_id = encoder.game_id(game_name)
    prev_z = encoder.encode(game_name, obs)
    z_seq.append(prev_z)

    episode_summary = None
    if meta_encoder is not None and EpisodeSummary is not None:
        episode_summary = EpisodeSummary(
            z_dim=z_dim,
            max_steps=max_steps,
            n_actions=getattr(runner, "n_actions", 0),
        )

    for step in range(max_steps):
        dual_encoder = (
            dual_encoders.get(game_name) if dual_encoders is not None else None
        )
        if dual_encoder is not None and world_model is not None:
            try:
                dual_encoder.surprise_auto_mode(
                    world_model, game_id, obs,
                    threshold=0.4, window=3, use_map=True,
                )
            except Exception as exc:
                if verbose:
                    print(f"  [dual-mode] {game_name}: {exc}")

        action = agent.step(obs)
        n_act = getattr(runner, "n_actions", 8)
        action = CellDreamer.guided_training_action(
            action, prev_z, world_model, game_id, n_act, step, depth=1
        )

        result = runner.step(action)
        next_obs, reward, done = result[0], result[1], result[2]

        next_z = encoder.encode(game_name, next_obs)
        z_seq.append(next_z)

        if world_model is not None:
            transitions.append((prev_z, action, next_z, reward, game_id))
            world_model.store_transition(prev_z, action, next_z, reward, game_id)

        if episode_summary is not None:
            surprise = prediction_error(
                world_model, prev_z, action, next_z, game_id
            )
            episode_summary.record(prev_z, action, reward, surprise)

        if hasattr(agent, "learn_with_next_obs"):
            agent.learn_with_next_obs(reward, done, next_obs)
        elif hasattr(agent, "learn"):
            agent.learn(reward, next_obs, done)

        total_reward += reward
        prev_z = next_z
        obs = next_obs
        steps += 1

        if done:
            break

    if episode_summary is not None and meta_encoder is not None:
        meta_encoder.update(game_name, episode_summary.build())

    return {
        "total_reward": total_reward,
        "steps": steps,
        "z_seq": z_seq,
        "transitions": transitions,
        "game": game_name,
    }


def compute_dream_accuracy(
    world_model: "MultiGameWorldModel",
    transitions: List,
    n_samples: int = 20,
) -> float:
    """Fraction of samples where dream argmax matches the real action."""
    if not transitions:
        return 0.0

    hits = 0
    evaluated = 0
    sample = transitions[: min(n_samples, len(transitions))]
    for z, real_action, _z_next, _real_reward, game_id in sample:
        if not is_ready_for_game(world_model, game_id):
            continue
        dream_vals = dream_all_actions_for_game(
            world_model, z, depth=1, game_id=game_id
        )
        if int(np.argmax(dream_vals)) == int(real_action):
            hits += 1
        evaluated += 1
    return hits / max(1, evaluated)
