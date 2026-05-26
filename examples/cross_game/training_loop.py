"""Cross-game training loop: encoder warm-up, joint WM training, transfer test."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np

_TORCH_AVAILABLE = True
try:
    from src.cell.world_model import MultiGameWorldModel
except ImportError:
    _TORCH_AVAILABLE = False
    MultiGameWorldModel = None  # type: ignore[misc, assignment]

_GYM_AVAILABLE = True
try:
    import gymnasium as gym  # noqa: F401
except ImportError:
    _GYM_AVAILABLE = False

_MARIO_AGENT_AVAILABLE = True
try:
    from src.games.mario.mario_agent import make_mario_agent
except ImportError:
    _MARIO_AGENT_AVAILABLE = False
    make_mario_agent = None

_META_ENCODER_AVAILABLE = True
try:
    from src.encoder.meta_encoder import MetaEncoder
except ImportError:
    _META_ENCODER_AVAILABLE = False
    MetaEncoder = None

_DUAL_MODE_AVAILABLE = True
try:
    from src.encoder.dual_mode_encoder import DualModeEncoder
except ImportError:
    _DUAL_MODE_AVAILABLE = False
    DualModeEncoder = None

from src.cell.dreamer import CellDreamer
from src.encoder.go_explore_adapter import GoExploreRunner, ZCellArchive
from src.encoder.universal_encoder import EncoderRegistry, get_n_games
from src.learning.checkpoint_manager import CheckpointManager
from src.learning.elite_replay import EliteReplayManager
from src.learning.numpy_linear_agent import SimpleNumpyAgent

from examples.cross_game.episode import compute_dream_accuracy, run_episode
from examples.cross_game.runners import GAME_CONFIGS, gym_obs_dim_for_env


def warm_up_encoders(
    runners: Dict[str, object],
    encoder: EncoderRegistry,
    n_obs: int = 300,
    contrastive: bool = True,
    contrastive_epochs: int = 20,
    verbose: bool = True,
) -> Dict[str, bool]:
    """Collect random obs, fit PCA, then optional contrastive pre-training."""
    obs_by_game: Dict[str, list] = {}

    for game_name, runner in runners.items():
        obs_list = []
        try:
            obs = runner.reset()
            obs_list.append(obs)
            for _ in range(n_obs - 1):
                action = np.random.randint(
                    runner.n_actions if hasattr(runner, "n_actions") else 4
                )
                result = runner.step(action)
                next_obs = result[0]
                obs_list.append(next_obs)
                if result[2]:
                    obs = runner.reset()
                    obs_list.append(obs)
                else:
                    obs = next_obs
        except Exception as e:
            if verbose:
                print(f"  warm_up_encoders: skipped {game_name!r} ({e})")
            continue
        obs_by_game[game_name] = obs_list

    encoder.fit_all(obs_by_game)
    pca_fitted = {
        g: getattr(
            getattr(encoder._encoders[g], "_projection", None),
            "is_pca_fitted",
            False,
        )
        for g in encoder._encoders
    }
    if verbose:
        for g, f in pca_fitted.items():
            print(f"  encoder[{g!r}]: {'PCA fitted' if f else 'random (skipped)'}")

    if contrastive and obs_by_game:
        if verbose:
            print(
                f"  warm_up_encoders: contrastive pre-training "
                f"({contrastive_epochs} epochs)..."
            )
        encoder.fit_contrastive_all(
            obs_by_game, n_epochs=contrastive_epochs, verbose=verbose
        )
        if verbose:
            for g, f in encoder.is_contrastive_fitted.items():
                print(
                    f"  encoder[{g!r}]: "
                    f"{'contrastive fitted' if f else 'not fitted'}"
                )

    return pca_fitted


def run_training(
    games: List[str],
    total_episodes: int = 300,
    wm_train_steps_per_episode: int = 5,
    max_steps_per_episode: int = 400,
    z_dim: int = 32,
    log_every: int = 20,
    save_path: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
    elite_n: int = 3,
    checkpoint_every: int = 50,
    checkpoint_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    use_dual_mode: bool = False,
) -> Dict:
    """Run joint cross-game world model training."""
    import json

    t0 = time.time()
    np.random.seed(seed)

    enc = EncoderRegistry(z_dim=z_dim, games=games)
    out_dim = enc.out_dim

    if verbose:
        print("═══ Cross-Game World Model Training ═══")
        print(f"  Games: {games}")
        print(f"  z_dim={z_dim}, out_dim={out_dim}, episodes/game={total_episodes}")
        print(f"  Torch available: {_TORCH_AVAILABLE}")
        print(f"  Gym available: {_GYM_AVAILABLE}")
        print()

    world_model = None
    if _TORCH_AVAILABLE:
        max_actions = max(GAME_CONFIGS[g]["n_actions"] for g in games)
        world_model = MultiGameWorldModel(
            feature_dim=out_dim,
            n_actions=max_actions,
            n_games=get_n_games(),
            game_embed_dim=8,
            hidden_size=128,
            lr=1e-3,
            buffer_size=3000,
            batch_size=64,
            min_transitions=50,
        )

    runners = {}
    agents = {}
    for g in games:
        cfg = GAME_CONFIGS[g]
        try:
            runners[g] = cfg["runner_cls"](**cfg["runner_kwargs"], seed=seed)
        except Exception as e:
            print(f"  [WARN] Could not init runner for {g}: {e}")
            continue

        n_actions = cfg["n_actions"]
        if g == "mario":
            if _MARIO_AGENT_AVAILABLE and make_mario_agent is not None:
                agents[g] = make_mario_agent(
                    obs_dim=378, n_actions=n_actions, curiosity=True, backend="numpy"
                )
            else:
                agents[g] = SimpleNumpyAgent(obs_dim=378, n_actions=n_actions)
        else:
            gym_env_name = cfg["runner_kwargs"].get("env_name", "").lower()
            obs_dim = next(
                (v for k, v in _gym_obs_dims.items() if k in gym_env_name),
                8,
            )
            agents[g] = SimpleNumpyAgent(obs_dim=obs_dim, n_actions=n_actions)

    active_games = [g for g in games if g in runners]
    if not active_games:
        print("[ERROR] No games could be initialized")
        return {}

    if verbose:
        print("  Warm-up: collecting observations to fit PCA projections...")
    warm_up_encoders(
        {g: runners[g] for g in active_games}, enc, n_obs=300, verbose=verbose
    )
    if verbose:
        print()

    go_archives = {g: ZCellArchive(z_dim=z_dim) for g in active_games}
    go_runner = GoExploreRunner(archive=None)
    explore_every = max(1, total_episodes // 20)

    meta_encoder = None
    if _META_ENCODER_AVAILABLE:
        meta_encoder = MetaEncoder(z_dim=z_dim)
        if verbose:
            print("  MetaEncoder: enabled for cross-game archive routing.")
            print()

    dual_encoders: Dict[str, DualModeEncoder] = {}
    if use_dual_mode and _DUAL_MODE_AVAILABLE:
        for g in active_games:
            try:
                dual_encoders[g] = DualModeEncoder(game_name=g, z_dim=z_dim)
            except Exception as e:
                if verbose:
                    print(f"  [WARN] DualModeEncoder for {g}: {e}")
        if verbose and dual_encoders:
            print(f"  DualModeEncoder: enabled for {list(dual_encoders.keys())}")
            print()

    ckpt_base = checkpoint_path or "results/checkpoints"
    elite_replay = EliteReplayManager(games=active_games, n=elite_n)
    ckpt_manager = CheckpointManager(ckpt_base, keep_last=3)
    start_episode = 0

    if resume_from:
        try:
            start_episode = ckpt_manager.load(
                resume_from, world_model, enc, go_archives, elite_replay
            )
            if verbose:
                print(f"  Resumed from checkpoint: episode {start_episode}")
                print(f"  Elite replay: {elite_replay.stats()}")
        except Exception as e:
            if verbose:
                print(f"  [WARN] Could not load checkpoint: {e} -- starting fresh")
        if verbose:
            print()

    history = {
        g: {"rewards": [], "steps": [], "wm_loss": [], "surprise": []}
        for g in active_games
    }
    wm_history = []

    for ep in range(total_episodes):
        ep_results = {}
        all_transitions = []

        for g in active_games:
            result = run_episode(
                runners[g],
                agents[g],
                enc,
                g,
                world_model,
                max_steps=max_steps_per_episode,
                dual_encoders=dual_encoders or None,
                meta_encoder=meta_encoder,
                z_dim=z_dim,
                verbose=verbose,
            )
            ep_results[g] = result
            all_transitions.extend(result["transitions"])
            history[g]["rewards"].append(result["total_reward"])
            history[g]["steps"].append(result["steps"])

            if result["z_seq"]:
                cumulative_r = result["total_reward"]
                for step_i, z in enumerate(result["z_seq"]):
                    go_archives[g].add(z, score=cumulative_r, trajectory_len=step_i + 1)

            actions = [a for (_, a, _, _, _) in result["transitions"]]
            elite_replay.try_add(
                game=g,
                actions=actions,
                score=result["total_reward"],
                episode=ep + start_episode,
            )

        if (ep + 1) % explore_every == 0:
            for g in active_games:
                go_runner.archive = go_archives[g]
                try:
                    go_stats = go_runner.run_explore_episode(
                        runners[g],
                        enc,
                        game_name=g,
                        n_random_steps=30,
                        max_episode_steps=100,
                    )
                    if verbose:
                        print(
                            f"    [GoExplore] {g}: "
                            f"+{go_stats['n_new_cells']} new cells, "
                            f"archive={go_archives[g].size}"
                        )
                except Exception as e:
                    if verbose:
                        print(f"    [GoExplore] {g}: skipped ({e})")

            if meta_encoder is not None and len(meta_encoder.known_games()) >= 2:
                for g in active_games:
                    desc = meta_encoder.descriptor(g)
                    if desc is None:
                        continue
                    src_game, sim = meta_encoder.nearest_game(desc, exclude=g)
                    if src_game and go_archives[src_game].size >= 5:
                        top_cells = go_archives[src_game].top_k_cells(k=10)
                        n_seeded = go_archives[g].seed_from(top_cells)
                        if verbose and n_seeded > 0:
                            print(
                                f"    [GoExplore] MetaEncoder seeded {n_seeded} cells "
                                f"from {src_game} -> {g} (sim={sim:.3f})"
                            )

            for g in active_games:
                g_id = GAME_CONFIGS[g]["game_id"]
                inject_stats = elite_replay.inject_episode(
                    runners[g], enc, g, world_model, g_id, force=False, p=0.20
                )
                if inject_stats and verbose:
                    print(
                        f"    [Elite] {g}: replayed {inject_stats['label']} "
                        f"score={inject_stats['elite_score']:.1f} "
                        f"+{inject_stats['transitions_added']} transitions"
                    )

        if meta_encoder is not None and (ep + 1) % max(1, log_every) == 0:
            meta_encoder.fit_projection(min_episodes_per_game=3, verbose=False)

        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0:
            try:
                ckpt_dir = ckpt_manager.save(
                    ep + start_episode + 1,
                    world_model,
                    enc,
                    go_archives,
                    elite_replay,
                )
                if verbose:
                    print(
                        f"    [Checkpoint] saved ep={ep + start_episode + 1} -> {ckpt_dir}"
                    )
            except Exception as e:
                if verbose:
                    print(f"    [Checkpoint] save failed: {e}")

        wm_metrics = {}
        if world_model is not None and world_model._multi_buffer.size > 0:
            for _ in range(wm_train_steps_per_episode):
                wm_metrics = world_model.train_step_multi_game()
            wm_history.append(wm_metrics)

        if verbose and (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  Episode {ep + 1}/{total_episodes}  ({elapsed:.0f}s)")
            for g in active_games:
                rews = history[g]["rewards"]
                avg_r = float(np.mean(rews[-log_every:]))
                surp = (
                    world_model.surprise(GAME_CONFIGS[g]["game_id"])
                    if world_model
                    else 0.0
                )
                print(f"    {g:12s}: avg_r={avg_r:+8.2f}  surprise={surp:.4f}")
            if wm_metrics:
                print(
                    f"    WM: loss={wm_metrics.get('wm_loss', 0):.4f}  "
                    f"buffer={wm_metrics.get('wm_buffer', 0)}"
                )
            if world_model and world_model.is_ready:
                da = compute_dream_accuracy(world_model, all_transitions[:20])
                print(f"    Dream accuracy: {da:.1%}")
            print()

    results = {
        "games": active_games,
        "z_dim": z_dim,
        "total_episodes": total_episodes,
        "history": {
            g: {
                "avg_reward_last20": float(np.mean(history[g]["rewards"][-20:])),
                "rewards": history[g]["rewards"][-50:],
            }
            for g in active_games
        },
        "wm_ready": bool(world_model and world_model.is_ready),
        "wm_stats": world_model.multi_stats() if world_model else {},
        "elapsed": round(time.time() - t0, 1),
        "world_model": world_model,
        "encoder": enc,
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        serializable = {
            k: v
            for k, v in results.items()
            if k not in ("world_model", "encoder")
        }
        with open(save_path, "w") as f:
            json.dump(
                serializable,
                f,
                indent=2,
                default=lambda x: float(x) if hasattr(x, "__float__") else str(x),
            )
        if verbose:
            print(f"  Results saved to: {save_path}")

    if verbose:
        print("\n═══ Training Complete ═══")
        for g in active_games:
            avg = float(np.mean(history[g]["rewards"][-20:]))
            print(f"  {g:12s}: last-20 avg reward = {avg:+.2f}")
        if world_model:
            print(f"  World model: {world_model.multi_stats()}")

    return results


def run_transfer_test(
    world_model: MultiGameWorldModel,
    encoder: EncoderRegistry,
    target_game: str = "lunarlander",
    solve_threshold: float = 200.0,
    max_episodes: int = 500,
    verbose: bool = True,
) -> Dict:
    """Freeze WM and compare solve speed with vs without dream guidance."""
    if not _GYM_AVAILABLE:
        print("[SKIP] gymnasium not available, skipping transfer test")
        return {}

    results = {}
    cfg = GAME_CONFIGS.get(target_game)
    if cfg is None:
        print(f"[ERROR] Unknown target game: {target_game}")
        return {}

    for mode in ["without_wm", "with_wm"]:
        try:
            runner = cfg["runner_cls"](**cfg["runner_kwargs"])
        except Exception as e:
            print(f"[ERROR] {e}")
            break

        gym_env_name = cfg["runner_kwargs"].get("env_name", "")
        obs_dim = gym_obs_dim_for_env(gym_env_name)
        agent = SimpleNumpyAgent(obs_dim=obs_dim, n_actions=cfg["n_actions"])
        episodes_to_solve = max_episodes

        for ep in range(max_episodes):
            obs = runner.reset()
            total_r = 0.0
            for step in range(1000):
                policy_action = agent.step(obs)
                if mode == "with_wm":
                    gid = cfg["game_id"]
                    z = encoder.encode(target_game, obs)
                    n_act = cfg["n_actions"]
                    action = CellDreamer.guided_training_action(
                        policy_action,
                        z,
                        world_model,
                        gid,
                        n_act,
                        step,
                        dream_eps_start=0.3,
                        dream_eps_end=0.05,
                        dream_eps_decay=0.002,
                        depth=2,
                    )
                else:
                    action = policy_action

                result = runner.step(action)
                next_obs, reward, done = result[0], result[1], result[2]
                agent.learn(reward, next_obs, done)
                total_r += reward
                obs = next_obs
                if done:
                    break

            if total_r >= solve_threshold:
                episodes_to_solve = ep + 1
                break

        results[mode] = episodes_to_solve
        if verbose:
            print(f"  Transfer ({mode}): solved in {episodes_to_solve} episodes")

    if "with_wm" in results and "without_wm" in results:
        wo, wi = results["without_wm"], results["with_wm"]
        improvement = (wo - wi) / max(1, wo) * 100
        results["improvement_pct"] = round(improvement, 1)
        if verbose:
            print(
                f"  Transfer efficiency: {improvement:.1f}% fewer episodes with world model"
            )

    return results
