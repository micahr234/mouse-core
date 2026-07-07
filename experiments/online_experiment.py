#!/usr/bin/env python
"""Online ICRL experiment runner for MOUSE on Procedural FrozenLake.

Trains a MOUSE DQN model online (as in examples/03_train_online.ipynb) under a
configurable set of hyperparameters, and periodically evaluates greedy
performance from a *fresh context* on:

  * held-out maps (map_seed offset +1_000_000, as in examples/04_inference.ipynb)
  * training maps (same map_seed band as training, but a fresh reset stream)

so memorization and generalization can be separated. Eval also records
wall-bump and stuck-loop diagnostics to quantify the "keeps running into the
wall / repeats behavior" failure mode.

Results (config + per-cycle training log + eval metrics) are written as JSON to
experiments/results/<name>.json.

Run (GPU required):
    .venv/bin/python experiments/online_experiment.py --name baseline
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import procedural_frozenlake  # noqa: F401 — registers Procedural-FrozenLake-v1
from mouse_gym import EnvConfig, make_env
from mouse_core.data import Augmenter, DataLoader, Datastore
from mouse_core.models import Model
from mouse_core.models.backbone import Qwen3Backbone
from mouse_core.models.embedding import StepEmbedder
from mouse_core.models.heads import DiscreteActionValueHead
from mouse_core.objectives import DqnObjective

MAX_ACTIONS = 4
MAX_OBS_DISCRETE = 64
HOLDOUT_MAP_OFFSET = 1_000_000
EVAL_RESET_OFFSET = 500_000


@dataclasses.dataclass
class Config:
    name: str = "baseline"
    seed: int = 0

    # Model
    backbone_layers: int | None = None  # None = full Qwen3-0.6B (28 layers)

    # Envs / rollout
    num_envs: int = 1000
    env_steps_per_cycle: int = 1000
    steps_per_env: int = 100
    rotate_envs: bool = False  # False reproduces notebook 03 (restart at env 0 each cycle)

    # Replay augmentation: per-sequence random permutation of action and
    # observation ids (as in examples/02_train_offline.ipynb). Defeats
    # id-level map memorization, forcing in-context inference.
    augment: bool = False

    # Optimization
    gradient_steps: int = 20000
    gradient_steps_per_cycle: int = 1000
    learning_starts: int = 2000
    sequence_length: int = 512
    batch_size: int = 4
    lr: float = 1.0e-5

    # Exploration
    # "global": linear epsilon decay by total env step (notebook 03 behavior)
    # "per_env": linear epsilon decay by each env's own step count
    # "oracle": behavior policy mixes random -> Q* expert per env-local step
    #           (online analogue of notebook 01's collection schedule); the
    #           model is never queried during rollout.
    exploration_mode: str = "global"
    exploration_ends: int = 10000  # global mode: env step when decay finishes
    per_env_exploration_steps: int = 200  # per_env/oracle modes: local steps to decay over
    epsilon_floor: float = 0.0

    # Objective
    gamma_step: float = 1.0
    gamma_episode: float = 0.99  # applied to all four boundary discounts
    tau: float = 0.0005

    # Env reward shaping
    step_penalty: float = 0.0

    # Eval
    eval_maps: int = 16
    eval_steps: int = 250
    eval_fracs: tuple[float, ...] = (0.5, 1.0)
    final_eval_temperatures: tuple[float, ...] = (0.0, 0.2)


# ---------------------------------------------------------------------------
# Environment / model construction
# ---------------------------------------------------------------------------


def make_frozenlake(
    name: str,
    map_seed: int,
    reset_seed: int,
    step_penalty: float,
    with_q_star: bool = False,
):
    config = EnvConfig(
        id="Procedural-FrozenLake-v1",
        name=name,
        reset_seed=reset_seed,
        episodes_per_task=0,
        kwargs={
            "min_width": 4,
            "max_width": 8,
            "min_height": 4,
            "max_height": 8,
            "max_episode_steps": 50,
            "map_seed": map_seed,
            "step_penalty": step_penalty,
            # q_star_step_penalty defaults to step_penalty (or -1e-6 when zero),
            # so emitted Q* is always a non-degenerate shortest-path expert.
            "emit_q_star": with_q_star,
        },
    )
    return make_env(config)


def build_model(cfg: Config, device: torch.device) -> Model:
    backbone_kwargs = {} if cfg.backbone_layers is None else {"num_layers": cfg.backbone_layers}
    backbone = Qwen3Backbone(pretrained="Qwen/Qwen3-0.6B", **backbone_kwargs)
    encoder = StepEmbedder(
        hidden_dim=backbone.hidden_dim,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": MAX_ACTIONS, "std": 0.02, "tokens": 1},
            {"field": "observation", "type": "discrete", "vocab_size": MAX_OBS_DISCRETE, "std": 0.02, "tokens": 1},
            {"field": "reward", "type": "rff", "std": 0.02, "in_min": 0.01, "in_max": 100.0, "tokens": 1},
            {"field": "done", "type": "discrete", "vocab_size": 5, "std": 0.02, "tokens": 1},
        ],
        modality_fusion="sum",
        include_type_token=False,
    )
    head = DiscreteActionValueHead(
        in_features=backbone.hidden_dim,
        out_features=MAX_ACTIONS,
        hidden_dim=backbone.hidden_dim,
        num_layers=1,
        scale=0.1,
    )
    return Model(encoder=encoder, backbone=backbone, heads=head).to(device)


# ---------------------------------------------------------------------------
# Exploration schedules
# ---------------------------------------------------------------------------


def epsilon_global(cfg: Config, env_step: int) -> float:
    frac = min(env_step / cfg.exploration_ends, 1.0)
    return max(cfg.epsilon_floor, 1.0 - frac)


def epsilon_per_env(cfg: Config, env_local_step: int) -> float:
    frac = min(env_local_step / cfg.per_env_exploration_steps, 1.0)
    return max(cfg.epsilon_floor, 1.0 - frac)


# ---------------------------------------------------------------------------
# Rollout (adapted from examples/03_train_online.ipynb, with optional rotation)
# ---------------------------------------------------------------------------


class RolloutState:
    def __init__(self) -> None:
        self.cursor = 0  # next env index to visit (used when rotate_envs=True)
        self.env_steps = 0


def rollout(
    model: Model,
    cfg: Config,
    envs: list,
    stores: list[Datastore],
    contexts: list[deque],
    state: RolloutState,
) -> dict:
    for env in envs:
        env.metrics.clear()
    model.eval()

    budget = cfg.env_steps_per_cycle
    collected = 0
    visited: set[int] = set()
    n = len(envs)
    order = (
        [(state.cursor + k) % n for k in range(n)]
        if cfg.rotate_envs
        else list(range(n))
    )

    for idx in order:
        if collected >= budget:
            break
        env, store, context = envs[idx], stores[idx], contexts[idx]
        visited.add(idx)

        kv_cache = None
        cache_count = 0
        ctx_count = 0
        chunk = min(cfg.steps_per_env, budget - collected)

        for _ in range(chunk):
            if cfg.exploration_mode in ("per_env", "oracle"):
                epsilon = epsilon_per_env(cfg, len(store))
            else:
                epsilon = epsilon_global(cfg, state.env_steps)

            if not context or torch.rand(1).item() < epsilon:
                inp = env.sample_random_input()
            elif cfg.exploration_mode == "oracle":
                q_star = context[-1]["info_q_star"]
                inp = {"action": np.int64(np.asarray(q_star).argmax())}
            else:
                ctx_list = list(context)
                with torch.no_grad():
                    if kv_cache is None:
                        preds, _, kv_cache = model([ctx_list], use_cache=True)
                    else:
                        uncached = ctx_count - cache_count
                        preds, _, kv_cache = model(
                            [ctx_list[-uncached:]], cache=kv_cache, use_cache=True
                        )
                    cache_count = ctx_count
                action = model.get_action(preds, temperature=0.0, num_actions=MAX_ACTIONS)
                inp = {"action": action.squeeze().cpu().numpy()}

            out = env.step(inp)
            row = {**inp, **out}
            info = row.pop("info", None) or {}
            if "q_star" in info:
                row["info_q_star"] = info["q_star"]
            store.append(row)
            context.append(row)
            ctx_count += 1
            state.env_steps += 1
            collected += 1

        if cfg.rotate_envs:
            state.cursor = (idx + 1) % n

    rewards: list[float] = []
    lengths: list[float] = []
    for env in envs:
        rewards.extend(env.metrics.episode_cum_rewards)
        lengths.extend(env.metrics.episode_lengths)
    stats = {
        "env_steps": state.env_steps,
        "episodes": len(rewards),
        "mean_reward": float(torch.tensor(rewards).mean()) if rewards else None,
        "mean_length": float(torch.tensor(lengths).float().mean()) if lengths else None,
        "envs_visited": sorted(visited),
    }
    return stats


# ---------------------------------------------------------------------------
# Held-out / fresh-context evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: Model,
    cfg: Config,
    *,
    holdout: bool,
    temperature: float = 0.0,
    policy: str = "model",
    train_map_ids: list[int] | None = None,
) -> dict:
    """Greedy rollout from an empty context on eval maps; return metrics.

    holdout=True uses unseen maps (map_seed + HOLDOUT_MAP_OFFSET); False uses
    the first cfg.eval_maps *actually trained* maps with a fresh reset stream.
    policy="random" gives a model-free reference.
    """
    model.eval()
    if holdout:
        map_ids = [HOLDOUT_MAP_OFFSET + j for j in range(cfg.eval_maps)]
    else:
        pool = train_map_ids if train_map_ids else list(range(cfg.eval_maps))
        map_ids = pool[: cfg.eval_maps]

    per_env = []
    for j, map_id in enumerate(map_ids):
        env = make_frozenlake(
            name=f"eval_{map_id}",
            map_seed=map_id,
            reset_seed=EVAL_RESET_OFFSET + j,
            step_penalty=cfg.step_penalty,
        )
        env.metrics.clear()

        cache = None
        inp = None
        out = None
        prev_obs = None
        bumps = 0
        move_steps = 0
        obs_window: deque = deque(maxlen=10)
        stuck_steps = 0
        episode_rewards_in_order: list[float] = []

        for _t in range(cfg.eval_steps):
            if out is None or policy == "random":
                inp = env.sample_random_input()
            else:
                with torch.no_grad():
                    row = {**inp, **out}
                    row.pop("info", None)
                    pred, _, cache = model([[row]], cache=cache, use_cache=True)
                    action = model.get_action(
                        pred, temperature=temperature, num_actions=MAX_ACTIONS
                    )
                inp = {"action": action.squeeze().cpu().numpy()}
            out = env.step(inp)

            done = int(out["done"])
            obs = int(out["observation"])
            t_in_ep = int(out["time"])
            if done == 0 and t_in_ep > 0 and prev_obs is not None:
                move_steps += 1
                if obs == prev_obs:
                    bumps += 1
            prev_obs = obs if done == 0 else None
            obs_window.append((obs, done))
            if len(obs_window) == obs_window.maxlen and len(set(obs_window)) == 1:
                stuck_steps += 1

        episode_rewards_in_order = list(env.metrics.episode_cum_rewards)
        env.close()

        # Goal reward is 1.0; failed episodes accumulate at most 50 * step_penalty
        # (<= 0), so 0.25 cleanly separates success even with penalties enabled.
        successes = [1.0 if r > 0.25 else 0.0 for r in episode_rewards_in_order]
        n_ep = len(successes)
        half = n_ep // 2
        per_env.append(
            {
                "map_id": map_id,
                "episodes": n_ep,
                "success_rate": sum(successes) / n_ep if n_ep else None,
                "first_episode_success": successes[0] if n_ep else None,
                "first_half_success": sum(successes[:half]) / half if half else None,
                "second_half_success": sum(successes[half:]) / (n_ep - half) if n_ep - half else None,
                "mean_length": (
                    sum(env.metrics.episode_lengths) / n_ep if n_ep else None
                ),
                "bump_frac": bumps / move_steps if move_steps else None,
                "stuck_frac": stuck_steps / cfg.eval_steps,
            }
        )

    def agg(key: str) -> float | None:
        vals = [e[key] for e in per_env if e[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "holdout": holdout,
        "policy": policy,
        "temperature": temperature,
        "episodes_total": sum(e["episodes"] for e in per_env),
        "success_rate": agg("success_rate"),
        "first_half_success": agg("first_half_success"),
        "second_half_success": agg("second_half_success"),
        "mean_episode_length": agg("mean_length"),
        "bump_frac": agg("bump_frac"),
        "stuck_frac": agg("stuck_frac"),
        "per_env": per_env,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_burst(model, optimizer, objective, loader, burst_steps: int) -> dict:
    model.train()
    loader.refresh()
    losses = []
    q_means = []
    for _ in range(burst_steps):
        batch = loader.next_batch()
        predictions, objective_data, _ = model(batch)
        loss, metrics = objective(objective_data, predictions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.polyak_update(action_value_tau=objective.tau)
        losses.append(loss.item())
        q_means.append(metrics["q_values_mean"])
    return {
        "loss_mean": sum(losses) / len(losses),
        "loss_last": losses[-1],
        "q_mean": sum(q_means) / len(q_means),
    }


def run(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.name}.json"

    result: dict = {
        "config": dataclasses.asdict(cfg),
        "device": str(device),
        "cycles": [],
        "evals": [],
        "timing": {},
    }

    def save() -> None:
        out_path.write_text(json.dumps(result, indent=2))

    t0 = time.time()
    model = build_model(cfg, device)
    print(f"[{cfg.name}] model built in {time.time() - t0:.1f}s on {device}", flush=True)

    envs = [
        make_frozenlake(
            name=f"train_{i}",
            map_seed=i,
            reset_seed=i,
            step_penalty=cfg.step_penalty,
            with_q_star=cfg.exploration_mode == "oracle",
        )
        for i in range(cfg.num_envs)
    ]
    stores = [Datastore(name=env.name) for env in envs]
    contexts = [deque(maxlen=cfg.sequence_length) for _ in envs]

    augmenter = None
    if cfg.augment:
        augmenter = Augmenter(
            modalities=[
                {"field": "action", "type": "discrete", "vocab_size": MAX_ACTIONS, "permute": True},
                {"field": "observation", "type": "discrete", "vocab_size": MAX_OBS_DISCRETE, "permute": True},
            ],
            keep_fields=["action", "observation", "reward", "done"],
        )
    loader = DataLoader(
        stores,
        sequence_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        weight_mode="per_step",
        pack=True,
        num_workers=0,
        augmenter=augmenter,
    )
    objective = DqnObjective(
        gamma_step=cfg.gamma_step,
        gamma_episode_terminal=cfg.gamma_episode,
        gamma_episode_truncated=cfg.gamma_episode,
        gamma_task_terminal=cfg.gamma_episode,
        gamma_task_truncated=cfg.gamma_episode,
        tau=cfg.tau,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8
    )

    # Random-policy reference on the held-out eval maps (model-free).
    rand_eval = evaluate(model, cfg, holdout=True, policy="random")
    rand_eval["grad_steps"] = 0
    result["evals"].append(rand_eval)
    print(
        f"[{cfg.name}] random policy holdout: success={rand_eval['success_rate']}",
        flush=True,
    )
    save()

    state = RolloutState()
    grad_steps = 0
    eval_points = sorted({max(1, int(cfg.gradient_steps * f)) for f in cfg.eval_fracs})
    visited_envs: set[int] = set()

    rollout_time = 0.0
    train_time = 0.0
    eval_time = 0.0

    while grad_steps < cfg.gradient_steps:
        t = time.time()
        stats = rollout(model, cfg, envs, stores, contexts, state)
        rollout_time += time.time() - t
        visited_envs.update(stats.pop("envs_visited"))
        stats["distinct_envs_visited"] = len(visited_envs)
        stats["grad_steps"] = grad_steps
        if cfg.exploration_mode == "global":
            stats["epsilon"] = epsilon_global(cfg, state.env_steps)
        result["cycles"].append(stats)
        print(
            f"[{cfg.name}] env_step={state.env_steps} grad_step={grad_steps}"
            f" | envs_seen={len(visited_envs)}"
            f" | episodes={stats['episodes']} mean_reward={stats['mean_reward']}",
            flush=True,
        )

        if state.env_steps >= cfg.learning_starts:
            burst = min(cfg.gradient_steps_per_cycle, cfg.gradient_steps - grad_steps)
            t = time.time()
            tstats = train_burst(model, optimizer, objective, loader, burst)
            train_time += time.time() - t
            grad_steps += burst
            tstats["grad_steps"] = grad_steps
            result["cycles"][-1]["train"] = tstats
            print(
                f"[{cfg.name}] grad_step={grad_steps}"
                f" | loss={tstats['loss_mean']:.4f} q={tstats['q_mean']:.3f}",
                flush=True,
            )

            while eval_points and grad_steps >= eval_points[0]:
                point = eval_points.pop(0)
                final = not eval_points
                temps = cfg.final_eval_temperatures if final else (0.0,)
                t = time.time()
                for temp in temps:
                    ev = evaluate(model, cfg, holdout=True, temperature=temp)
                    ev["grad_steps"] = grad_steps
                    result["evals"].append(ev)
                    print(
                        f"[{cfg.name}] EVAL holdout @ {grad_steps} temp={temp}:"
                        f" success={ev['success_rate']}"
                        f" bump={ev['bump_frac']} stuck={ev['stuck_frac']}",
                        flush=True,
                    )
                ev = evaluate(
                    model,
                    cfg,
                    holdout=False,
                    train_map_ids=sorted(visited_envs),
                )
                ev["grad_steps"] = grad_steps
                result["evals"].append(ev)
                print(
                    f"[{cfg.name}] EVAL train-maps @ {grad_steps}:"
                    f" success={ev['success_rate']}"
                    f" bump={ev['bump_frac']} stuck={ev['stuck_frac']}",
                    flush=True,
                )
                eval_time += time.time() - t
        save()

    loader.close()
    for env in envs:
        env.close()

    result["timing"] = {
        "total_s": time.time() - t0,
        "rollout_s": rollout_time,
        "train_s": train_time,
        "eval_s": eval_time,
    }
    save()
    print(f"[{cfg.name}] done in {result['timing']['total_s']:.0f}s -> {out_path}", flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Config):
        arg = "--" + field.name.replace("_", "-")
        if field.type == "bool":
            parser.add_argument(arg, type=lambda s: s.lower() in ("1", "true", "yes"), default=field.default)
        elif field.type == "int | None":
            parser.add_argument(arg, type=int, default=field.default)
        elif field.type == "tuple[float, ...]":
            parser.add_argument(arg, type=float, nargs="+", default=list(field.default))
        elif field.type == "float":
            parser.add_argument(arg, type=float, default=field.default)
        elif field.type == "int":
            parser.add_argument(arg, type=int, default=field.default)
        else:
            parser.add_argument(arg, type=str, default=field.default)
    ns = parser.parse_args()
    kwargs = {f.name: getattr(ns, f.name) for f in dataclasses.fields(Config)}
    for key in ("eval_fracs", "final_eval_temperatures"):
        kwargs[key] = tuple(kwargs[key])
    return Config(**kwargs)


if __name__ == "__main__":
    run(parse_args())
