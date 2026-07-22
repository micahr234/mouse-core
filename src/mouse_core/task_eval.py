"""Held-out task evaluation for live ``mouse-gym`` environments.

Training notebooks should keep **train** and **eval** environment groups
separate (different ``reset_seed`` / ``map_seed`` streams). Call
:func:`run_task_eval` periodically during training with ``model.eval()`` and
greedy actions (``temperature=0``).

Requires the ``examples`` extra (``mouse-gym``, ``procedural-frozenlake``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mouse_core.models.base import Model
from mouse_core.models.kv_policy import (
    cache_needs_rebuild,
    rebuild_starts,
    resolve_cache_bounds,
)

# Eval seeds sit far above typical train seeds (0..NUM_ENVS) so maps never overlap.
DEFAULT_EVAL_SEED_OFFSET = 1_000_000


def make_procedural_frozenlake_group(
    *,
    num_envs: int,
    episodes_per_task: int,
    max_episode_steps: int = 30,
    seed_offset: int = 0,
    width: int = 8,
    height: int = 8,
    slippery_success_rate: float = 1.0,
    permute_obs: bool = True,
    permute_actions: bool = True,
    render: bool = False,
    name_prefix: str = "eval_frozenlake",
) -> Any:
    """Build a ``GroupEnv`` of Procedural-FrozenLake streams for train or eval.

    Use a distinct ``seed_offset`` for eval (default helpers use
    :data:`DEFAULT_EVAL_SEED_OFFSET`) so evaluation maps are not the training set.
    """
    import procedural_frozenlake  # noqa: F401 — registers the Gym ID
    from mouse_gym import EnvConfig, make_group_env

    render_mode = "rgb_array" if render else None
    configs = []
    for i in range(num_envs):
        seed = seed_offset + i
        kwargs: dict[str, Any] = {
            "width": width,
            "height": height,
            "max_episode_steps": max_episode_steps,
            "map_seed": seed,
            "slippery_success_rate": slippery_success_rate,
            "permute_obs": permute_obs,
            "permute_actions": permute_actions,
        }
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        configs.append(
            EnvConfig(
                id="Procedural-FrozenLake-v1",
                name=f"{name_prefix}_{i}",
                reset_seed=seed,
                episodes_per_task=episodes_per_task,
                task_reset_options={"regenerate_map": True},
                kwargs=kwargs,
            )
        )
    return make_group_env(configs)


def run_task_eval(
    model: Model,
    env: Any,
    *,
    tasks_per_env: int = 1,
    max_cache: int = 512,
    start_cache: int | None = None,
    temperature: float = 0.0,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Greedy task-budget eval on a ``GroupEnv`` (does not mutate train replay).

    Each env runs ``tasks_per_env`` tasks (fresh map each). Scores are points
    per task from ``env.metrics.task_cum_rewards`` (typically 0..episodes_per_task).

    Returns
    -------
    dict
        ``mean_task_score``, ``task_scores`` (flat list), ``scores_per_env``,
        ``n_tasks``, ``steps``, ``episodes_per_task`` (from configs when available).
    """
    was_training = model.training
    model.eval()

    max_cache, start_cache = resolve_cache_bounds(max_cache, start_cache)
    n = len(env.names)
    kv_cache = None
    contexts: list[list[dict]] = [[] for _ in range(n)]
    cached_starts = np.zeros(n, dtype=np.int64)
    cached_ends = np.zeros(n, dtype=np.int64)
    context_start = np.zeros(n, dtype=np.int64)
    eval_task_scores: list[list[float]] = [[] for _ in range(n)]
    inputs = None
    outputs = None
    env.metrics.clear()
    num_actions = env.action_space.spaces[0].n
    steps_done = 0

    try:
        while min(len(scores) for scores in eval_task_scores) < tasks_per_env:
            if outputs is None:
                inputs = env.sample_random_input()
            else:
                task_ended = False
                for i, (inp, out) in enumerate(zip(inputs, outputs)):
                    row = {**inp, **out}
                    row.pop("info", None)
                    contexts[i].append(row)
                    if int(row["done"]) >= 3:
                        contexts[i] = []
                        context_start[i] = 0
                        task_ended = True
                ends = np.array([len(c) for c in contexts], dtype=np.int64)
                need_rebuild = task_ended or cache_needs_rebuild(
                    has_cache=kv_cache is not None,
                    cached_starts=cached_starts,
                    cached_ends=cached_ends,
                    ends=ends,
                    context_start=context_start,
                    max_cache=max_cache,
                    batch_complete=True,
                )
                with torch.no_grad():
                    if need_rebuild:
                        starts = rebuild_starts(
                            ends,
                            context_start=context_start,
                            start_cache=start_cache,
                            max_cache=max_cache,
                        )
                        batch = [
                            contexts[i][int(starts[i]) : int(ends[i])] for i in range(n)
                        ]
                        predictions, _, kv_cache = model(batch, use_cache=True)
                        cached_starts = starts
                        cached_ends = ends.copy()
                    else:
                        batch = [
                            contexts[i][int(cached_ends[i]) : int(ends[i])]
                            for i in range(n)
                        ]
                        predictions, _, kv_cache = model(
                            batch, cache=kv_cache, use_cache=True
                        )
                        cached_ends = ends.copy()
                    actions = model.get_action(
                        predictions, temperature=temperature, num_actions=num_actions
                    )
                random_inputs = env.sample_random_input()
                inputs = [
                    {"action": action} if contexts[i] else random_inputs[i]
                    for i, action in enumerate(actions.cpu().numpy())
                ]
            outputs = env.step(inputs)
            for i, out in enumerate(outputs):
                if int(out["done"]) >= 3:
                    eval_task_scores[i].append(float(env.metrics.task_cum_rewards[i][-1]))
            steps_done += 1
            if progress_every > 0 and steps_done % progress_every == 0:
                flat = [r for env_tasks in eval_task_scores for r in env_tasks]
                mean = sum(flat) / len(flat) if flat else float("nan")
                print(
                    f"  eval step {steps_done} | {len(flat)} tasks done"
                    f" | mean task score {mean:.2f}"
                )
    finally:
        del kv_cache
        if was_training:
            model.train()

    # Cap to tasks_per_env in case an env finished an extra boundary.
    scores_per_env = [scores[:tasks_per_env] for scores in eval_task_scores]
    flat_scores = [s for scores in scores_per_env for s in scores]
    mean_task_score = sum(flat_scores) / len(flat_scores) if flat_scores else float("nan")
    return {
        "mean_task_score": mean_task_score,
        "task_scores": flat_scores,
        "scores_per_env": scores_per_env,
        "n_tasks": len(flat_scores),
        "steps": steps_done,
    }
