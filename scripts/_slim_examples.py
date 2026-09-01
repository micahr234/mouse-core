"""One-shot helper: slim example notebooks (train = train, eval = 09)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "examples"

OFFLINE = {
    "02_train_offline_dqn.ipynb",
    "04_train_offline_layerwise_dqn.ipynb",
    "05_train_offline_sv.ipynb",
    "06_train_offline_text_dqn.ipynb",
    "10_train_offline_sp.ipynb",
}
ONLINE = {
    "03_train_online_dqn.ipynb",
    "07_train_online_ppo.ipynb",
    "08_train_online_grpo.ipynb",
}
TRAIN = OFFLINE | ONLINE


def src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    if text and not text.endswith("\n"):
        text += "\n"
    cell["source"] = [line for line in text.splitlines(keepends=True)]
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def clear_outputs(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)


def drop_cells(nb: dict, pred) -> None:
    nb["cells"] = [c for c in nb["cells"] if not pred(c)]


KV_IMPORT = """from mouse_core.models.kv_policy import (
    cache_needs_rebuild,
    rebuild_starts,
    resolve_cache_bounds,
)

"""

GYM_OFFLINE = "import procedural_frozenlake  # noqa: F401 — registers Procedural-FrozenLake-v1\nfrom mouse_gym import EnvConfig, make_group_env\n"


INTROS = {
    "02_train_offline_dqn.ipynb": """# 02 - Train a DQN Model Offline

This notebook shows the offline training workflow in Mouse Core:

1. Load previously collected `Datastore` streams from the Hub.
2. Build a `DataLoader` that samples fixed-length sequences from those streams.
3. Assemble a `Model` from an embedder, a backbone, and an action-value head.
4. Train with `DqnObjective` and save with `push_model_to_hub`.

The dataset comes from the collection notebook, but the same Mouse Core pieces apply to any sequential environment data stored as step dictionaries.

This is a short usage example, not a full experiment. Evaluate a saved checkpoint in `09_inference.ipynb`.
""",
    "03_train_online_dqn.ipynb": """# 03 - Train a DQN Model Online

This notebook shows the online training workflow in Mouse Core. Instead of loading a fixed dataset, it keeps live environments in the loop:

1. Build a train `GroupEnv`.
2. For each of `NUM_CYCLES` cycles: collect `ROLLOUT_STEPS` env steps, then run `TRAIN_STEPS` optimizer updates.
3. Append transitions into in-memory `Datastore` replay streams.
4. Sample those streams with `DataLoader`.
5. Train with `DqnObjective`.

Online and offline training use the same row format, model interface, datastores, dataloader, and objective. Only the source of rows changes.

This is a short usage example, not a full experiment. Evaluate a saved checkpoint in `09_inference.ipynb`.
""",
    "04_train_offline_layerwise_dqn.ipynb": """# 04 - Train a Layerwise DQN Model Offline

This notebook follows the same offline workflow as `02_train_offline_dqn.ipynb`, but swaps the standard DQN head and objective for their layerwise variants:

1. Load previously collected `Datastore` streams from the Hub.
2. Build a `DataLoader` that samples fixed-length sequences from those streams.
3. Assemble a `Model` with `LayerwiseDiscreteActionValueHead` — one value head per backbone layer.
4. Train with `LayerwiseDqnObjective` and save with `push_model_to_hub`.

Layerwise DQN is deep supervision: every transformer block gets its own Q head on the same Bellman target. With undiscounted within-task gammas (`1.0`), there is no finite-horizon ladder to interpolate across layers — each head trains on the same objective.

This is a short usage example, not a full experiment. Evaluate a saved checkpoint in `09_inference.ipynb`.
""",
    "05_train_offline_sv.ipynb": """# 05 - Train a Supervised-Value Model Offline

This notebook is the simple offline check: can the model fit expert Q*?

It uses the same data, embedder, backbone, and action-value head as
`02_train_offline_dqn.ipynb`, but replaces `DqnObjective` with `SvObjective`.
The head is trained to match `info_q_star` at every step (MSE). There is no
Bellman backup and no target network.

1. Load previously collected `Datastore` streams from the Hub (must include `info_q_star`).
2. Build a `DataLoader` that samples fixed-length sequences from those streams.
3. Assemble a `Model` from an embedder, a backbone, and an action-value head.
4. Train with `SvObjective` and save with `push_model_to_hub`.

`Augmenter` remaps `action` ids and, via `input_vector_field` /
`output_vector_field` on `info_q_star`, reorders the Q vector with the **same**
permutation so it stays aligned with the remapped ids.

This is a short usage example, not a full experiment. Evaluate a saved checkpoint in `09_inference.ipynb`.
""",
    "06_train_offline_text_dqn.ipynb": """# 06 - Offline DQN with TextEmbedder

Same FrozenLake offline DQN loop as `02_train_offline_dqn.ipynb`, but each step is rendered with `TextEmbedder`: `action` is `type: "token"` (integer id → one `embed_tokens` row), while observation/reward/episode_done/task_done are `type: "text"` (value → string via `format` → tokenizer). Whole-step template `"<action={action},{observation},{reward},{episode_done},{task_done}>"`. Reward `0.0` and done codes `0` use `skip` so those text fragments are omitted (commas stay). Heads read the last token of each step (no learnable scratch tokens).

`type: "image"` is also supported when `pretrained` is a vision-language checkpoint (not used in this notebook).

This is a short usage example, not a full experiment. Evaluate a saved checkpoint in `09_inference.ipynb`.
""",
    "07_train_online_ppo.ipynb": """# 07 - Train a PPO Model Online

This notebook shows on-policy PPO training in Mouse Core. It mirrors the online DQN loop in `03_train_online_dqn.ipynb`, but swaps the action-value head and TD loss for an actor-critic pair:

1. Build a train `GroupEnv`.
2. For each of `NUM_CYCLES` cycles: collect `ROLLOUT_STEPS` env steps, then run `PPO_EPOCHS` × `TRAIN_STEPS` optimizer updates.
3. Store each transition with its behavior `old_log_prob` in an on-policy `Datastore`.
4. Sample those rows with `DataLoader` and train with `PpoObjective` (GAE + clipped surrogate + value + entropy).
5. Clear the on-policy stores after each cycle.

The shared Mouse Core pieces stay the same: row dicts, `Datastore` / `DataLoader`, `Model.forward`, and a plain objective callable. Only the heads, action sampling, and objective change.

This is a short usage example, not a full experiment. Evaluate a saved checkpoint in `09_inference.ipynb`.
""",
    "08_train_online_grpo.ipynb": """# 08 - Train Online with GRPO (branched rollouts)

This notebook puts **Group Relative Policy Optimization** on paper in Mouse Core terms.

GRPO is PPO's clipped policy update **without a critic**. Instead of GAE from a value head, advantages come from comparing **G stochastic completions** that share the same start:

1. Grow a **trunk** trajectory (env + context of length `L`).
2. At chosen context lengths, **fork**: `deepcopy(env)` × G and copy `context[:L]`.
3. From each fork, sample a different completion with the stochastic policy.
4. Score the G completions (here: sum of rewards on the branch suffix).
5. Z-score those scores inside the group → one advantage per branch.
6. Stamp advantages onto completion steps, train with `GrpoObjective`.

```text
trunk context length L
        │
        ├─ branch 0  ~~~stochastic~~~→  return r0
        ├─ branch 1  ~~~stochastic~~~→  return r1
        ├─ ...
        └─ branch G-1 ~~~~~~~~~~~~~~~→  return r_{G-1}

        A_i = (r_i - mean(r)) / std(r)     # group_relative_advantages
```

Forks happen at **many** `L` values (not only task start), so the policy is trained under short and long in-context histories.

Compared to `07_train_online_ppo.ipynb`: policy head only, no value head; collection is branched rather than a single on-policy stream.

This is a short usage example, not a full experiment. Evaluate a saved checkpoint in `09_inference.ipynb`.
""",
    "10_train_offline_sp.ipynb": """# 10 - Train a Supervised-Policy Model Offline

This notebook is the ranking check: can the model copy expert `argmax(Q*)`?

It uses the same data, embedder, and backbone as `05_train_offline_sv.ipynb`, but swaps the value head and `SvObjective` for `DiscreteActionHead` and `SpObjective`. The head is trained with hard cross-entropy onto a random argmax of `info_q_star` (ties broken uniformly) while the episode is running (`episode_done == 0`). There is no Bellman backup, no target network, and no magnitude regression.

1. Load previously collected `Datastore` streams from the Hub (must include `info_q_star`).
2. Build a `DataLoader` that samples fixed-length sequences from those streams.
3. Assemble a `Model` from an embedder, a backbone, and an action head.
4. Train with `SpObjective` and save with `push_model_to_hub`.

`Augmenter` remaps `action` ids and, via `input_vector_field` /
`output_vector_field` on `info_q_star`, reorders the Q vector with the **same**
permutation so it stays aligned with the remapped ids.

This is a short usage example, not a full experiment. Evaluate a saved checkpoint in `09_inference.ipynb`.
""",
}


def rewrite_offline_imports(text: str) -> str:
    text = text.replace("import torch\nimport numpy as np\n\n", "import torch\n\n")
    text = text.replace(GYM_OFFLINE, "")
    text = text.replace(KV_IMPORT, "")
    text = text.replace("    compose,\n    pack_token_batch,\n    load_stores_from_hub,\n", "    compose,\n    load_stores_from_hub,\n")
    text = re.sub(
        r"\nNUM_CYCLES = 20\s+# outer train/eval cycles\n"
        r"TRAIN_STEPS = 1000\s+# optimizer updates per cycle \(passed to run_train\)\n"
        r"EVAL_STEPS = 512\s+# lockstep env steps per cycle \(passed to run_eval\)\n"
        r"EVAL_NUM_ENVS = \d+\s+# separate eval env streams \(not used for replay\)\n"
        r"MAX_EPISODES_PER_TASK = 20\s+# max episodes per task\n"
        r"MAX_STEPS_PER_EPISODE = 30\s+# max steps per episode\n"
        r"EVAL_SEED_OFFSET = 1_000_000\s+# held-out env seed stream \(far from train\)\n",
        "\nNUM_CYCLES = 2                               # outer train cycles (print cadence)\n"
        "TRAIN_STEPS = 50                             # optimizer updates per cycle (passed to run_train)\n",
        text,
    )
    return text


def rewrite_online_constants(text: str) -> str:
    text = text.replace(
        "NUM_ENVS = 30                                 # number of environment streams in the GroupEnv",
        "NUM_ENVS = 4                                  # number of environment streams in the GroupEnv",
    )
    text = re.sub(
        r"NUM_CYCLES = 100\s+# outer rollout/train/eval cycles\n"
        r"ROLLOUT_STEPS = 500\s+# env steps per cycle \(passed to run_rollout\)\n"
        r"TRAIN_STEPS = (\d+)\s+# optimizer updates per cycle \(passed to run_train\)\n"
        r"EVAL_STEPS = 512\s+# lockstep env steps per cycle \(passed to run_eval\)\n\n"
        r"EVAL_NUM_ENVS = 4\n"
        r"EVAL_SEED_OFFSET = 1_000_000\s+# held-out env seed stream \(far from train\)\n",
        "NUM_CYCLES = 2                               # outer rollout/train cycles\n"
        "ROLLOUT_STEPS = 64                           # env steps per cycle (passed to run_rollout)\n"
        r"TRAIN_STEPS = \1                             # optimizer updates per cycle (passed to run_train)\n",
        text,
    )
    # PPO has extra PPO_EPOCHS line
    text = re.sub(
        r"NUM_CYCLES = 100\s+# outer rollout/train/eval cycles\n"
        r"ROLLOUT_STEPS = 500\s+# env steps per cycle \(passed to run_rollout\)\n"
        r"TRAIN_STEPS = 50\s+# optimizer updates per PPO epoch \(passed to run_train\)\n"
        r"PPO_EPOCHS = 4\s+# run_train calls per cycle \(4 \* 50 = 200 updates\)\n"
        r"EVAL_STEPS = 512\s+# lockstep env steps per cycle \(passed to run_eval\)\n\n"
        r"EVAL_NUM_ENVS = 4\n"
        r"EVAL_SEED_OFFSET = 1_000_000\s+# held-out env seed stream \(far from train\)\n",
        "NUM_CYCLES = 2                               # outer rollout/train cycles\n"
        "ROLLOUT_STEPS = 64                           # env steps per cycle (passed to run_rollout)\n"
        "TRAIN_STEPS = 20                             # optimizer updates per PPO epoch (passed to run_train)\n"
        "PPO_EPOCHS = 2                               # run_train calls per cycle\n",
        text,
    )
    text = text.replace(
        "LEARNING_STARTS = 15_000                      # replay rows collected before the first optimizer update",
        "LEARNING_STARTS = 64                          # replay rows collected before the first optimizer update",
    )
    text = text.replace(
        "LEARNING_STARTS = 15_000                      # env steps collected before the first optimizer update",
        "LEARNING_STARTS = 64                          # env steps collected before the first optimizer update",
    )
    text = text.replace(
        "EXPLORATION_ENDS = 1_500_000                  # env-step horizon for epsilon decay",
        "EXPLORATION_ENDS = 1_000                      # env-step horizon for epsilon decay",
    )
    # GRPO
    text = re.sub(
        r"NUM_CYCLES = 40\s+# outer rollout/train/eval cycles\n"
        r"ROLLOUT_STEPS = 128\s+# trunk growth steps per cycle \(passed to run_rollout\)\n"
        r"TRAIN_STEPS = 25\s+# optimizer updates per GRPO epoch \(passed to run_train\)\n"
        r"GRPO_EPOCHS = 2\s+# run_train calls per cycle \(2 \* 25 = 50 updates\)\n"
        r"EVAL_STEPS = 512\s+# lockstep env steps per cycle \(passed to run_eval\)\n\n"
        r"EVAL_NUM_ENVS = 4\n"
        r"EVAL_SEED_OFFSET = 1_000_000\s+# held-out env seed stream \(far from train\)\n",
        "NUM_CYCLES = 2                               # outer rollout/train cycles\n"
        "ROLLOUT_STEPS = 64                           # trunk growth steps per cycle (passed to run_rollout)\n"
        "TRAIN_STEPS = 10                             # optimizer updates per GRPO epoch (passed to run_train)\n"
        "GRPO_EPOCHS = 2                               # run_train calls per cycle\n",
        text,
    )
    text = text.replace(
        "from mouse_gym import EnvConfig, make_env, make_group_env\n",
        "from mouse_gym import EnvConfig, make_env\n",
    )
    return text


def rewrite_pipeline_md(text: str) -> str:
    text = text.replace(
        "Compose two pipelines that share selector + tokenizer:\n"
        "`train_transform = compose(augmenter, selector, tokenizer)` and\n"
        "`eval_transform = compose(selector, tokenizer)`.\n"
        "`DataLoader(transform=train_transform)` maps each step and packs into a `TokenBatch`.\n"
        "Eval / decode uses `eval_transform` so the model sees raw values.\n",
        "Compose `train_transform = compose(augmenter, selector, tokenizer)`.\n"
        "`DataLoader(transform=train_transform)` maps each step and packs into a `TokenBatch`.\n"
        "Live inference in `09_inference.ipynb` uses the tokenizer without the augmenter so chosen actions match the env.\n",
    )
    # SV/SP extra sentence about eval_transform
    text = re.sub(
        r"Eval / decode uses `eval_transform` so the model sees raw values\. Eval rows stamp\n"
        r"`info_q_star` from `info\[\"q_star\"\]` so the tokenizer keep-list is present\.\n",
        "Live inference in `09_inference.ipynb` uses the tokenizer without the augmenter so chosen actions match the env.\n",
        text,
    )
    text = text.replace(
        "`eval_transform = compose(selector, tokenizer)`.\n"
        "Eval / decode uses `eval_transform` so the model sees raw values. Eval rows stamp\n"
        "`info_q_star` from `info[\"q_star\"]` so the tokenizer keep-list is present.\n",
        "Live inference in `09_inference.ipynb` uses the tokenizer without the augmenter so chosen actions match the env.\n",
    )
    return text


def strip_eval_transform_block(text: str) -> str:
    text = text.replace("train_transform = compose(augmenter, selector, tokenizer)\neval_transform = compose(selector, tokenizer)\n", "train_transform = compose(augmenter, selector, tokenizer)\n")
    text = re.sub(
        r"\n\ndef pack_rows\(rows: list\[list\[dict\]\]\):.*",
        "\n",
        text,
        flags=re.S,
    )
    return text


def rewrite_train_md(text: str) -> str:
    text = text.replace(
        "The offline loop is intentionally small. Each outer cycle runs `TRAIN_STEPS` optimizer updates via `run_train`, then scores the held-out env with `run_eval`. Mouse Core abstractions do most of the work:\n",
        "Each outer cycle runs `TRAIN_STEPS` optimizer updates via `run_train`. Mouse Core abstractions do most of the work:\n",
    )
    text = text.replace(
        "The offline loop is intentionally small. Each outer cycle runs `TRAIN_STEPS` optimizer updates via `run_train`, then scores the held-out env with `run_eval`.\n",
        "Each outer cycle runs `TRAIN_STEPS` optimizer updates via `run_train`.\n",
    )
    return text


def rewrite_run_md(text: str) -> str:
    replacements = [
        (
            "Each of `NUM_CYCLES` cycles calls `run_train(num_steps=TRAIN_STEPS)`, then `run_eval(num_steps=EVAL_STEPS)`.\n",
            "Each of `NUM_CYCLES` cycles calls `run_train(num_steps=TRAIN_STEPS)`. Score the checkpoint later in `09_inference.ipynb`.\n",
        ),
        (
            "The main loop runs `NUM_CYCLES` times. Each cycle calls `run_rollout(num_steps=ROLLOUT_STEPS)`, then `run_train(num_steps=TRAIN_STEPS)` once replay has at least `LEARNING_STARTS` rows, then `run_eval(num_steps=EVAL_STEPS)`.\n",
            "The main loop runs `NUM_CYCLES` times. Each cycle calls `run_rollout(num_steps=ROLLOUT_STEPS)`, then `run_train(num_steps=TRAIN_STEPS)` once replay has at least `LEARNING_STARTS` rows. Score the checkpoint later in `09_inference.ipynb`.\n",
        ),
        (
            "Each of `NUM_CYCLES` cycles calls `run_rollout(num_steps=ROLLOUT_STEPS)`, then `run_train(num_steps=TRAIN_STEPS)` for `PPO_EPOCHS` passes (once replay has `LEARNING_STARTS` rows), then clears the buffers and runs `run_eval(num_steps=EVAL_STEPS)`.\n",
            "Each of `NUM_CYCLES` cycles calls `run_rollout(num_steps=ROLLOUT_STEPS)`, then `run_train(num_steps=TRAIN_STEPS)` for `PPO_EPOCHS` passes (once replay has `LEARNING_STARTS` rows), then clears the buffers. Score the checkpoint later in `09_inference.ipynb`.\n",
        ),
        (
            "Each of `NUM_CYCLES` cycles collects forks over `ROLLOUT_STEPS` of trunk growth, runs `GRPO_EPOCHS` × `run_train(num_steps=TRAIN_STEPS)`, clears branch stores, then scores held-out envs with `run_eval(num_steps=EVAL_STEPS)`.\n",
            "Each of `NUM_CYCLES` cycles collects forks over `ROLLOUT_STEPS` of trunk growth, runs `GRPO_EPOCHS` × `run_train(num_steps=TRAIN_STEPS)`, then clears branch stores. Score the checkpoint later in `09_inference.ipynb`.\n",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def rewrite_run_code(text: str) -> str:
    # Remove eval call + print from train loops
    text = re.sub(
        r"\n    stats = run_eval\(num_steps=EVAL_STEPS, max_cache=SEQUENCE_LENGTH, model=model, env=eval_env, contexts=eval_contexts\)\n"
        r"    print\(f\"cycle=\{cycle\} eval\s+mean_task_score=\{stats\['mean_task_score'\]:.2f\}/\{MAX_EPISODES_PER_TASK\}  n_tasks=\{stats\['n_tasks'\}\"\)\n",
        "\n",
        text,
    )
    text = text.replace("loader.close()\neval_env.close()\n", "loader.close()\n")
    text = text.replace("env.close()\neval_env.close()\n", "env.close()\n")
    text = text.replace("for env in trunk_envs:\n    env.close()\neval_env.close()\n", "for env in trunk_envs:\n    env.close()\n")
    return text


def rewrite_env_md(text: str, name: str) -> str:
    if name in OFFLINE:
        return text
    if name == "03_train_online_dqn.ipynb":
        return (
            "## Build Environment\n"
            "\n"
            "Online training uses the same `EnvConfig` and `make_group_env` pattern as data collection. Build a **train** `GroupEnv` for rollouts.\n"
            "\n"
            "The group environment steps all configured instances together and returns one output dictionary per instance. Keep row fields consistent with the model and objective: this notebook consumes `action`, `observation`, `reward`, `episode_done`, and `task_done`; the DQN objective uses `episode_done` and `task_done` to decide whether bootstrapping should continue across a boundary.\n"
        )
    if name == "07_train_online_ppo.ipynb":
        return (
            "## Build Environment\n"
            "\n"
            "Online PPO uses the same `EnvConfig` / `make_group_env` setup as the other live-env notebooks. Build a **train** `GroupEnv` for on-policy rollouts.\n"
            "\n"
            "Keep row fields consistent with the model modalities and with `PpoObjective` (`action`, `observation`, `reward`, `episode_done`, `task_done`, plus rollout-only `old_log_prob`).\n"
        )
    if name == "08_train_online_grpo.ipynb":
        return (
            "## Build Environment\n"
            "\n"
            "Each trunk is a **single** `mouse_gym` env (not a big `GroupEnv`). Forking uses `copy.deepcopy(env)` so every branch starts from the same physical state as the trunk at length `L`, then diverges through stochastic actions.\n"
        )
    return text


def rewrite_rollout_mentions(text: str) -> str:
    text = text.replace("same grow-then-rebuild policy as `run_eval`", "same grow-then-rebuild policy as `09_inference.ipynb`")
    text = text.replace("policy as ``run_eval``", "policy as ``09_inference.ipynb``")
    return text


def slim_01(nb: dict) -> None:
    for cell in nb["cells"]:
        t = src(cell)
        if "NUM_ENVS = 30" in t:
            t = t.replace("NUM_ENVS = 30                         # number of environment streams in the GroupEnv", "NUM_ENVS = 4                          # number of environment streams in the GroupEnv")
            t = t.replace("STEPS_PER_ENV = 50_000                # rows collected for each datastore", "STEPS_PER_ENV = 256                    # rows collected for each datastore")
            set_src(cell, t)
        if cell.get("cell_type") == "markdown" and t.startswith("# 01"):
            if "short usage example" not in t:
                t = t.rstrip() + "\n\nThis is a short usage example, not a full experiment. Scale collection up in [mouse-experiment](https://github.com/micahr234/mouse-experiment).\n"
                set_src(cell, t)


def slim_09(nb: dict) -> None:
    cell = nb["cells"][0]
    set_src(
        cell,
        """# 09 - Evaluate a Trained Model

Run this **after** a training notebook. This is the evaluation notebook: load a saved Mouse Core checkpoint and score it on live environments.

The inference flow is:

1. Build evaluation environments with `EnvConfig` and `make_group_env`.
2. Load a checkpoint with `load_model`.
3. Step environments, convert model predictions to actions with `model.get_action`, and carry the cache returned by `model(..., use_cache=True)`.
4. Record scores and optional render frames.

The rendering code is specific to this example environment; the model loading and cached inference pattern is the reusable Mouse Core part.
""",
    )


def should_drop(cell: dict, name: str) -> bool:
    t = src(cell)
    heading = t.lstrip().split("\n", 1)[0] if t.strip() else ""
    if name in OFFLINE and heading == "## Build Environment":
        return True
    if t.startswith("eval_configs ="):
        return True
    if heading == "## Evaluation Phase":
        return True
    if "def run_eval(" in t and name != "09_inference.ipynb":
        return True
    return False


def transform(name: str, nb: dict) -> None:
    clear_outputs(nb)
    if name == "01_collect_dataset.ipynb":
        slim_01(nb)
        return
    if name == "09_inference.ipynb":
        slim_09(nb)
        return
    if name not in TRAIN:
        return

    drop_cells(nb, lambda c: should_drop(c, name))

    if name in INTROS:
        set_src(nb["cells"][0], INTROS[name])

    for cell in nb["cells"]:
        t = src(cell)
        orig = t
        heading = t.lstrip().split("\n", 1)[0] if t.strip() else ""

        if cell.get("cell_type") == "code" and ("DATASET_ID" in t or "MODEL_ID" in t) and "import torch" in t:
            if name in OFFLINE:
                t = rewrite_offline_imports(t)
            else:
                t = rewrite_online_constants(t)

        if heading == "## Data pipeline" or heading.startswith("## Data pipeline"):
            t = rewrite_pipeline_md(t)

        if "train_transform = compose" in t and "eval_transform" in t:
            t = strip_eval_transform_block(t)

        if heading == "## Build Environment":
            t = rewrite_env_md(t, name)

        if heading == "## Training Phase":
            t = rewrite_train_md(t)

        if heading in {"## Run", "## Run Online Training", "## Run Online PPO"}:
            t = rewrite_run_md(t)

        if "run_eval(" in t or "eval_env.close()" in t:
            t = rewrite_run_code(t)

        t = rewrite_rollout_mentions(t)

        if t != orig:
            set_src(cell, t)


def main() -> None:
    for path in sorted(ROOT.glob("*.ipynb")):
        nb = json.loads(path.read_text())
        transform(path.name, nb)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"updated {path.name} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
