"""Run notebook examples locally with real models, data, gradients, and rendering.

Example (the tokenizer source must already exist locally)::

    PYTHON_GIL=0 .venv/bin/python scripts/run_examples.py \
        --output-dir /tmp/mouse-examples --tokenizer-source /path/to/cached/Qwen3 \
        --threads 2 --seed 7 --examples all

This is an explicit short CPU profile, not a reproduction of full training runs.
It creates a random one-layer Qwen3 checkpoint (32 hidden units, two heads) and
copies the real cached tokenizer. Notebook model/tokenizer constructors, forwards,
backwards, optimizers, cached decoding, environments, and rendering run normally.
Only publication cells are skipped. Hub dataset reads use local Arrow datasets
written by notebooks 01 and 17; notebook 15 reloads notebook 02's trained model.
All run settings, cell output, validation results, and artifacts stay in output-dir.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CPU_PROFILE = {
    "MAX_STEPS_PER_EPISODE": 4,
    "REPEAT_COUNT": 10,
    "NUM_ENVS": 2,
    "STEPS_PER_ENV": 128,
    "NUM_CYCLES": 1,
    "TRAIN_STEPS": 2,
    "TOKEN_BUDGET": 1024,
    "EPISODES_PER_STREAM": 1,
    "MAX_CONTEXT": 64,
    "LEARNING_STARTS": 1,
    "EXPLORATION_ENDS": 1,
    "PPO_EPOCHS": 1,
    "BRANCHES_PER_ENV": 2,
    "PASSES_PER_COLLECTION": 1,
    "EVAL_STEPS": 64,
    "EPISODES_PER_ENV": 1,
}


class Settings(ast.NodeTransformer):
    """Replace named first-cell settings and explicitly select the CPU kernel."""

    def __init__(self, names: set[str], *, settings_cell: bool):
        self.names = names
        self.settings_cell = settings_cell

    def visit_Assign(self, node):
        if self.settings_cell and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in self.names:
                node.value = ast.Subscript(
                    value=ast.Name(id="_local_settings", ctx=ast.Load()),
                    slice=ast.Constant(value=target.id), ctx=ast.Load(),
                )
        return self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg == "train_kernel" and isinstance(node.value, ast.Constant):
            node.value = ast.Constant(value="reference")
        return self.generic_visit(node)


def prepare_checkpoint(*, source: Path, target: Path, seed: int):
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=4096,
        attention_dropout=0.0, tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    target.mkdir(parents=True, exist_ok=True)
    Qwen3Model(config).save_pretrained(target)
    tokenizer.save_pretrained(target)


def local_settings(*, notebook: Path, output: Path, seed: int) -> dict[str, Any]:
    import torch

    settings = {
        **CPU_PROFILE,
        "PRETRAINED": str(output / "tiny-qwen3"),
        "device": torch.device("cpu"),
        "SEED": seed,
        "CHECKPOINT_PATH": output / "models" / "21-cosine.pt",
    }
    if notebook.name.startswith("15_"):
        settings["MODEL_ID"] = str(output / "models" / "02_train_offline_dqn")
    if notebook.name.startswith(("17_", "18_")):
        # Plain CartPole keeps its real 500-step limit and exposes only position/angle.
        settings.pop("MAX_STEPS_PER_EPISODE")
        settings.pop("REPEAT_COUNT")
        settings.update(STEPS_PER_ENV=256, TOKEN_BUDGET=4096)
    if notebook.name.startswith(("19_", "20_")):
        # Native 200-step Pendulum / 50-step Reacher episodes and their real tiny models.
        settings = {"NUM_CYCLES": 1, "TRAIN_STEPS": 2, "SEED": seed,
                    "EPISODES_PER_CYCLE": 1, "device": torch.device("cpu")}
        if notebook.name.startswith("19_"):
            settings.update(SAVE_ARTIFACTS=True, OUTPUT_DIR=output / "models" / notebook.stem)
    return settings


def read_stores(output: Path, *, repo_id: str, split: str, **kwargs):
    from datasets import load_from_disk
    from mouse_core.data import Datastore

    root = output / "datasets" / repo_id / split
    if not root.is_dir():
        raise FileNotFoundError(f"Run collector 01 or 17 first: missing local dataset {root}")
    stores = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            store = Datastore(name=path.name)
            store.from_dataset(ds=load_from_disk(str(path)))
            stores.append(store)
    if not stores:
        raise ValueError(f"No collected stores in {root}")
    print(f"Loaded {sum(map(len, stores))} real source events from {root}")
    return stores


def write_stores(namespace, output: Path):
    root = output / "datasets" / namespace["DATASET_ID"] / "train"
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for store in namespace["stores"]:
        dataset = store.to_dataset()
        dataset.save_to_disk(str(root / store.name))
        counts[store.name] = len(dataset)
    if not counts or not sum(counts.values()):
        raise AssertionError("Collector did not produce source events")
    print(f"Saved local datasets: {counts}")
    return counts


def is_publication(tree: ast.AST) -> bool:
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id in {"push_model_to_hub", "push_stores_to_hub"}
               for node in ast.walk(tree))


def run_one(*, notebook: Path, output: Path, seed: int) -> dict[str, Any]:
    import numpy as np
    import torch
    from mouse_core.data import save_tokenizer
    from mouse_core.models import load_model, save_model

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    settings = local_settings(notebook=notebook, output=output, seed=seed)
    document = json.loads(notebook.read_text())
    first_source = next("".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code")
    assigned_names = {target.id for node in ast.parse(first_source).body if isinstance(node, ast.Assign)
                      for target in node.targets if isinstance(target, ast.Name)}
    settings = {name: value for name, value in settings.items() if name in assigned_names}
    # A notebook kernel owns an actual __main__ module. Keep each execution
    # isolated while allowing saved condition references to resolve normally.
    previous_main = sys.modules["__main__"]
    notebook_module = ModuleType("__main__")
    namespace: dict[str, Any] = notebook_module.__dict__
    namespace["_local_settings"] = settings
    result: dict[str, Any] = {"notebook": notebook.name, "settings": {k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in settings.items()},
                              "train_kernel": "reference", "decode_kernel": "flex",
                              "executed_cells": [], "publication_cells_skipped": []}
    before = []
    optimizer = None
    hook = None
    steps = 0

    def record_step(optimizer, args, kwargs):
        nonlocal steps
        steps += 1
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if not torch.isfinite(parameter).all():
                    raise AssertionError("Optimizer produced nonfinite parameters")
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise AssertionError("Optimizer received nonfinite gradients")

    started = time.monotonic()
    try:
        sys.modules["__main__"] = notebook_module
        first_code = True
        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            tree = ast.parse(source)
            if is_publication(tree):
                result["publication_cells_skipped"].append(index)
                print(f"CELL {index}: publication skipped; artifacts are saved locally")
                continue
            tree = Settings(set(settings), settings_cell=first_code).visit(tree)
            ast.fix_missing_locations(tree)
            print(f"CELL {index}: executing", flush=True)
            exec(compile(tree, str(notebook) + f":cell-{index}", "exec"), namespace)
            result["executed_cells"].append(index)
            if first_code:
                # This changes only the storage location. Model and data processing are real.
                namespace["load_stores_from_hub"] = lambda **kw: read_stores(output, **kw)
                first_code = False
            if optimizer is None and "optimizer" in namespace:
                optimizer = namespace["optimizer"]
                before = [(parameter, parameter.detach().clone()) for group in optimizer.param_groups
                          for parameter in group["params"]]
                # MOUSE's AdamW delegates updates to the actual PyTorch optimizer.
                hook = optimizer._inner.register_step_post_hook(record_step)
        if optimizer is not None:
            if steps < 1:
                raise AssertionError("Training example did not execute an optimizer update")
            changed = sum(not torch.equal(parameter.detach(), original) for parameter, original in before)
            if changed == 0:
                raise AssertionError("Optimizer ran without changing any parameters")
            result.update(optimizer_steps=steps, changed_parameter_tensors=changed)
            if "loss" in namespace and not torch.isfinite(torch.as_tensor(namespace["loss"])).all():
                raise AssertionError("Training example produced a nonfinite loss")
        if notebook.name.startswith(("01_", "17_")):
            result["stored_events"] = write_stores(namespace, output)
        if "modules" in namespace and "tokenizer" in namespace:
            path = output / "models" / notebook.stem
            modules = namespace["modules"].eval().to("cpu")
            save_model(model=modules, path=path)
            save_tokenizer(tokenizer=namespace["tokenizer"], path=path)
            restored = load_model(model=deepcopy(modules), repo_id_or_path=path, map_location="cpu")
            for name, value in modules.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            result["model_roundtrip"] = str(path)
        if notebook.name.startswith("15_"):
            conditions = [field.when for field in namespace["tokenizer"].input_fields
                          if field.when is not None]
            if not conditions or not all(condition.__module__ == "__main__"
                                         and condition.__globals__ is namespace
                                         and namespace[condition.__name__] is condition
                                         for condition in conditions):
                raise AssertionError("Loaded tokenizer conditions must belong to the fresh inference notebook")
            result["tokenizer_conditions"] = "fresh inference notebook __main__"
            counts = [len(frames) for frames in namespace["frames_per_env"]]
            if not counts or not all(counts):
                raise AssertionError("Inference did not render frames for every stream")
            result["rendered_frames"] = counts
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], file=sys.stderr)
    finally:
        try:
            if hook is not None:
                hook.remove()
            for name in ("loader", "env", "eval_env"):
                resource = namespace.get(name)
                if resource is not None and hasattr(resource, "close"):
                    resource.close()
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        finally:
            sys.modules["__main__"] = previous_main
            namespace.clear()
            before.clear()
            gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-source", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--examples", nargs="+", required=True, help="all, or notebook number prefixes such as 01 02 15")
    args = parser.parse_args()
    if not args.tokenizer_source.is_dir():
        parser.error("--tokenizer-source must be an existing local tokenizer directory")
    if args.threads < 1:
        parser.error("--threads must be positive")
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "MPLBACKEND": "Agg", "SDL_VIDEODRIVER": "dummy", "CUDA_VISIBLE_DEVICES": "",
                       "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[key] = value
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    notebooks = sorted((ROOT / "examples").glob("*.ipynb"))
    if args.examples != ["all"]:
        selected = set(args.examples)
        notebooks = [path for path in notebooks if path.name.split("_", 1)[0] in selected]
        if len(notebooks) != len(selected) or len(selected) != len(args.examples):
            parser.error("unknown or duplicated notebook prefixes")
    import torch

    torch.set_num_threads(args.threads)
    manifest = {"seed": args.seed, "threads": args.threads, "tokenizer_source": str(args.tokenizer_source.resolve()),
                "checkpoint": {"hidden_size": 32, "layers": 1, "heads": 2}, "results": []}
    manifest_path = output / "results.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous[key] != manifest[key] for key in ("seed", "threads", "tokenizer_source", "checkpoint")):
            parser.error("use a new output directory when changing the seed, threads, or checkpoint settings")
        manifest["results"] = previous["results"]
    prepare_checkpoint(source=args.tokenizer_source.resolve(), target=output / "tiny-qwen3", seed=args.seed)
    failed = False
    for notebook in notebooks:
        print(f"RUN {notebook.stem}", flush=True)
        with (output / f"{notebook.stem}.log").open("w") as log:
            with redirect_stdout(log), redirect_stderr(log):
                result = run_one(notebook=notebook, output=output, seed=args.seed)
        manifest["results"] = sorted([row for row in manifest["results"] if row["notebook"] != notebook.name] + [result],
                                     key=lambda row: row["notebook"])
        (output / f"{notebook.stem}.result.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{result['status'].upper()} {notebook.stem} ({result['elapsed_seconds']}s)", flush=True)
        failed |= result["status"] != "passed"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
