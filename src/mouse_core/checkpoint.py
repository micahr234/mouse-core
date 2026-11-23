"""Local model snapshots with frozen parameters and tokenizers stored once.

Storage only: callers own scheduling, metadata, run names and publication.
Snapshots restore model weights, not optimizer state or training progress.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping
from urllib.parse import quote

import torch
from mouse_core.data import Tokenizer, load_tokenizer, save_tokenizer

def _snapshot_path(path: str | Path) -> Path:
    path = Path(path)
    if (path / "latest.json").is_file():
        name = json.loads((path / "latest.json").read_text())["snapshot"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid checkpoint snapshot name")
        path = path / name
    return path


def load_checkpoint(path: str | Path, *, model: torch.nn.Module) -> torch.nn.Module:
    """Restore a step snapshot (or the latest in a checkpoint directory).

    The containing checkpoint directory supplies the frozen weights saved once
    per run. Construct the model first; no optimizer or delayed state is restored.
    """
    path = _snapshot_path(path)
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict) or config.keys() != {"class"}:
        raise ValueError("Checkpoint metadata must contain exactly the model class")
    expected = f"{type(model).__module__}.{type(model).__qualname__}"
    if config.get("class") != expected:
        raise ValueError(f"Model class mismatch: saved {config.get('class')!r}, supplied {expected!r}")
    state = torch.load(path.parent / "frozen_weights.bin", map_location="cpu", weights_only=True)
    changing = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if state.keys() & changing.keys():
        raise ValueError("Frozen and changing checkpoint weights overlap")
    state.update(changing)
    current = model.state_dict()
    if state.keys() != current.keys():
        raise RuntimeError("Model state keys do not match saved weights")
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and isinstance(current[name], torch.Tensor):
            if value.shape != current[name].shape:
                raise RuntimeError(f"Model size mismatch for {name}: saved {tuple(value.shape)}, supplied {tuple(current[name].shape)}")
    model.load_state_dict(state, strict=True)
    return model


def load_checkpoint_tokenizer(path: str | Path) -> Tokenizer:
    """Load the tokenizer shared by all snapshots in this run."""
    from transformers import AutoTokenizer

    path = _snapshot_path(path)
    if not (path / "tokenizer" / "tokenizer.json").is_file():
        path = path.parent
    spec = json.loads((path / "tokenizer" / "tokenizer.json").read_text())
    backends = {
        name: AutoTokenizer.from_pretrained(path / "tokenizer_backends" / quote(name, safe=""), local_files_only=True)
        for name, source in spec["tokenizers"].items() if not isinstance(source, dict)
    }
    return load_tokenizer(repo_id_or_path=str(path / "tokenizer"), tokenizers=backends)


class CheckpointStore:
    """Write atomic step snapshots into a new, exclusively owned directory.

    Frozen parameters are stored once; learnable parameters and all persistent
    buffers are saved at every step. Trainability must remain fixed. A repeated
    step returns its existing snapshot, allowing a caller to retry publication.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._frozen_names: set[str] | None = None
        self._model_class: str | None = None
        self._schema: dict | None = None

    def save(self, *, step: int, model: torch.nn.Module,
             tokenizer: Tokenizer | None = None, metadata: Mapping | None = None,
             files: Mapping[str, str] | None = None) -> Path:
        if type(step) is not int or step < 0:
            raise ValueError("checkpoint step must be a nonnegative integer")
        files = {} if files is None else dict(files)
        reserved = {"config.json", "metadata.json", "pytorch_model.bin"}
        for name, contents in files.items():
            if (not isinstance(name, str) or not name or Path(name).name != name
                    or name in reserved | {".", ".."}):
                raise ValueError("Checkpoint attachment must have an unreserved file name")
            if not isinstance(contents, str):
                raise TypeError("Checkpoint attachments must contain text")
        metadata_text = json.dumps({**(metadata or {}), "step": step}, indent=2) + "\n"
        frozen_names = {name for name, parameter in model.named_parameters(remove_duplicate=False)
                        if not parameter.requires_grad}
        state = model.state_dict()
        model_class = f"{type(model).__module__}.{type(model).__qualname__}"
        schema = {name: (value.shape, value.dtype) if isinstance(value, torch.Tensor)
                  else type(value) for name, value in state.items()}
        if self._frozen_names is not None:
            if frozen_names != self._frozen_names:
                raise ValueError("Checkpoint parameter trainability changed within a run")
            if model_class != self._model_class or schema != self._schema:
                raise ValueError("Checkpoint model schema changed within a run")
        else:
            # Never reuse frozen weights from another run with the same name.
            self.directory.mkdir(parents=True, exist_ok=False)
            try:
                torch.save({name: value for name, value in state.items() if name in frozen_names},
                           self.directory / "frozen_weights.bin")
                if tokenizer is not None:
                    save_tokenizer(tokenizer=tokenizer, path=self.directory / "tokenizer")
                    for name, backend in tokenizer.tokenizers.items():
                        from mouse_core.data.vocabulary import VocabularyTokenizer
                        if isinstance(backend, VocabularyTokenizer):
                            continue
                        save_backend = getattr(backend, "save_pretrained", None)
                        if not callable(save_backend):
                            raise TypeError(f"Checkpoint tokenizer backend {name!r} must support save_pretrained")
                        save_backend(self.directory / "tokenizer_backends" / quote(name, safe=""))
            except BaseException:
                shutil.rmtree(self.directory)
                raise
            self._frozen_names = frozen_names
            self._model_class = model_class
            self._schema = schema
        path = self.directory / f"step-{step:08d}"
        if not path.exists():
            temporary = self.directory / f".step-{step:08d}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            try:
                torch.save({name: value for name, value in state.items() if name not in frozen_names},
                           temporary / "pytorch_model.bin")
                (temporary / "config.json").write_text(json.dumps({
                    "class": model_class,
                }, indent=2) + "\n")
                (temporary / "metadata.json").write_text(metadata_text)
                for name, contents in files.items():
                    (temporary / name).write_text(contents)
                temporary.rename(path)
            except BaseException:
                shutil.rmtree(temporary)
                raise
        pointer = self.directory / ".latest.tmp"
        pointer.write_text(json.dumps({"snapshot": path.name}) + "\n")
        pointer.replace(self.directory / "latest.json")
        return path
