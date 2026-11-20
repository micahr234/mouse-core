"""Standalone Core storage: shared bases, mutable buffers and atomic snapshots."""
import copy
import json
import pytest
import torch
from torch import nn
from mouse_core import CheckpointStore, load_checkpoint
from mouse_core.checkpoint import CHECKPOINT_FORMAT


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2).requires_grad_(False)
        self.head = nn.Linear(2, 1)
        self.alias = self.base
        self.register_buffer("count", torch.tensor(0))


def test_snapshots_roundtrip_frozen_aliases_and_changing_buffers(tmp_path):
    model = TinyModel()
    before = copy.deepcopy(model)
    store = CheckpointStore(tmp_path / "run")
    first = store.save(step=0, model=model, metadata={"experiment": "example"}, files={"settings.toml": "x=1"})
    shared = (store.directory / "frozen_weights.bin").read_bytes()
    with torch.no_grad():
        model.head.weight.add_(1)
        model.count.add_(1)
    second = store.save(step=10, model=model)
    assert shared == (store.directory / "frozen_weights.bin").read_bytes()
    assert set(torch.load(first / "pytorch_model.bin", weights_only=True)) == {"head.weight", "head.bias", "count"}
    assert set(torch.load(store.directory / "frozen_weights.bin", weights_only=True)) == {
        "base.weight", "base.bias", "alias.weight", "alias.bias"}
    assert json.loads((first / "metadata.json").read_text()) == {"step": 0, "experiment": "example"}
    restored = TinyModel()
    for path, expected in ((first, before), (store.directory, model)):
        assert load_checkpoint(path, model=restored) is restored
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
    # Retrying publication of an existing step must preserve that snapshot.
    assert store.save(step=10, model=model) == second


def test_shared_initialization_failure_is_retryable(tmp_path, monkeypatch):
    import mouse_core.checkpoint as storage
    model = TinyModel()
    store = CheckpointStore(tmp_path / "run")
    save = storage.torch.save
    def fail(*args, **kwargs):
        raise OSError("disk error")
    monkeypatch.setattr(storage.torch, "save", fail)
    with pytest.raises(OSError):
        store.save(step=0, model=model)
    assert not store.directory.exists()
    monkeypatch.setattr(storage.torch, "save", save)
    assert store.save(step=0, model=model).is_dir()


def test_failed_step_preserves_last_snapshot_and_can_retry(tmp_path, monkeypatch):
    import mouse_core.checkpoint as storage
    model = TinyModel()
    store = CheckpointStore(tmp_path / "run")
    first = store.save(step=0, model=model)
    save = storage.torch.save
    def fail(*args, **kwargs):
        raise OSError("disk error")
    monkeypatch.setattr(storage.torch, "save", fail)
    with pytest.raises(OSError):
        store.save(step=1, model=model)
    assert json.loads((store.directory / "latest.json").read_text())["snapshot"] == first.name
    assert not list(store.directory.glob(".*.tmp"))
    monkeypatch.setattr(storage.torch, "save", save)
    assert store.save(step=1, model=model).is_dir()


def test_cannot_reuse_another_run_or_change_trainability(tmp_path):
    model = TinyModel()
    store = CheckpointStore(tmp_path / "run")
    store.save(step=0, model=model)
    with pytest.raises(FileExistsError):
        CheckpointStore(store.directory).save(step=1, model=model)
    model.base.requires_grad_(True)
    with pytest.raises(ValueError, match="trainability"):
        store.save(step=1, model=model)


@pytest.mark.parametrize("name", ["../escape", "..", ".", "", "config.json", "metadata.json", "pytorch_model.bin"])
def test_attachments_cannot_escape_or_overwrite_storage(tmp_path, name):
    with pytest.raises(ValueError):
        CheckpointStore(tmp_path / "run").save(step=0, model=TinyModel(), files={name: "bad"})
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("name", ["../escape", "..", ".", "", 3])
def test_latest_cannot_escape_checkpoint_directory(tmp_path, name):
    (tmp_path / "latest.json").write_text(json.dumps({"snapshot": name}))
    with pytest.raises(ValueError, match="snapshot name"):
        load_checkpoint(tmp_path, model=TinyModel())


def test_strict_load_validation_happens_before_parameter_mutation(tmp_path):
    model = TinyModel()
    store = CheckpointStore(tmp_path / "run")
    path = store.save(step=0, model=model)
    target = TinyModel()
    before = copy.deepcopy(target.state_dict())
    state = torch.load(path / "pytorch_model.bin", weights_only=True)
    state["head.bias"] = torch.zeros(3)
    torch.save(state, path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(path, model=target)
    for name, value in before.items():
        torch.testing.assert_close(target.state_dict()[name], value)
    assert json.loads((path / "config.json").read_text())["format"] == CHECKPOINT_FORMAT
