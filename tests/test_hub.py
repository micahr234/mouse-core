from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset

from mouse_core.data import Datastore
from mouse_core.data import hub


class _FakeRepoUrl:
    repo_id = "user/test-dataset"

    def __str__(self) -> str:
        return "https://huggingface.co/datasets/user/test-dataset"


class _FakeHfApi:
    def __init__(self) -> None:
        self.settings_updates: list[tuple[str, bool]] = []
        self.commits: list[list[str]] = []

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> _FakeRepoUrl:
        assert repo_id == "test-dataset"
        assert repo_type == "dataset"
        assert private is True
        assert exist_ok is True
        return _FakeRepoUrl()

    def update_repo_settings(self, *, repo_id: str, repo_type: str, private: bool) -> None:
        assert repo_type == "dataset"
        self.settings_updates.append((repo_id, private))

    def list_repo_files(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str | None = None,
        token: str | bool | None = None,
    ) -> list[str]:
        assert repo_id == "user/test-dataset"
        assert repo_type == "dataset"
        assert revision is None
        assert token is None
        return [
            "README.md",
            "dataset_infos.json",
            "data/old-00000-of-00001.parquet",
            "data/cartpole/train-00000-of-00001.parquet",
            "notes/old.txt",
        ]

    def create_commit(self, *, repo_id: str, repo_type: str, operations: list, commit_message: str) -> None:
        assert repo_id == "user/test-dataset"
        assert repo_type == "dataset"
        assert commit_message
        self.commits.append([op.path_in_repo for op in operations])

    def whoami(self, token: str | bool | None = None) -> dict[str, str]:
        assert token == "token"
        return {"name": "user"}


def _store(*actions: int, name: str | None = None) -> Datastore:
    store = Datastore(name=name)
    for i, action in enumerate(actions):
        store.append({
            "observation": {"discrete": i},
            "action": {"discrete": action},
            "reward": float(i),
            "done": 0,
            "time": i,
        })
    return store


def _loaded_store_datasets() -> dict[str, Dataset]:
    return {
        "cartpole": Dataset.from_list([{
            "observation": {"discrete": 1},
            "action": {"discrete": 1},
            "reward": 1.0,
            "done": 0,
            "time": 0,
        }]),
        "lunar": Dataset.from_list([{
            "observation": {"discrete": 2},
            "action": {"discrete": 2},
            "reward": 2.0,
            "done": 0,
            "time": 0,
        }]),
    }


def _write_snapshot(root: Path, *store_names: str, split: str = "train") -> Path:
    for store_name in store_names:
        store_dir = root / "data" / store_name
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / f"{split}-00000-of-00001.parquet").touch()
    return root


def test_load_stores_from_hub_loads_requested_stores_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict]] = []
    snapshot_calls: list[tuple[str, list[str] | None, str, str | None, str | bool | None]] = []
    snapshot_dir = _write_snapshot(tmp_path, "cartpole", "lunar")

    def fake_load_dataset(path: str, **kwargs):
        calls.append((path, kwargs))
        return _loaded_store_datasets()

    def fake_snapshot(repo_id: str, *, store_names: list[str] | None, split: str, revision: str | None, token: str | bool | None, force_download: bool = False):
        snapshot_calls.append((repo_id, store_names, split, revision, token))
        return snapshot_dir

    monkeypatch.setattr(hub, "_snapshot_store_repo", fake_snapshot)
    monkeypatch.setattr(hub, "load_dataset", fake_load_dataset)

    stores = hub.load_stores_from_hub("org/dataset", ["cartpole", "lunar"], split="train")

    assert [store.name for store in stores] == ["cartpole", "lunar"]
    assert [len(store) for store in stores] == [1, 1]
    assert snapshot_calls == [("org/dataset", ["cartpole", "lunar"], "train", None, None)]
    assert calls == [("parquet", {
        "data_files": {
            "cartpole": [str(snapshot_dir / "data/cartpole/train-00000-of-00001.parquet")],
            "lunar": [str(snapshot_dir / "data/lunar/train-00000-of-00001.parquet")],
        },
    })]
    assert "*" not in repr(calls[0][1]["data_files"])


def test_load_stores_from_hub_discovers_store_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict]] = []
    snapshot_calls: list[tuple[str, list[str] | None, str, str | None, str | bool | None]] = []
    snapshot_dir = _write_snapshot(tmp_path, "cartpole", "lunar")

    def fake_load_dataset(path: str, **kwargs):
        calls.append((path, kwargs))
        return _loaded_store_datasets()

    def fake_snapshot(repo_id: str, *, store_names: list[str] | None, split: str, revision: str | None, token: str | bool | None, force_download: bool = False):
        snapshot_calls.append((repo_id, store_names, split, revision, token))
        return snapshot_dir

    monkeypatch.setattr(hub, "_snapshot_store_repo", fake_snapshot)
    monkeypatch.setattr(hub, "load_dataset", fake_load_dataset)

    stores = hub.load_stores_from_hub("org/dataset", split="train", revision="main")

    assert [store.name for store in stores] == ["cartpole", "lunar"]
    assert snapshot_calls == [("org/dataset", None, "train", "main", None)]
    assert calls == [("parquet", {
        "data_files": {
            "cartpole": [str(snapshot_dir / "data/cartpole/train-00000-of-00001.parquet")],
            "lunar": [str(snapshot_dir / "data/lunar/train-00000-of-00001.parquet")],
        },
    })]


def test_load_stores_from_hub_scopes_short_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict]] = []
    snapshot_calls: list[tuple[str, list[str] | None, str, str | None, str | bool | None]] = []
    snapshot_dir = _write_snapshot(tmp_path, "config")

    def fake_load_dataset(path: str, **kwargs):
        calls.append((path, kwargs))
        return {
            "config": Dataset.from_list([{
                "observation": {"discrete": 0},
                "action": {"discrete": 0},
                "reward": 0.0,
                "done": 0,
                "time": 0,
            }])
        }

    def fake_snapshot(repo_id: str, *, store_names: list[str] | None, split: str, revision: str | None, token: str | bool | None, force_download: bool = False):
        snapshot_calls.append((repo_id, store_names, split, revision, token))
        return snapshot_dir

    monkeypatch.setattr(hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(hub, "_snapshot_store_repo", fake_snapshot)
    monkeypatch.setattr(hub, "load_dataset", fake_load_dataset)

    stores = hub.load_stores_from_hub("dataset", ["config"], split="train", token="token")

    assert stores[0].name == "config"
    assert snapshot_calls == [("user/dataset", ["config"], "train", None, "token")]
    assert calls == [("parquet", {
        "data_files": {
            "config": [str(snapshot_dir / "data/config/train-00000-of-00001.parquet")],
        },
    })]


def test_load_stores_from_hub_requires_non_empty_store_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hub.load_stores_from_hub("dataset", ["cartpole", ""])


def test_load_stores_from_hub_requires_discovered_store_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hub,
        "_snapshot_store_repo",
        lambda repo_id, *, store_names, split, revision, token, force_download=False: Path("/tmp/no-store-snapshot"),
    )

    with pytest.raises(ValueError, match="No parquet store configs found"):
        hub.load_stores_from_hub("org/dataset")


def test_load_stores_from_hub_requires_unique_store_names() -> None:
    with pytest.raises(ValueError, match="unique store names"):
        hub.load_stores_from_hub("dataset", ["same", "same"])


def test_push_stores_to_hub_pushes_one_config_per_store(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _FakeHfApi()
    commits: list[dict] = []

    monkeypatch.setattr(hub, "HfApi", lambda: api)

    def fake_commit_dataset_repo(
        api,
        *,
        repo_id: str,
        folder_path: Path,
        commit_message: str,
        clear: bool,
    ) -> None:
        assert repo_id == "user/test-dataset"
        assert commit_message == "New rollout data"
        commits.append({
            "files": sorted(path.relative_to(folder_path).as_posix() for path in folder_path.rglob("*") if path.is_file()),
            "readme": (folder_path / "README.md").read_text(encoding="utf-8"),
            "clear": clear,
        })

    monkeypatch.setattr(hub, "_commit_dataset_repo", fake_commit_dataset_repo)

    url = hub.push_stores_to_hub(
        [_store(1, 2, name="cartpole"), _store(3, name="lunar")],
        repo_id="test-dataset",
        split="train",
        private=True,
    )

    assert url == "https://huggingface.co/datasets/user/test-dataset"
    assert api.settings_updates == [("user/test-dataset", True)]
    assert api.commits == []
    assert commits == [{
        "files": [
            "README.md",
            "data/cartpole/train-00000-of-00001.parquet",
            "data/lunar/train-00000-of-00001.parquet",
        ],
        "readme": (
            "---\n"
            "configs:\n"
            "- config_name: cartpole\n"
            "  data_files:\n"
            "  - split: train\n"
            "    path: data/cartpole/train-*.parquet\n"
            "- config_name: lunar\n"
            "  data_files:\n"
            "  - split: train\n"
            "    path: data/lunar/train-*.parquet\n"
            "---\n"
        ),
        "clear": True,
    }]


def test_repo_files_to_clear_deletes_everything_except_replacements() -> None:
    api = _FakeHfApi()

    to_delete = hub._repo_files_to_clear(
        cast(Any, api),
        repo_id="user/test-dataset",
        addition_paths={
            "README.md",
            "data/cartpole/train-00000-of-00001.parquet",
        },
    )

    assert to_delete == [
        "dataset_infos.json",
        "data/old-00000-of-00001.parquet",
        "notes/old.txt",
    ]


def test_push_stores_to_hub_requires_named_stores() -> None:
    with pytest.raises(ValueError, match="non-empty name"):
        hub.push_stores_to_hub([_store(1), _store(2, name="cartpole")], repo_id="test-dataset")


def test_push_stores_to_hub_requires_unique_store_names() -> None:
    with pytest.raises(ValueError, match="unique store names"):
        hub.push_stores_to_hub([_store(1, name="same"), _store(2, name="same")], repo_id="test-dataset")


@pytest.mark.parametrize("bad_name", [
    "env#0",
    "frozenlake_slippery#1",
    "my env",
    "config?subset=1",
    "config/sub",
])
def test_push_stores_to_hub_rejects_url_unsafe_names(bad_name: str) -> None:
    with pytest.raises(ValueError, match="safe as Hugging Face"):
        hub.push_stores_to_hub([_store(1, name=bad_name)], repo_id="test-dataset")
