from __future__ import annotations

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

    def list_repo_files(self, *, repo_id: str, repo_type: str) -> list[str]:
        assert repo_id == "user/test-dataset"
        assert repo_type == "dataset"
        return [
            "README.md",
            "dataset_infos.json",
            "data/old-00000-of-00001.parquet",
            "data/cartpole/train-00000-of-00001.parquet",
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


def test_load_stores_from_hub_loads_multiple_named_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict]] = []

    def fake_load_dataset(repo_id: str, config_name: str, **kwargs):
        calls.append((repo_id, config_name, kwargs))
        return Dataset.from_list([{
            "observation": {"discrete": len(calls)},
            "action": {"discrete": len(calls)},
            "reward": float(len(calls)),
            "done": 0,
            "time": 0,
        }])

    monkeypatch.setattr(hub, "load_dataset", fake_load_dataset)

    stores = hub.load_stores_from_hub("org/dataset", ["cartpole", "lunar"], split="train")

    assert [store.name for store in stores] == ["cartpole", "lunar"]
    assert [len(store) for store in stores] == [1, 1]
    assert calls == [
        ("org/dataset", "cartpole", {"split": "train"}),
        ("org/dataset", "lunar", {"split": "train"}),
    ]


def test_load_stores_from_hub_discovers_store_names(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict]] = []
    discovered: list[tuple[str, dict]] = []

    def fake_get_dataset_config_names(repo_id: str, **kwargs):
        discovered.append((repo_id, kwargs))
        return ["cartpole", "lunar"]

    def fake_load_dataset(repo_id: str, config_name: str, **kwargs):
        calls.append((repo_id, config_name, kwargs))
        return Dataset.from_list([{
            "observation": {"discrete": len(calls)},
            "action": {"discrete": len(calls)},
            "reward": float(len(calls)),
            "done": 0,
            "time": 0,
        }])

    monkeypatch.setattr(hub, "get_dataset_config_names", fake_get_dataset_config_names)
    monkeypatch.setattr(hub, "load_dataset", fake_load_dataset)

    stores = hub.load_stores_from_hub("org/dataset", split="train", revision="main")

    assert [store.name for store in stores] == ["cartpole", "lunar"]
    assert discovered == [("org/dataset", {"revision": "main"})]
    assert calls == [
        ("org/dataset", "cartpole", {"split": "train", "revision": "main"}),
        ("org/dataset", "lunar", {"split": "train", "revision": "main"}),
    ]


def test_load_stores_from_hub_scopes_short_names(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict]] = []

    def fake_load_dataset(repo_id: str, config_name: str, **kwargs):
        calls.append((repo_id, config_name, kwargs))
        return Dataset.from_list([{
            "observation": {"discrete": 0},
            "action": {"discrete": 0},
            "reward": 0.0,
            "done": 0,
            "time": 0,
        }])

    monkeypatch.setattr(hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(hub, "load_dataset", fake_load_dataset)

    stores = hub.load_stores_from_hub("dataset", ["config"], split="train", token="token")

    assert stores[0].name == "config"
    assert calls == [("user/dataset", "config", {"split": "train", "token": "token"})]


def test_load_stores_from_hub_requires_non_empty_store_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        hub.load_stores_from_hub("dataset", ["cartpole", ""])


def test_load_stores_from_hub_requires_discovered_store_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub, "get_dataset_config_names", lambda repo_id: [])

    with pytest.raises(ValueError, match="No store configs found"):
        hub.load_stores_from_hub("org/dataset")


def test_load_stores_from_hub_requires_unique_store_names() -> None:
    with pytest.raises(ValueError, match="unique store names"):
        hub.load_stores_from_hub("dataset", ["same", "same"])


def test_push_stores_to_hub_pushes_one_config_per_store(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _FakeHfApi()
    pushes: list[tuple[str, str, int]] = []

    monkeypatch.setattr(hub, "HfApi", lambda: api)

    def fake_push_dataset_dict(dataset_dict, *, repo_id: str, commit_message: str, config_name: str) -> None:
        assert repo_id == "user/test-dataset"
        assert commit_message == "New rollout data"
        pushes.append((config_name, next(iter(dataset_dict.keys())), len(dataset_dict["train"])))

    monkeypatch.setattr(hub, "_push_dataset_dict", fake_push_dataset_dict)

    url = hub.push_stores_to_hub(
        [_store(1, 2, name="cartpole"), _store(3, name="lunar")],
        repo_id="test-dataset",
        split="train",
        private=True,
    )

    assert url == "https://huggingface.co/datasets/user/test-dataset"
    assert api.settings_updates == [("user/test-dataset", True)]
    assert api.commits == [[
        "README.md",
        "dataset_infos.json",
        "data/old-00000-of-00001.parquet",
        "data/cartpole/train-00000-of-00001.parquet",
    ]]
    assert pushes == [("cartpole", "train", 2), ("lunar", "train", 1)]


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
