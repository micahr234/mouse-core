from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import tempfile
import pytest
from datasets import Dataset
from mouse_core.data import Datastore
from mouse_core.data import hub

class _FakeRepoUrl:
    repo_id = 'user/test-dataset'

    def __str__(self) -> str:
        return 'https://huggingface.co/datasets/user/test-dataset'

class _FakeHfApi:

    def __init__(self, *, repo_present: bool = True) -> None:
        self.settings_updates: list[tuple[str, bool]] = []
        self.commits: list[list[str]] = []
        self.commit_messages: list[str] = []
        self.card_texts: list[str] = []
        self.deleted_repos: list[str] = []
        self.create_exist_oks: list[bool] = []
        self._repo_present = repo_present

    def create_repo(self, *, repo_id: str, repo_type: str, private: bool, exist_ok: bool) -> _FakeRepoUrl:
        assert repo_id == 'test-dataset'
        assert repo_type == 'dataset'
        assert private is True
        self.create_exist_oks.append(exist_ok)
        self._repo_present = True
        return _FakeRepoUrl()

    def update_repo_settings(self, *, repo_id: str, repo_type: str, private: bool) -> None:
        assert repo_type == 'dataset'
        self.settings_updates.append((repo_id, private))

    def delete_repo(self, *, repo_id: str, repo_type: str, missing_ok: bool = False) -> None:
        assert repo_type == 'dataset'
        assert missing_ok is True
        self.deleted_repos.append(repo_id)
        self._repo_present = False

    def repo_exists(self, *, repo_id: str, repo_type: str, token: str | bool | None = None) -> bool:
        assert repo_type == 'dataset'
        return self._repo_present

    def hf_hub_download(self, *, repo_id: str, filename: str, repo_type: str, **kwargs) -> str:
        assert repo_id == 'user/test-dataset'
        assert filename == 'README.md'
        assert repo_type == 'dataset'
        path = Path(tempfile.mkdtemp()) / 'README.md'
        path.write_text('---\nconfigs: []\n---\n', encoding='utf-8')
        return str(path)

    def create_commit(self, *, repo_id: str, repo_type: str, operations: list, commit_message: str) -> None:
        assert repo_id == 'user/test-dataset'
        assert repo_type == 'dataset'
        assert commit_message
        self.commit_messages.append(commit_message)
        self.commits.append([op.path_in_repo for op in operations])
        for op in operations:
            payload = getattr(op, 'path_or_fileobj', None)
            if op.path_in_repo == 'README.md' and isinstance(payload, bytes):
                self.card_texts.append(payload.decode('utf-8'))
            elif op.path_in_repo == 'README.md' and payload is not None and not isinstance(payload, (bytes, bytearray)):
                self.card_texts.append(Path(payload).read_text(encoding='utf-8'))

    def whoami(self, token: str | bool | None=None) -> dict[str, str]:
        return {'name': 'user'}

def _store(*actions: int, name: str | None=None) -> Datastore:
    store = Datastore(name=name)
    for i, action in enumerate(actions):
        store.append({'observation': {'discrete': i}, 'action': {'discrete': action}, 'reward': float(i), 'episode_done': 0, 'task_done': 0, 'step_index': i})
    return store

def _loaded_store_datasets() -> dict[str, Dataset]:
    return {'cartpole': Dataset.from_list([{'observation': {'discrete': 1}, 'action': {'discrete': 1}, 'reward': 1.0, 'episode_done': 0, 'task_done': 0, 'step_index': 0}]), 'lunar': Dataset.from_list([{'observation': {'discrete': 2}, 'action': {'discrete': 2}, 'reward': 2.0, 'episode_done': 0, 'task_done': 0, 'step_index': 0}])}

def _write_snapshot(root: Path, *store_names: str, split: str='train') -> Path:
    for store_name in store_names:
        store_dir = root / 'data' / store_name
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / f'{split}-00000-of-00001.parquet').touch()
    return root

def test_load_stores_from_hub_loads_requested_stores_in_one_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    snapshot_calls: list[tuple[str, list[str] | None, str, str | None, str | bool | None]] = []
    snapshot_dir = _write_snapshot(tmp_path, 'cartpole', 'lunar')

    def fake_load_dataset(path: str, **kwargs):
        calls.append((path, kwargs))
        return _loaded_store_datasets()

    def fake_snapshot(*, repo_id: str, store_names: list[str] | None, split: str, revision: str | None, token: str | bool | None, force_download: bool=False):
        snapshot_calls.append((repo_id, store_names, split, revision, token))
        return snapshot_dir
    monkeypatch.setattr(hub, '_snapshot_store_repo', fake_snapshot)
    monkeypatch.setattr(hub, 'load_dataset', fake_load_dataset)
    stores = hub.load_stores_from_hub(split='train', repo_id='org/dataset', store_names=['cartpole', 'lunar'])
    assert [store.name for store in stores] == ['cartpole', 'lunar']
    assert [len(store) for store in stores] == [1, 1]
    assert snapshot_calls == [('org/dataset', ['cartpole', 'lunar'], 'train', None, None)]
    assert calls == [('parquet', {'data_files': {'cartpole': [str(snapshot_dir / 'data/cartpole/train-00000-of-00001.parquet')], 'lunar': [str(snapshot_dir / 'data/lunar/train-00000-of-00001.parquet')]}})]
    assert '*' not in repr(calls[0][1]['data_files'])

def test_load_stores_from_hub_discovers_store_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    snapshot_calls: list[tuple[str, list[str] | None, str, str | None, str | bool | None]] = []
    snapshot_dir = _write_snapshot(tmp_path, 'cartpole', 'lunar')

    def fake_load_dataset(path: str, **kwargs):
        calls.append((path, kwargs))
        return _loaded_store_datasets()

    def fake_snapshot(*, repo_id: str, store_names: list[str] | None, split: str, revision: str | None, token: str | bool | None, force_download: bool=False):
        snapshot_calls.append((repo_id, store_names, split, revision, token))
        return snapshot_dir
    monkeypatch.setattr(hub, '_snapshot_store_repo', fake_snapshot)
    monkeypatch.setattr(hub, 'load_dataset', fake_load_dataset)
    stores = hub.load_stores_from_hub(split='train', revision='main', repo_id='org/dataset')
    assert [store.name for store in stores] == ['cartpole', 'lunar']
    assert snapshot_calls == [('org/dataset', None, 'train', 'main', None)]
    assert calls == [('parquet', {'data_files': {'cartpole': [str(snapshot_dir / 'data/cartpole/train-00000-of-00001.parquet')], 'lunar': [str(snapshot_dir / 'data/lunar/train-00000-of-00001.parquet')]}})]

def test_load_stores_from_hub_scopes_short_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []
    snapshot_calls: list[tuple[str, list[str] | None, str, str | None, str | bool | None]] = []
    snapshot_dir = _write_snapshot(tmp_path, 'config')

    def fake_load_dataset(path: str, **kwargs):
        calls.append((path, kwargs))
        return {'config': Dataset.from_list([{'observation': {'discrete': 0}, 'action': {'discrete': 0}, 'reward': 0.0, 'episode_done': 0, 'task_done': 0, 'step_index': 0}])}

    def fake_snapshot(*, repo_id: str, store_names: list[str] | None, split: str, revision: str | None, token: str | bool | None, force_download: bool=False):
        snapshot_calls.append((repo_id, store_names, split, revision, token))
        return snapshot_dir
    monkeypatch.setattr(hub, 'HfApi', _FakeHfApi)
    monkeypatch.setattr(hub, '_snapshot_store_repo', fake_snapshot)
    monkeypatch.setattr(hub, 'load_dataset', fake_load_dataset)
    stores = hub.load_stores_from_hub(split='train', token='token', repo_id='dataset', store_names=['config'])
    assert stores[0].name == 'config'
    assert snapshot_calls == [('user/dataset', ['config'], 'train', None, 'token')]
    assert calls == [('parquet', {'data_files': {'config': [str(snapshot_dir / 'data/config/train-00000-of-00001.parquet')]}})]

def test_load_stores_from_hub_requires_non_empty_store_names() -> None:
    with pytest.raises(ValueError, match='non-empty'):
        hub.load_stores_from_hub(repo_id='dataset', store_names=['cartpole', ''])

def test_load_stores_from_hub_requires_discovered_store_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hub, '_snapshot_store_repo', lambda repo_id, *, store_names, split, revision, token, force_download=False: Path('/tmp/no-store-snapshot'))
    with pytest.raises(ValueError, match='No parquet store configs found'):
        hub.load_stores_from_hub(repo_id='org/dataset')

def test_load_stores_from_hub_requires_unique_store_names() -> None:
    with pytest.raises(ValueError, match='unique store names'):
        hub.load_stores_from_hub(repo_id='dataset', store_names=['same', 'same'])

def test_push_stores_to_hub_pushes_one_config_per_store(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _FakeHfApi()
    commits: list[dict] = []
    monkeypatch.setattr(hub, 'HfApi', lambda: api)

    def fake_commit_dataset_repo(*, api, repo_id: str, folder_path: Path, commit_message: str) -> None:
        assert repo_id == 'user/test-dataset'
        assert commit_message == 'New rollout data'
        commits.append({'files': sorted((path.relative_to(folder_path).as_posix() for path in folder_path.rglob('*') if path.is_file())), 'readme': (folder_path / 'README.md').read_text(encoding='utf-8')})
    monkeypatch.setattr(hub, '_commit_dataset_repo', fake_commit_dataset_repo)
    url = hub.push_stores_to_hub(repo_id='test-dataset', split='train', private=True, stores=[_store(1, 2, name='cartpole'), _store(3, name='lunar')])
    assert url == 'https://huggingface.co/datasets/user/test-dataset'
    assert api.settings_updates == [('user/test-dataset', True)]
    assert api.deleted_repos == ['user/test-dataset']
    assert api.create_exist_oks == [False]
    assert api.commit_messages == ['Enable dataset viewer']
    assert api.card_texts == ['---\nviewer: true\nconfigs:\n- config_name: cartpole\n  data_files:\n  - split: train\n    path: data/cartpole/train-*.parquet\n- config_name: lunar\n  data_files:\n  - split: train\n    path: data/lunar/train-*.parquet\n---\n']
    assert commits == [{'files': ['README.md', 'data/cartpole/train-00000-of-00001.parquet', 'data/lunar/train-00000-of-00001.parquet'], 'readme': '---\nviewer: false\nconfigs:\n- config_name: cartpole\n  data_files:\n  - split: train\n    path: data/cartpole/train-*.parquet\n- config_name: lunar\n  data_files:\n  - split: train\n    path: data/lunar/train-*.parquet\n---\n'}]

def test_delete_dataset_repo_if_exists_scopes_short_names() -> None:
    api = _FakeHfApi()
    assert hub._delete_dataset_repo_if_exists(cast(Any, api), 'test-dataset') == 'user/test-dataset'
    assert api.deleted_repos == ['user/test-dataset']

def test_card_with_viewer_sets_true_and_false() -> None:
    card = '---\nconfigs:\n- config_name: cartpole\n---\n'
    assert hub._card_with_viewer(card, viewer=False).startswith('---\nviewer: false\n')
    assert hub._card_with_viewer(card, viewer=True).startswith('---\nviewer: true\n')

def test_push_stores_to_hub_clear_true_deletes_then_toggles_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    """clear=True deletes the repo, uploads with viewer: false, then sets viewer: true."""
    api = _FakeHfApi()
    monkeypatch.setattr(hub, 'HfApi', lambda: api)
    hub.push_stores_to_hub(repo_id='test-dataset', private=True, stores=[_store(1, name='cartpole')])
    assert api.deleted_repos == ['user/test-dataset']
    assert api.create_exist_oks == [False]
    assert api.commit_messages == ['New rollout data', 'Enable dataset viewer']
    assert api.commits[0] == ['README.md', 'data/cartpole/train-00000-of-00001.parquet']
    assert api.commits[1] == ['README.md']
    assert 'viewer: false' in api.card_texts[0]
    assert 'viewer: true' in api.card_texts[1]

def test_push_stores_to_hub_clear_false_deletes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """clear=False layers the push on top of existing files: additions only."""
    api = _FakeHfApi()
    monkeypatch.setattr(hub, 'HfApi', lambda: api)
    hub.push_stores_to_hub(repo_id='test-dataset', clear=False, private=True, stores=[_store(1, name='cartpole')])
    assert api.deleted_repos == []
    assert api.create_exist_oks == [True]
    assert api.commit_messages == ['New rollout data']
    assert api.commits == [['README.md', 'data/cartpole/train-00000-of-00001.parquet']]
    assert api.card_texts == ['---\nconfigs:\n- config_name: cartpole\n  data_files:\n  - split: train\n    path: data/cartpole/train-*.parquet\n---\n']
    assert 'viewer:' not in api.card_texts[0]

def test_push_to_hub_clear_flag_deletes_then_toggles_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    api = _FakeHfApi()
    pushes: list[str] = []
    monkeypatch.setattr(hub, 'HfApi', lambda: api)
    monkeypatch.setattr(hub, '_push_dataset_dict', lambda dataset_dict, *, repo_id, commit_message, config_name: pushes.append(config_name))
    hub.push_to_hub(repo_id='test-dataset', private=True, splits={'train': [_store(1, name='cartpole')]})
    assert api.deleted_repos == ['user/test-dataset']
    assert api.create_exist_oks == [False]
    assert pushes == ['default']
    assert api.commit_messages == ['Upload dataset with viewer: false', 'Enable dataset viewer']
    assert api.card_texts[0].startswith('---\nviewer: false\n')
    assert api.card_texts[1].startswith('---\nviewer: true\n')
    hub.push_to_hub(repo_id='test-dataset', clear=False, private=True, splits={'train': [_store(1, name='cartpole')]})
    assert api.deleted_repos == ['user/test-dataset']
    assert api.create_exist_oks == [False, True]
    assert pushes == ['default', 'default']
    assert api.commit_messages == ['Upload dataset with viewer: false', 'Enable dataset viewer']

def test_push_stores_to_hub_requires_named_stores() -> None:
    with pytest.raises(ValueError, match='non-empty name'):
        hub.push_stores_to_hub(repo_id='test-dataset', stores=[_store(1), _store(2, name='cartpole')])

def test_push_stores_to_hub_requires_unique_store_names() -> None:
    with pytest.raises(ValueError, match='unique store names'):
        hub.push_stores_to_hub(repo_id='test-dataset', stores=[_store(1, name='same'), _store(2, name='same')])

@pytest.mark.parametrize('bad_name', ['env#0', 'frozenlake_slippery#1', 'my env', 'config?subset=1', 'config/sub'])
def test_push_stores_to_hub_rejects_url_unsafe_names(bad_name: str) -> None:
    with pytest.raises(ValueError, match='safe as Hugging Face'):
        hub.push_stores_to_hub(repo_id='test-dataset', stores=[_store(1, name=bad_name)])
