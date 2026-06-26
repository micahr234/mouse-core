"""Hugging Face Hub utilities for pushing Datastore data.

Pushes write the raw rows (whatever shape you stored) using standard
``DatasetDict.push_to_hub`` with ``config_name`` for subsets/bins.

Before each push we wipe previous parquet shards, ``dataset_infos.json``,
and the README (dataset card). This ensures that ``push_to_hub`` writes a
fresh README whose leading ``dataset_info:`` (features, splits, sizes, etc.)
exactly matches the dataset being uploaded right now.

This is required when uploading a brand new dataset or when the schema,
configs, or split structure has changed.

If you want a custom ``configs:`` YAML block (per the HF repository structure
guide) in addition to the auto-generated info, add or restore it after the
push (e.g. by editing on the Hub) or manage the card separately.

See https://huggingface.co/docs/datasets/repository_structure

Public API
----------
``load_stores_from_hub(repo_id, split=...)``
    Load named stores from Hub dataset configs in one batched parquet read,
    discovering names by default.

``push_to_hub(splits, repo_id, config_name=...)``
    Push splits under a named config (subset/bin).

``push_stores_to_hub(stores, repo_id, split=...)``
    Single-split convenience with one config per named store.
"""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from datasets import config as datasets_config
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import CommitOperationAdd, CommitOperationDelete

if TYPE_CHECKING:
    from mouse_core.data.datastore import Datastore


# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------

_HUB_PARQUET_SHARD_RE = re.compile(r"^data/(?:[^/]+/)?[^/]+-\d{5}-of-\d{5}\.parquet$")


def _wipe_hub_repo_data(api: HfApi, repo_id: str) -> None:
    """Delete previous data shards, dataset infos, and the README card.

    We remove the README (``README.md``) so that the subsequent
    ``DatasetDict.push_to_hub`` call generates a brand new dataset card.
    The leading ``dataset_info:`` section (features, splits, sizes, etc.)
    in that card must reflect the exact data being uploaded now.

    This matters when:
    - Uploading to a brand new dataset repository, or
    - The columns, configs, or split layout have changed since the last push.

    We also clean stale parquet shards and ``dataset_infos.json`` so old
    split/config metadata does not leak into the new card.
    """
    try:
        files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except RepositoryNotFoundError:
        return
    stale = datasets_config.DATASETDICT_INFOS_FILENAME
    to_delete = [
        f for f in files
        if _HUB_PARQUET_SHARD_RE.match(f)
        or f in (datasets_config.REPOCARD_FILENAME, stale)
    ]
    if to_delete:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=f) for f in to_delete],
            commit_message="chore: wipe stale shards + card before fresh push",
        )


# ---------------------------------------------------------------------------
# Schema alignment
# ---------------------------------------------------------------------------

def _is_null_typed(feature: Any) -> bool:
    return isinstance(feature, Value) and feature.dtype == "null"


def _placeholder_column_like(ref_ds: Dataset, col: str, n_rows: int) -> list[Any]:
    """Build ``n_rows`` placeholder values matching the shape/dtype of ``ref_ds[col][0]``."""
    sample = ref_ds[col][0]
    if isinstance(sample, str):
        return [""] * n_rows
    if isinstance(sample, (float, np.floating)):
        return [float("nan")] * n_rows
    if isinstance(sample, (bool, np.bool_)):
        return [False] * n_rows
    if isinstance(sample, (int, np.integer)):
        return [0] * n_rows

    def _fill_array(a: np.ndarray) -> np.ndarray:
        if a.dtype.kind == "f":
            return np.full_like(a, np.nan)
        if a.dtype.kind in ("i", "u"):
            return np.zeros_like(a)
        if a.dtype.kind == "b":
            return np.zeros_like(a, dtype=bool)
        return np.zeros_like(a)

    if isinstance(sample, list):
        a = np.asarray(sample)
        if a.dtype == object or a.size == 0:
            return [[] for _ in range(n_rows)]
        row = _fill_array(a).tolist()
        return [list(row) for _ in range(n_rows)]

    a = np.asarray(sample)
    if a.dtype == object:
        return [None] * n_rows
    if a.size == 0:
        return [a.copy() for _ in range(n_rows)]
    fill = _fill_array(a)
    return [fill.copy() for _ in range(n_rows)]


def _align_splits(splits: dict[str, Dataset]) -> dict[str, Dataset]:
    """Ensure every split has identical columns, types, and order.

    Three cases are handled:

    - A column is entirely absent from a split (different env params across runs).
    - A column exists but is ``Value('null')`` in one split while another has a
      proper dtype — the non-null type wins.
    - Column order differs between splits.

    Strategy: build a canonical ``Features`` schema (preferring non-null types)
    then fill missing columns with placeholders and ``Dataset.cast`` to it.
    """
    if len(splits) <= 1:
        return splits

    # Canonical column order: union in insertion order across all splits.
    canonical_cols: list[str] = []
    seen: set[str] = set()
    for ds in splits.values():
        for col in ds.column_names:
            if col not in seen:
                canonical_cols.append(col)
                seen.add(col)

    # For each column, prefer a split with a non-null dtype as the reference.
    best_feature: dict[str, Any] = {}
    for col in canonical_cols:
        for ds in splits.values():
            if col in ds.column_names and not _is_null_typed(ds.features[col]):
                best_feature[col] = ds.features[col]
                break
        if col not in best_feature:
            for ds in splits.values():
                if col in ds.column_names:
                    best_feature[col] = ds.features[col]
                    break

    canonical_features = Features(best_feature)

    out: dict[str, Dataset] = {}
    for name, ds in splits.items():
        cur = ds
        for col in canonical_cols:
            if col not in cur.column_names:
                ref_ds = next(d for d in splits.values() if col in d.column_names)
                cur = cur.add_column(
                    col,
                    _placeholder_column_like(ref_ds=ref_ds, col=col, n_rows=len(cur)),
                )
        cur = cur.select_columns(canonical_cols).cast(canonical_features)
        out[name] = cur
    return out


def _raise_for_hub_http_error(error: HfHubHTTPError, repo_id: str) -> NoReturn:
    code = getattr(getattr(error, "response", None), "status_code", None)
    if code == 403:
        raise RuntimeError(
            "Hugging Face returned 403 when creating or pushing the dataset. "
            f"repo_id={repo_id!r}: you need write access to that namespace "
            "(use a short name to push under your logged-in user, e.g. "
            "'my_dataset', or 'youruser/my_dataset' / an org you belong to). "
            "Check your token at https://huggingface.co/settings/tokens"
        ) from error
    raise error


def _build_split_datasets(splits: dict[str, list[Datastore]]) -> dict[str, Dataset]:
    from mouse_core.data.datastore import Datastore as _DS

    resolved: dict[str, Dataset] = {}
    for split_name, stores in splits.items():
        merged = _DS()
        merged.append(stores)
        combined = merged.to_dataset()
        if len(combined) > 0:
            resolved[split_name] = combined
    return _align_splits(resolved)


def _create_or_update_dataset_repo(
    api: HfApi,
    repo_id: str,
    *,
    private: bool,
) -> tuple[str, str]:
    repo_url = api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    hub_repo_id = repo_url.repo_id
    # create_repo only sets visibility on creation; enforce it on every push so
    # re-pushing an existing repo with a different ``private`` value takes effect.
    api.update_repo_settings(repo_id=hub_repo_id, repo_type="dataset", private=private)
    return str(repo_url), hub_repo_id


def _push_dataset_dict(
    dataset_dict: DatasetDict,
    *,
    repo_id: str,
    commit_message: str,
    config_name: str,
) -> None:
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        data_dir=f"data/{config_name}",
        config_name=config_name,
    )


def _dataset_card_for_configs(config_names: list[str], *, split: str) -> str:
    lines = [
        "---",
        "configs:",
    ]
    for config_name in config_names:
        lines.extend([
            f"- config_name: {config_name}",
            "  data_files:",
            f"  - split: {split}",
            f"    path: data/{config_name}/{split}-*.parquet",
        ])
    lines.extend([
        "---",
        "",
    ])
    return "\n".join(lines)


def _snapshot_allow_patterns(store_names: list[str] | None, *, split: str) -> list[str]:
    if store_names is None:
        return [f"data/*/{split}-*.parquet"]
    return [f"data/{store_name}/{split}-*.parquet" for store_name in store_names]


def _snapshot_store_repo(
    repo_id: str,
    *,
    store_names: list[str] | None,
    split: str,
    revision: str | None,
    token: str | bool | None,
    force_download: bool = False,
) -> Path:
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "allow_patterns": _snapshot_allow_patterns(store_names, split=split),
        "force_download": force_download,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if token is not None:
        kwargs["token"] = token
    return Path(snapshot_download(**kwargs))


def _local_parquet_data_files(
    snapshot_dir: Path,
    *,
    store_names: list[str] | None,
    split: str,
) -> tuple[list[str], dict[str, list[str]]]:
    shard_re = re.compile(rf"^data/([^/]+)/{re.escape(split)}-\d{{5}}-of-\d{{5}}\.parquet$")
    files_by_store: dict[str, list[str]] = {}
    discovered_names: list[str] = []
    seen: set[str] = set()
    for path in sorted(snapshot_dir.glob(f"data/*/{split}-*.parquet")):
        rel_path = path.relative_to(snapshot_dir).as_posix()
        match = shard_re.match(rel_path)
        if match is None:
            continue
        store_name = match.group(1)
        files_by_store.setdefault(store_name, []).append(str(path))
        if store_name not in seen:
            discovered_names.append(store_name)
            seen.add(store_name)

    resolved_store_names = discovered_names if store_names is None else store_names
    missing = [store_name for store_name in resolved_store_names if store_name not in files_by_store]
    if missing:
        raise FileNotFoundError(
            f"No parquet shards found for split {split!r} in dataset snapshot {str(snapshot_dir)!r} "
            f"for store configs: {missing}."
        )
    return resolved_store_names, {
        store_name: files_by_store[store_name]
        for store_name in resolved_store_names
    }


def _write_store_dataset_repo(
    root: Path,
    prepared: list[tuple[str, Dataset]],
    *,
    split: str,
) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / datasets_config.REPOCARD_FILENAME).write_text(
        _dataset_card_for_configs([config_name for config_name, _ in prepared], split=split),
        encoding="utf-8",
    )

    def write_one(config_name: str, ds: Dataset) -> None:
        config_dir = root / "data" / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(config_dir / f"{split}-00000-of-00001.parquet")

    for config_name, ds in prepared:
        write_one(config_name, ds)


def _repo_files_to_clear(api: HfApi, repo_id: str, addition_paths: set[str]) -> list[str]:
    try:
        files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except RepositoryNotFoundError:
        return []
    return [
        path for path in files
        if path not in addition_paths
    ]


def _commit_dataset_repo(
    api: HfApi,
    *,
    repo_id: str,
    folder_path: Path,
    commit_message: str,
    clear: bool,
) -> None:
    additions = [
        path for path in sorted(folder_path.rglob("*"))
        if path.is_file()
    ]
    addition_paths = {path.relative_to(folder_path).as_posix() for path in additions}
    deletions = _repo_files_to_clear(api, repo_id, addition_paths) if clear else []
    operations = [
        CommitOperationDelete(path_in_repo=path)
        for path in deletions
    ] + [
        CommitOperationAdd(
            path_in_repo=path.relative_to(folder_path).as_posix(),
            path_or_fileobj=path,
        )
        for path in additions
    ]

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        operations=operations,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_dataset_repo_id(repo_id: str, token: str | bool | None = None) -> str:
    """Resolve an unscoped dataset repo name under the authenticated Hub user."""
    if "/" in repo_id:
        return repo_id
    user = HfApi().whoami(token=token)["name"]
    return f"{user}/{repo_id}"


def load_stores_from_hub(
    repo_id: str,
    store_names: list[str] | None = None,
    *,
    split: str = "train",
    force_download: bool = False,
    token: str | bool | None = None,
    **kwargs: Any,
) -> list[Datastore]:
    """Load Hub dataset configs into ``Datastore`` objects.

    Short names such as ``"my-dataset"`` resolve under the authenticated Hub
    user before loading the parquet shards written by ``push_stores_to_hub``.
    Matching parquet shards are downloaded as a Hugging Face snapshot first,
    then local exact file paths are passed to one
    ``datasets.load_dataset("parquet", data_files=...)`` call. This avoids one
    Hub tree/glob request per store. When ``store_names`` is omitted, store
    names are discovered from local ``data/{store}/{split}-*.parquet`` paths.
    Each store name is copied onto the returned ``Datastore.name``.
    """
    if store_names is not None:
        if not store_names:
            raise ValueError("load_stores_from_hub requires at least one store name.")
        missing = [i for i, name in enumerate(store_names) if not name]
        if missing:
            raise ValueError(
                "load_stores_from_hub requires every store name to be non-empty. "
                f"Empty store name indices: {missing}."
            )
        if len(set(store_names)) != len(store_names):
            raise ValueError("load_stores_from_hub requires unique store names.")

    resolved_repo_id = _resolve_dataset_repo_id(repo_id, token=token)
    load_kwargs = dict(kwargs)

    revision = load_kwargs.pop("revision", None)
    snapshot_dir = _snapshot_store_repo(
        resolved_repo_id,
        store_names=store_names,
        split=split,
        revision=revision,
        token=token,
        force_download=force_download,
    )
    store_names, data_files = _local_parquet_data_files(
        snapshot_dir,
        store_names=store_names,
        split=split,
    )

    if not store_names:
        raise ValueError(
            f"No parquet store configs found for dataset {resolved_repo_id!r} and split {split!r}."
        )
    missing = [i for i, name in enumerate(store_names) if not name]
    if missing:
        raise ValueError(
            "load_stores_from_hub requires every store name to be non-empty. "
            f"Empty store name indices: {missing}."
        )
    if len(set(store_names)) != len(store_names):
        raise ValueError("load_stores_from_hub requires unique store names.")

    from mouse_core.data.datastore import Datastore as _DS

    loaded = load_dataset(
        "parquet",
        data_files=data_files,
        **load_kwargs,
    )

    stores: list[Datastore] = []
    for store_name in store_names:
        store = _DS(name=store_name)
        store.from_dataset(loaded[store_name])
        stores.append(store)
    return stores


def push_to_hub(
    splits: dict[str, list[Datastore]],
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "New rollout data",
    config_name: str = "default",
    clear: bool = True,
) -> str | None:
    """Combine stores by split and push to the Hugging Face Hub.

    Parameters
    ----------
    splits :
        Mapping of split name → list of ``Datastore`` objects.  All stores
        for the same split are concatenated before pushing.
    repo_id :
        Hub repository ID (``"user/dataset"`` or an unscoped name which is
        assigned to the logged-in user).
    private :
        Repository visibility. Applied on every push: creating it as private,
        and updating an existing repository's visibility to match.
    commit_message :
        Commit message written to the Hub.
    config_name :
        Hugging Face dataset configuration / subset name (also called "config").
    clear :
        Replace the dataset repository contents before uploading. Existing
        remote files are deleted unless the same path is written by this push.
        Defaults to ``True``.
        Use this to organize your data into different "bins" (e.g. different
        collection runs, different environment families, different policies).
        Default is ``"default"``. When loading later use
        ``load_stores_from_hub(repo, [config_name], split=...)``.

    Returns
    -------
    The canonical dataset URL on the Hub (e.g.
    ``"https://huggingface.co/datasets/user/dataset"``), or ``None`` if there
    was nothing to push.

    Examples
    --------
    ::

        from mouse_core.data.hub import push_to_hub

        url = push_to_hub(
            {"train": [train_store], "eval": [eval_store]},
            repo_id="your-org/your-dataset",
        )

        # Using a config/subset as a "bin"
        url = push_to_hub(
            {"train": [store]},
            repo_id="your-org/your-dataset",
            config_name="cartpole_ppo_expert",
        )
    """
    resolved = _build_split_datasets(splits)

    if not resolved:
        print(f"push_to_hub: nothing to push to {repo_id!r}")
        return None

    dataset_dict = DatasetDict(list(resolved.items()))

    try:
        api = HfApi()
        repo_url, hub_repo_id = _create_or_update_dataset_repo(api, repo_id, private=private)
        if clear:
            _wipe_hub_repo_data(api=api, repo_id=hub_repo_id)
        _push_dataset_dict(
            dataset_dict,
            repo_id=hub_repo_id,
            commit_message=commit_message,
            config_name=config_name,
        )
    except HfHubHTTPError as e:
        _raise_for_hub_http_error(e, repo_id)

    parts_str = ", ".join(f"{k}: {len(v)}" for k, v in resolved.items())
    print(f"Pushed to {hub_repo_id} ({parts_str} steps)")
    return str(repo_url)


def push_stores_to_hub(
    stores: list[Datastore],
    repo_id: str,
    *,
    split: str = "train",
    private: bool = False,
    commit_message: str = "New rollout data",
    clear: bool = True,
) -> str | None:
    """Push each ``Datastore`` to the Hub as a separate config/subset.

    Convenience wrapper around :func:`push_to_hub` for the common case where
    every store belongs to the same split but should live in its own config.

    Every store must have a non-empty ``store.name``. Each store is pushed
    under that config name, and can later be loaded with the standard HF loader::

        load_stores_from_hub(repo_id, [store.name], split=split)

    Parameters
    ----------
    stores :
        One or more ``Datastore`` objects to push.
    repo_id :
        Hub repository ID.
    split :
        Split name to push under (default ``"train"``).
    private :
        Repository visibility. Applied on every push: creating it as private,
        and updating an existing repository's visibility to match.
    commit_message :
        Commit message written to the Hub.
    clear :
        Delete all existing parquet shards, dataset info, and the README card
        before uploading. Keeps the dataset card and schema consistent with the
        data being pushed. Defaults to ``True``.
    Returns
    -------
    The canonical dataset URL on the Hub, or ``None`` if there was nothing to push.

    Examples
    --------
    ::

        from mouse_core.data import Datastore, push_stores_to_hub

        url = push_stores_to_hub(
            [Datastore(name="cartpole_v1_ppo_202406")],
            repo_id="your-org/your-dataset",
            split="train",
        )

        # Save each store under its own named subset/bin
        url = push_stores_to_hub(
            [cartpole_store, lunar_store],
            repo_id="your-org/your-dataset",
            split="train",
        )
    """
    missing = [i for i, store in enumerate(stores) if not store.name]
    if missing:
        raise ValueError(
            "push_stores_to_hub requires every store to have a non-empty name. "
            f"Unnamed store indices: {missing}."
        )
    store_names = [store.name for store in stores if store.name]
    if len(set(store_names)) != len(store_names):
        raise ValueError("push_stores_to_hub requires unique store names.")

    _SAFE_CONFIG_NAME = re.compile(r"^[A-Za-z0-9_\-.]+$")
    unsafe = [name for name in store_names if not _SAFE_CONFIG_NAME.match(name)]
    if unsafe:
        bad_chars = sorted({ch for name in unsafe for ch in name if not re.match(r"[A-Za-z0-9_\-.]", ch)})
        raise ValueError(
            f"Store names must contain only letters, digits, hyphens, underscores, and dots "
            f"so they are safe as Hugging Face dataset config names and URLs. "
            f"Offending names: {unsafe}. Illegal characters found: {bad_chars!r}. "
            f"The '#' separator in multi-env slot names (e.g. 'env#0') breaks the "
            f"HuggingFace dataset viewer — use '_' instead."
        )

    prepared: list[tuple[str, Dataset]] = []
    for store, config_name in zip(stores, store_names, strict=True):
        ds = store.to_dataset()
        if len(ds) > 0:
            prepared.append((config_name, ds))

    if not prepared:
        print(f"push_stores_to_hub: nothing to push to {repo_id!r}")
        return None

    try:
        api = HfApi()
        repo_url, hub_repo_id = _create_or_update_dataset_repo(api, repo_id, private=private)
        with tempfile.TemporaryDirectory() as tmp:
            folder_path = Path(tmp)
            _write_store_dataset_repo(
                folder_path,
                prepared,
                split=split,
            )
            _commit_dataset_repo(
                api,
                repo_id=hub_repo_id,
                folder_path=folder_path,
                commit_message=commit_message,
                clear=clear,
            )
    except HfHubHTTPError as e:
        _raise_for_hub_http_error(e, repo_id)

    pushed = {config_name: len(ds) for config_name, ds in prepared}
    parts_str = ", ".join(f"{config}/{split}: {steps}" for config, steps in pushed.items())
    print(f"Pushed to {hub_repo_id} ({parts_str} steps)")
    return repo_url
