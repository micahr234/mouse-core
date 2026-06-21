"""Hugging Face Hub utilities for pushing DatasetStore data.

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
``push_to_hub(splits, repo_id, config_name=...)``
    Push splits under a named config (subset/bin).

``push_stores_to_hub(stores, repo_id, split=..., config_name=...)``
    Single-split convenience.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets
from datasets import config as datasets_config
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import CommitOperationDelete

if TYPE_CHECKING:
    from mouse_core.data.dataset_store import DatasetStore


# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------

_HUB_PARQUET_SHARD_RE = re.compile(r"^data/([^/]+)-\d{5}-of-\d{5}\.parquet$")


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def push_to_hub(
    splits: dict[str, list[DatasetStore]],
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "New rollout data",
    config_name: str = "default",
) -> str | None:
    """Combine stores by split and push to the Hugging Face Hub.

    Parameters
    ----------
    splits :
        Mapping of split name → list of ``DatasetStore`` objects.  All stores
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
        Use this to organize your data into different "bins" (e.g. different
        collection runs, different environment families, different policies).
        Default is ``"default"``. When loading later use
        ``load_dataset(repo, config_name, split=...)``.

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
    from mouse_core.data.dataset_store import DatasetStore as _DS

    resolved: dict[str, Dataset] = {}
    for split_name, stores in splits.items():
        parts = [_DS.merge_stores_to_dataset(stores)]
        combined = concatenate_datasets([p for p in parts if len(p) > 0])
        if len(combined) > 0:
            resolved[split_name] = combined

    if not resolved:
        print(f"push_to_hub: nothing to push to {repo_id!r}")
        return None

    resolved = _align_splits(resolved)
    dataset_dict = DatasetDict(list(resolved.items()))

    try:
        api = HfApi()
        repo_url = api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        hub_repo_id = repo_url.repo_id
        # create_repo only sets visibility on creation; enforce it on every push so
        # re-pushing an existing repo with a different ``private`` value takes effect.
        api.update_repo_settings(repo_id=hub_repo_id, repo_type="dataset", private=private)
        _wipe_hub_repo_data(api=api, repo_id=hub_repo_id)
        dataset_dict.push_to_hub(
            repo_id=hub_repo_id,
            commit_message=commit_message,
            data_dir="data",
            config_name=config_name,
        )
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code == 403:
            raise RuntimeError(
                "Hugging Face returned 403 when creating or pushing the dataset. "
                f"repo_id={repo_id!r}: you need write access to that namespace "
                "(use a short name to push under your logged-in user, e.g. "
                "'my_dataset', or 'youruser/my_dataset' / an org you belong to). "
                "Check your token at https://huggingface.co/settings/tokens"
            ) from e
        raise

    parts_str = ", ".join(f"{k}: {len(v)}" for k, v in resolved.items())
    print(f"Pushed to {hub_repo_id} ({parts_str} steps)")
    return str(repo_url)


def push_stores_to_hub(
    stores: list[DatasetStore],
    repo_id: str,
    *,
    split: str = "train",
    private: bool = False,
    commit_message: str = "New rollout data",
    config_name: str = "default",
) -> str | None:
    """Push a list of ``DatasetStore`` objects to the Hub as a single split (inside a config/subset).

    Convenience wrapper around :func:`push_to_hub` for the common case where
    all stores belong to one split.

    Use ``config_name`` to put the data under a named "bin"/configuration
    (e.g. a particular experiment, policy, or environment family). You can
    later load it with the standard HF loader::

        load_dataset(repo_id, config_name, split=split)

    Parameters
    ----------
    stores :
        One or more ``DatasetStore`` objects to concatenate and push.
    repo_id :
        Hub repository ID.
    split :
        Split name to push under (default ``"train"``).
    private :
        Repository visibility. Applied on every push: creating it as private,
        and updating an existing repository's visibility to match.
    commit_message :
        Commit message written to the Hub.
    config_name :
        Configuration / subset name (default ``"default"``). This is the HF
        "bin" for your data.

    Returns
    -------
    The canonical dataset URL on the Hub, or ``None`` if there was nothing to push.

    Examples
    --------
    ::

        from mouse_core.data.hub import push_stores_to_hub

        url = push_stores_to_hub([store], repo_id="your-org/your-dataset", split="train")

        # Save this collection under a named subset/bin
        url = push_stores_to_hub(
            [store],
            repo_id="your-org/your-dataset",
            split="train",
            config_name="cartpole_v1_ppo_202406",
        )
    """
    return push_to_hub(
        {split: stores},
        repo_id=repo_id,
        private=private,
        commit_message=commit_message,
        config_name=config_name,
    )
