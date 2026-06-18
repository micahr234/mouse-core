# Data pipeline

MOUSE reads offline RL data from HuggingFace Datasets and delivers batches as `TensorDict[B, S]` tensors. The two core classes are `DatasetStore` and `PrefetchBatchifier`.

The dataset column schema follows the [mouse-env rollout contract](https://github.com/micahr234/mouse-env/blob/main/docs/guide.md) — `action` and `observation` are typed-dict (struct) columns, with renamed scalar fields (`time`, `reward_episodic`, `q_star`).

---

## TensorDict layout

Every batch delivered by `PrefetchBatchifier` is a `TensorDict` of shape `[B, S]` where `B` is the batch size and `S` is the sequence length. The fields present depend on which observation modalities are configured:

| Field | dtype | Shape | Description |
|---|---|---|---|
| `action` | `int64` | `[B, S]` | Discrete action index (0 placeholder for continuous-only steps) |
| `action_continuous` | `float32` | `[B, S, Ac]` | Continuous action vector; present when `max_action_continuous_dim > 0` |
| `reward` | `float32` | `[B, S]` | Transition reward |
| `xformed_reward` | `float32` | `[B, S]` | Episodically-corrected reward (used by DQN loss when `use_xformed_reward=True`) |
| `done` | `int64` | `[B, S]` | Episode end flag: 0=alive, 1=terminal, 2=truncated |
| `q_star` | `float32` | `[B, S, A]` | Per-action Q* annotation; `-inf` = unavailable/invalid action |
| `time` | `int64` | `[B, S]` | Episode step index; −1 if unavailable |
| `obs_continuous` | `float64` | `[B, S, C]` | Continuous observation vector |
| `obs_discrete` | `int64` | `[B, S]` | Discrete state index |
| `obs_image` | `int64` | `[B, S, P]` | Flattened pixel values in `[0, 255]` |

Optional fields are only present as keys when the source dataset contains the corresponding column — absent fields are simply not in the TensorDict rather than being zero-filled. When a struct column carries a modality for only *some* steps (e.g. a dataset mixing discrete- and continuous-action environments), the missing steps are zero-filled per row (and the discrete `action` index falls back to a `0` placeholder).

### Transition alignment

Each step record at position `t` stores the **observation at step t** alongside the **action, reward, and done that arrived at step t** (i.e. the transition that *produced* `obs_t`). Consequently, the action, reward, and done for the transition *out of* `obs_t` live one step ahead at `t+1`. Both `dqn_loss` and `vec_dqn_loss` account for this offset internally.

---

## DatasetStore (`mouse_core.data.dataset_store`)

A step buffer backed by a HuggingFace `Dataset` (Arrow format). Designed so the full dataset is **never** loaded into RAM — only the active prefetch queue and any appended rollout steps are held.

### Loading from a HuggingFace Dataset

```python
from datasets import load_dataset
from mouse_core.data.dataset_store import DatasetStore

store = DatasetStore(max_action_dim=18, max_obs_continuous_dim=8)

# Single split
store.from_dataset(load_dataset("your-org/your-dataset", split="train"))

# All splits from a DatasetDict (concatenated in order)
store.from_dataset(load_dataset("your-org/your-dataset"))

# Selected splits by exact name
store.from_dataset(load_dataset("your-org/your-dataset"), splits=["train", "test"])

# Glob patterns — all splits matching "train_*" or "eval_*"
store.from_dataset(load_dataset("your-org/your-dataset"), split_pattern=["train_*", "eval_*"])

# Single pattern
store.from_dataset(load_dataset("your-org/your-dataset"), split_pattern="test_*")
```

Calling `from_dataset` more than once concatenates onto what is already loaded. Each call is O(1) — no data is copied.

### Expected HuggingFace Dataset columns

Columns follow the [mouse-env rollout contract](https://github.com/micahr234/mouse-env/blob/main/docs/guide.md). `action` and `observation` are always **dict** (struct) columns, and any sub-key name is valid. The encoder maps the modality sub-keys the model understands (`discrete` / `continuous` / `image`) onto flat tensors; every other sub-key is preserved verbatim in the stored dataset (and survives the `to_dataset` / `from_dataset` round-trip) but is not encoded into the training `TensorDict`.

| Column | Sub-key | Mapped to |
|---|---|---|
| `action` | `discrete` | `action` |
| `action` | `continuous` | `action_continuous` (present when `max_action_continuous_dim > 0`) |
| `reward` | — | `reward` |
| `reward_episodic` | — | `xformed_reward` (required if `use_xformed_reward=True`; omitting it raises a `KeyError`) |
| `done` | — | `done` |
| `time` | — | `time` (absent from TensorDict if column is not in dataset) |
| `q_star` | — | `q_star` (absent from TensorDict if column is not in dataset or all values are None) |
| `observation` | `continuous` | `obs_continuous` |
| `observation` | `discrete` | `obs_discrete` |
| `observation` | `image` | `obs_image` (native-shape array, flattened to pixels) |

Extra mouse-env columns (`group_id`, `episode_index`, `ns_params`) are carried in the dataset for filtering/analysis but are not encoded into the training `TensorDict`.

### Appending rollout steps

A runnable Gymnasium collection loop is in [`examples/01_collect_dataset.ipynb`](../examples/01_collect_dataset.ipynb). For online data collection, `append` stores single transitions in an in-memory list buffer:

```python
store.append({
    "action": {"discrete": 3},
    "reward": 1.0,
    "reward_episodic": 0.04,
    "done": 0,
    "time": 42,
    "observation": {"continuous": [0.1, 0.2, ..., 0.8]},
})
```

### Exporting to a HuggingFace Dataset

```python
ds = store.to_dataset()
ds.push_to_hub("your-org/your-rollout-dataset")
```

---

## Pushing stores to the Hub (`mouse_core.data.hub`)

`push_to_hub` and `push_stores_to_hub` combine one or more `DatasetStore` objects and push them as a `DatasetDict` to the Hugging Face Hub. Stale parquet shards are wiped before each push so the result is always a clean upload — no leftover split names from previous runs.

**Single split** (most common case):

```python
from mouse_core.data.hub import push_stores_to_hub

push_stores_to_hub([store], repo_id="your-org/your-dataset", split="train")
```

**Multiple splits** (e.g. train + eval):

```python
from mouse_core.data.hub import push_to_hub

push_to_hub(
    {"train": [train_store1, train_store2], "eval": [eval_store]},
    repo_id="your-org/your-dataset",
)
```

Columns that are absent from one split but present in another are filled with typed placeholders so the `DatasetDict` schema is consistent across splits.

Pass `private=True`/`private=False` to control repository visibility. It is applied on every push, so re-pushing an existing repo with a different value updates its visibility (not only at creation time).

---

## PrefetchBatchifier (`mouse_core.data.batch`)

Background-thread batchifier. Worker threads continuously fetch sequences, encode them via `DatasetStore.encode_hf_rows`, and park them in a bounded queue. `next_batch()` pops from the queue — instant once the queue is warm.

The full dataset is never held in memory; only `prefetch × batch_size × sequence_length` encoded steps exist at any time.

```python
from mouse_core.data.batch import PrefetchBatchifier

bf = PrefetchBatchifier(
    store,
    sequence_length=64,
    batch_size=32,
    sampling="random",   # "random" | "last" | "sequential" | "batch"
    prefetch=4,
    num_workers=2,
    pin_memory=True,     # enable for CUDA training
)
step_stream = bf.next_batch()   # TensorDict[32, 64]
bf.close()
```

### Sampling modes

| Mode | Behaviour |
|---|---|
| `random` | Uniformly random start indices each time |
| `last` | The final `B` non-overlapping windows (useful for recency-biased replay) |
| `sequential` | Epoch-order windows, reshuffled at each epoch boundary |
| `batch` | Like sequential but always starts from the next un-seen window |

### Synchronous mode

Pass `num_workers=0` to disable background threads. `next_batch()` blocks on the calling thread. Useful for debugging or environments where threading is undesirable.

```python
bf = PrefetchBatchifier(store, sequence_length=64, batch_size=32, num_workers=0)
step_stream = bf.next_batch()
bf.close()
```

---

## TokenAugmenter (`mouse_core.data.augment`)

Applies training-time augmentations to a batch. All transforms operate on a **clone** of the input; the original is never modified. If no augmentation is enabled the original is returned as-is.

```python
from mouse_core.data.augment import TokenAugmenter, AugmentTokensConfig, AugmentScalarSpec

cfg = AugmentTokensConfig(
    scale_reward=AugmentScalarSpec(mean=1.0, low=0.5, high=2.0),   # uniform [0.5, 2.0)
    permute_action="both",     # permute action IDs and q_star columns jointly
    mask_prob=...,             # per-field Bernoulli masking
)

augmenter = TokenAugmenter(
    augment=cfg,
    max_num_actions=18,
    max_num_obs_discrete=0,
    device=device,
)

# Call once per batch to sample permutations and scalars for this batch
augmenter.update_augmentations(step_stream)

# Call once per model head (or clone) to apply the same fixed permutations
augmented = augmenter(step_stream)
```

### Available augmentations

| Config field | Effect |
|---|---|
| `scale_reward` / `shift_reward` | Multiply/add a scalar drawn per batch |
| `scale_obs` / `shift_obs` | Scale/shift continuous observation values |
| `scale_obs_image` / `shift_obs_image` | Scale/shift pixel values (clamped 0–255) |
| `permute_action` | Randomly remap action IDs; `"input"` permutes `action` only, `"target"` permutes `q_star` only, `"both"` does both jointly |
| `permute_done` | Randomly swap done `{0,1,2}` flags per sequence |
| `permute_obs_discrete` | Randomly remap discrete state IDs per sequence |
| `mask_prob` | Per-field Bernoulli masking (MLM-style); masked positions are zeroed |

Permutations are drawn once per batch via `update_augmentations` and are consistent across all steps in a sequence row. `mask_prob` is sampled fresh on each `__call__`.
