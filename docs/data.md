# Data pipeline

MOUSE reads offline RL data from HuggingFace Datasets and delivers batches as `TensorDict[B, S]` tensors. The two core classes are `DatasetStore` and `PrefetchBatchifier`.

---

## TensorDict layout

Every batch delivered by `PrefetchBatchifier` is a `TensorDict` of shape `[B, S]` where `B` is the batch size and `S` is the sequence length. The fields present depend on which observation modalities are configured:

| Field | dtype | Shape | Description |
|---|---|---|---|
| `action` | `int64` | `[B, S]` | Discrete action index |
| `reward` | `float32` | `[B, S]` | Transition reward |
| `xformed_reward` | `float32` | `[B, S]` | Episodically-corrected reward (used by DQN loss when `use_xformed_reward=True`) |
| `done` | `int64` | `[B, S]` | Episode end flag: 0=alive, 1=terminal, 2=truncated |
| `q_star` | `float32` | `[B, S, A]` | Per-action Q* annotation; `-inf` = unavailable/invalid action |
| `time` | `int64` | `[B, S]` | Episode step index; −1 if unavailable |
| `obs_continuous` | `float64` | `[B, S, C]` | Continuous observation vector |
| `obs_discrete` | `int64` | `[B, S]` | Discrete state index |
| `obs_image` | `int64` | `[B, S, P]` | Flattened pixel values in `[0, 255]` |

Optional fields are only present as keys when the source dataset contains the corresponding column — absent fields are simply not in the TensorDict rather than being zero-filled.

### Transition alignment

Each step record at position `t` stores the **observation at step t** alongside the **action, reward, and done that arrived at step t** (i.e. the transition that *produced* `obs_t`). Consequently, the action, reward, and done for the transition *out of* `obs_t` live one step ahead at `t+1`. Both `dqn_loss` and `vec_dqn_loss` account for this offset internally.

---

## DatasetStore (`mouse.data.dataset_store`)

A step buffer backed by a HuggingFace `Dataset` (Arrow format). Designed so the full dataset is **never** loaded into RAM — only the active prefetch queue and any appended rollout steps are held.

### Loading from a HuggingFace Dataset

```python
from datasets import load_dataset
from mouse.data.dataset_store import DatasetStore

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

| Column | Mapped to |
|---|---|
| `action` | `action` |
| `reward` | `reward` |
| `xformed_reward` | `xformed_reward` (required if `use_xformed_reward=True`; omitting it raises a `KeyError`) |
| `done` | `done` |
| `episode_step` | `time` (absent from TensorDict if column is not in dataset) |
| `metadata_q_star` | `q_star` (absent from TensorDict if column is not in dataset or all values are None) |
| `observation` | `obs_continuous` |
| `observation_discrete` | `obs_discrete` |
| `observation_image` | `obs_image` (accepts PIL image, bytes dict, or numpy array) |

### Appending rollout steps

For online data collection, `append` stores single transitions in an in-memory list buffer:

```python
store.append({
    "action": 3,
    "reward": 1.0,
    "done": 0,
    "episode_step": 42,
    "observation": [0.1, 0.2, ..., 0.8],
})
```

### Exporting to a HuggingFace Dataset

```python
ds = store.to_dataset()
ds.push_to_hub("your-org/your-rollout-dataset")
```

---

## Pushing stores to the Hub (`mouse.data.hub`)

`push_to_hub` and `push_stores_to_hub` combine one or more `DatasetStore` objects and push them as a `DatasetDict` to the Hugging Face Hub. Stale parquet shards are wiped before each push so the result is always a clean upload — no leftover split names from previous runs.

**Single split** (most common case):

```python
from mouse.data.hub import push_stores_to_hub

push_stores_to_hub([store], repo_id="your-org/your-dataset", split="train")
```

**Multiple splits** (e.g. train + eval):

```python
from mouse.data.hub import push_to_hub

push_to_hub(
    {"train": [train_store1, train_store2], "eval": [eval_store]},
    repo_id="your-org/your-dataset",
)
```

Columns that are absent from one split but present in another are filled with typed placeholders so the `DatasetDict` schema is consistent across splits.

---

## PrefetchBatchifier (`mouse.data.batch`)

Background-thread batchifier. Worker threads continuously fetch sequences, encode them via `DatasetStore.encode_hf_rows`, and park them in a bounded queue. `next_batch()` pops from the queue — instant once the queue is warm.

The full dataset is never held in memory; only `prefetch × batch_size × sequence_length` encoded steps exist at any time.

```python
from mouse.data.batch import PrefetchBatchifier

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

## TokenAugmenter (`mouse.data.augment`)

Applies training-time augmentations to a batch. All transforms operate on a **clone** of the input; the original is never modified. If no augmentation is enabled the original is returned as-is.

```python
from mouse.data.augment import TokenAugmenter, AugmentTokensConfig, AugmentScalarSpec

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
