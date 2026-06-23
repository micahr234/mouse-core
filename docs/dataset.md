# Dataset pipeline

`Datastore` is a **sequential** container for step data.

It stores rows (plain Python dicts) in the exact order they were added or loaded. The backing storage is a Hugging Face `Dataset` (Arrow). There is no required schema or contract — each row contains whatever your collection script, rollout, or external dataset produced. Extra columns and nested structures are preserved as-is.

All I/O uses ordinary Hugging Face tools (`load_dataset`, `push_to_hub`, `DatasetDict`, etc.). The only mouse-core-specific step is turning slices of rows into model batches.

When training you ask for contiguous sequences of steps (e.g. 64 steps). Because the data is stored sequentially these are just slices. The tensors in the resulting batch are produced by `DataLoader` (or `encode_hf_rows`) according to the sizes you pass it and whatever fields happen to be present in the rows.

The store and the data on disk/Hub are just the raw rows. The model (via its embedders / tokenizer / encoder) decides how to interpret them.

`DataLoader` repeatedly takes slices and yields ready `TensorDict[B, S]` batches. Everything else (loading, saving, pushing) is standard Hugging Face Dataset machinery.

The two important classes are `Datastore` and `DataLoader`.

---

## Sequential storage

The data is stored in the exact order it was appended or loaded. It is just a flat sequence of rows. No built-in episode indexing, no automatic train/test split, no reordering.

Because it is sequential, the two operations that must be fast are cheap:

- Append new single records quickly (while an environment is running).
- Sample large continuous batches quickly (e.g. grab 64 steps at a time for a training batch).

Concretely:
- `append(row)` is a plain Python list append.
- A sequence of length `S` is the slice `dataset[start:start+S]`.

The entire history does not need to live in RAM. Only the active prefetch window of encoded batches is materialized while training.

## Batches come from contiguous slices

A training sequence of length `S` is just:

```python
rows = dataset[start : start + S]
```

`DataLoader` repeatedly takes slices like this (workers when useful), projects the conventional fields present in those rows into tensors (vector widths taken from the max present in the slice), and yields `TensorDict[B, S]` batches.

### What ends up in the TensorDict

`DataLoader` (and `encode_hf_rows`) look for a few conventional fields (`action`, `reward`, `done`, `observation.*`, `q_star`, ...) and stack them. For vector-valued sub-fields the width `D` of the emitted tensor is the largest length that appears in the rows of *that particular slice*; shorter or absent entries are zero-padded (or -inf-padded for `q_star`) to that D within the slice.

The resulting shapes are therefore driven by your data. Your model declares capacities (`max_num_obs_continuous`, `max_num_actions`, ...) when you build the embedders; those embedders internally adapt incoming vectors (pad or truncate) to their configured size.

Other content in the rows is ignored for the batch but remains in the stored data. The store and loader are generic; interpretation lives in the model.

### Note on ordering (for objective authors)

Because the store is strictly sequential, step `t` contains whatever your environment wrote for that step. If your collection convention is "observation at t plus the transition that produced it", then the "next" quantities live at `t+1`. Objectives that need the next-step target simply index accordingly. The store itself does not enforce or know about this convention.

---

## Datastore (`mouse_core.data.datastore`)

A sequential store backed by a Hugging Face `Dataset`. The full history stays in Arrow form; only the small prefetch window of encoded batches is materialized during training.

### Loading data (use the real `load_dataset`)

`Datastore.from_dataset` is intentionally thin. All the power for selecting *what* to load lives in the standard Hugging Face `load_dataset`:

- configs / subsets (your "bins")
- splits
- file patterns / globs
- unions of splits, streaming, etc.

Do your selection first, then hand the resulting `Dataset` (or a prepared slice) to the store.

```python
from datasets import load_dataset
from mouse_core.data.datastore import Datastore

store = Datastore()

# Specific config (subset/bin) + split
ds = load_dataset("your-org/your-dataset", "cartpole_ppo_expert", split="train")
store.from_dataset(ds)

# Another bin
ds2 = load_dataset("your-org/your-dataset", "lunar_random", split="train")
store.from_dataset(ds2)

# You can combine however you like before ingesting
# (or keep separate stores if you prefer logical separation)
```

`from_dataset` mainly exists so the *store* can:
- mix loaded data with later `append()` calls (the live collection buffer), and
- participate in `DataLoader` + our encoding to `TensorDict`.

It does not try to replicate or wrap the rich selection features of `datasets.load_dataset`. Use the real thing.

If you pass a `DatasetDict` it will concatenate the splits inside it (the store is a flat sequence). Prefer selecting exactly what you want via `load_dataset(..., config, split=...)` first.

Call `from_dataset` as many times as you like; each extends the sequential history with zero-copy reference where possible.

## Structuring your dataset repository on the Hub (configs + splits in YAML)

Hugging Face recommends declaring your splits and subsets (configs) explicitly in the dataset card using a YAML front-matter block. This is the generic, first-class way to organize data into "bins" and named splits without custom code.

See the official guide: https://huggingface.co/docs/datasets/repository_structure#define-your-splits-and-subsets-in-yaml

Example for a mouse-core dataset with two experiment bins (configs), each with train/eval splits:

```yaml
---
configs:
- config_name: cartpole_ppo_expert
  data_files:
  - split: train
    path: "data/cartpole_ppo_expert/train/*.parquet"
  - split: eval
    path: "data/cartpole_ppo_expert/eval/*.parquet"
- config_name: lunar_random
  data_files:
  - split: train
    path: "data/lunar_random/train/*.parquet"
---
```

You load exactly as you would with any HF dataset:

```python
from datasets import load_dataset

ds = load_dataset("your-org/your-dataset", "cartpole_ppo_expert", split="train")
store.from_dataset(ds)
```

Our `push_to_hub` / `push_stores_to_hub` (and raw `ds.push_to_hub(..., config_name=...)`) write parquet under `data/<config>/...` (or the layout produced by the HF pusher). The resulting files are compatible with the YAML `data_files` declarations above.

Pushing data through our helpers **deletes the previous README** (and stale shards/infos) before the push. This ensures that `DatasetDict.push_to_hub` writes a fresh dataset card whose `dataset_info:` header (features, splits, sizes, etc.) accurately describes the data you are uploading right now. This is especially important for a brand new dataset or when the schema or split structure changes.

If you also maintain a declarative `configs:` / `data_files:` block (see the HF repository structure guide), you can add or restore it after the push by editing the README on the Hub. Subsequent data pushes will again regenerate the card from the current data.

You can use globs, lists, or the automatic patterns (train/ directories, *-00000-of-00003 naming, etc.). `config_name` is the "subset/bin"; split names inside follow the usual rules.

This keeps mouse-core's data layer as a thin user of the standard HF repository structure.

### Rows are opaque to the store

The store does not look inside rows or impose structure. A row is whatever dictionary you append or load. The store only keeps them in order and gives fast contiguous slices.

The only place mouse-core looks at row contents is inside the row encoder used by `DataLoader` (see "What ends up in the TensorDict" above).

### Exporting to a HuggingFace Dataset (raw)

You can always drop down to the plain HF object:

```python
ds = store.to_dataset()

# Push under a specific config/subset ("bin") and split
ds.push_to_hub(
    "your-org/your-rollout-dataset",
    config_name="my_experiment",
    split="train",
)
```

This is completely standard `datasets.Dataset.push_to_hub` — no wrapper. Use it when you want full control (or when you only have one store).

---

## Pushing stores to the Hub (`mouse_core.data.hub`)

`push_to_hub` and `push_stores_to_hub` combine `Datastore`s and delegate to ordinary `DatasetDict.push_to_hub(..., config_name=...)`. 

What gets written to the Hub are the exact rows you stored. There is no mouse-core-specific wrapper format — downstream users (or you) can consume the dataset with plain `datasets` or even non-Python tools.

Before the push we delete previous parquet shards, `dataset_infos.json`, **and the README**. This guarantees that the dataset card written by the push has a `dataset_info:` header that matches the freshly uploaded data (critical when uploading a brand new dataset or when columns/splits/configs change).

### Configs (subsets) as bins + splits

Hugging Face datasets support two levels:

- **config** (also called subset or configuration name) — this is a great way to organize different "bins" of data inside one repo (different runs, different envs, different policies, etc.).
- **split** — inside a config (train / test / eval / whatever you like, as long as it matches the HF split name rules).

You keep using normal split names (`train`, `eval`, `my_eval`, ...), and additionally choose a `config_name` for the bin.

Loading the data later is pure HF:

```python
ds = load_dataset("your-org/your-dataset", "cartpole_ppo_expert", split="train")
store.from_dataset(ds)
```

Pushing:

```python
from mouse_core.data.hub import push_stores_to_hub, push_to_hub

# One bin, one split (very common)
push_stores_to_hub(
    [store],
    repo_id="your-org/your-dataset",
    split="train",
    config_name="cartpole_ppo_expert",
)

# Multiple splits inside one bin
push_to_hub(
    {"train": [train_store], "eval": [eval_store]},
    repo_id="your-org/your-dataset",
    config_name="lunar_v2_random",
)
```

When you later call the raw `to_dataset().push_to_hub(...)` you can also pass `config_name` directly to the HF method.

Split names must still satisfy HF's `^\w+(\.\w+)*$` rule (we forward the error). Config names have similar restrictions.

Columns that differ across the things you push in one call are aligned with placeholders (same as before).

`private=...` is applied on every push.

---

## DataLoader (`mouse_core.data`)

`DataLoader` turns slices of your `Datastore` into ready `TensorDict[B, S]` batches. It samples contiguous windows from the store and runs a row encoder that extracts conventional fields and stacks them (vector widths are the max present in each slice). Background workers can be used so `next_batch()` is usually non-blocking.

```python
from mouse_core.data import DataLoader

loader = DataLoader(
    store,
    sequence_length=64,
    batch_size=32,
    sampling="random",
    prefetch=4,
    num_workers=2,
    pin_memory=True,
)
step_stream = loader.next_batch()   # TensorDict[32, 64]
loader.close()
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
loader = DataLoader(store, sequence_length=64, batch_size=32, num_workers=0)
step_stream = loader.next_batch()
loader.close()
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
