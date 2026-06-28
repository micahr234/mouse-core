# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DataLoader.refresh()` re-snapshots underlying stores and drains any prefetched batches, so online replay can pick up newly appended rows without rebuilding the loader.
- `DataLoader` accepts an optional `seed` argument (default `None`) that controls its internal NumPy RNG; when set, worker `i` uses `seed + i` for deterministic multi-worker sampling.
- `StepEmbedder` accepts a new `type_embedding_std` parameter to control the initialisation std of the type embedding table independently from the content embedding `std`. **Required when `include_type_token=True`**; raises `ValueError` if omitted to prevent accidental type-to-content signal imbalance.

### Fixed
- `examples/03_train_online.ipynb` collect/train loop no longer consumes all env steps in a single rollout; `ENV_STEPS_PER_CYCLE` caps each cycle so DQN updates run repeatedly, and epsilon is recomputed per env transition instead of once per cycle.

### Changed
- Example notebooks share `GRADIENT_STEPS` as the total optimizer-step budget (`TRAIN_STEPS` renamed in `02_train_offline.ipynb`). Online adds `ENV_STEPS_PER_CYCLE`, `STEPS_PER_ENV`, and `GRADIENT_STEPS_PER_CYCLE` to interleave live env interaction with the same number of gradient updates as offline training. `EXPLORATION_FULL_AT_STEP` renamed to `EXPLORATION_ENDS`.
- README quick start now points to the example notebooks instead of an inline training skeleton.
- `DataLoader` with `pack=True` allows empty stores at construction and on `refresh()`; `next_batch()` raises if every store is still empty.
- Example notebooks updated for `mouse-env` 0.5.0: `make_env` now creates a `SingleEnv`, `make_group_env` creates a `GroupEnv`, and `sample_random_input()` replaces `sample_random_inputs()`.
- `examples/03_train_online.ipynb` collection keeps one KV cache per env visit (`STEPS_PER_ENV` transitions), growing it incrementally and discarding it when moving to the next env; the context deque may slide independently.
- `examples/03_train_online.ipynb` collection now processes one env at a time with a single KV cache (discarded after each env's `STEPS_PER_ENV`) instead of windowing envs into `ENV_BATCH_SIZE` parallel caches; step counters now count env transitions directly.
- `examples/03_train_online.ipynb` reorganizes collection and training into separate documented sections with `collect_round` and `train_round` helpers; episode stats print after each collect round instead of a separate evaluation phase.
- `SequenceAugmenter` renamed to `Augmenter` in `mouse_core.data`.
- `load_model` now defaults to `force_download=True`, always pulling the latest checkpoint from the Hub instead of serving a cached copy.
- `examples/03_train_online.ipynb` now ramps epsilon-greedy exploration from `0.0` to `1.0` by `EXPLORATION_ENDS`, then keeps full exploration for the remainder.
- `StepEmbedder` now uses `modality_fusion="sum"` or `"concat"` instead of the old `concat_modalities` boolean; the example notebooks use explicit sum fusion.
- `examples/02_train_offline.ipynb` and `examples/03_train_online.ipynb` now include final kernel shutdown cells to release all notebook-owned GPU memory after training.
- `DqnObjective`: replaced `use_episodic_reward` with an explicit `reward_key: str = "reward"` parameter; added `done_key: str = "done"` parameter (parallel to `action_key`).
- `DqnObjective`: removed `normalize_reward_mean`, `normalize_reward_std`, `normalize_reward_eps`, `normalize_reward_std_target`, `reward_scale`, and `reward_shift`.
- `DqnObjective`: discount is now computed via a vectorized gamma lookup (`gammas[done_next]`) instead of four boolean masks; the reset-frame bootstrap workaround is removed.
- `DqnObjective`: all S-1 consecutive pairs within a sampled sequence are now trained; the done-based valid-transition mask that excluded transitions from terminal states is removed.
- `StepEmbedder` now raises `ValueError` when `include_type_token=True` and `type_embedding_std` is not provided, forcing intentional configuration of the type embedding scale.
- `examples/02_train_offline.ipynb` and `examples/03_train_online.ipynb` now set `include_type_token=False` on `StepEmbedder`, keeping the summed token purely content-driven.
- `load_stores_from_hub` now snapshots matching dataset parquet shards once and loads local exact file paths in one `datasets.load_dataset("parquet", data_files=...)` call, avoiding one Hub tree/glob request per store.
- `examples/04_inference.ipynb` now separates evaluation env count from replay output count, allowing 100-env evaluation while only embedding the first 10 replay videos.
- FrozenLake training examples now use the full pretrained `Qwen/Qwen3-0.6B` backbone instead of truncating `Qwen3Backbone` with `num_layers=2`.
- `examples/03_train_online.ipynb` now uses the tuned offline baseline's training horizon, context length, batch size, learnable-token scale, and action-value head scale while keeping data collection online.

### Fixed
- `DataLoader` now snapshots both loaded source rows and newly appended rows from each `Datastore`, so replay built from Hub data plus live rollout does not silently ignore recent experience.
- `StepEmbedder` now raises when a required modality is missing from only some rows in a batch instead of silently filling those rows with default values.
- `DqnObjective` and `VecDqnObjective` now skip invalid reset-frame transitions after boundary rows and use the following reset row for boundary bootstrapping when available.
- `SpObjective` now supports `-inf` padded invalid actions in `info_q_star` while still rejecting NaN and `+inf` targets.
- `examples/04_inference.ipynb` now renders only the replay envs created with `render_mode="rgb_array"`, avoiding Gymnasium render-mode warnings from non-video evaluation envs.
- `examples/02_train_offline.ipynb` now passes AdamW beta coefficients with PyTorch's supported `betas` keyword.

## [0.4.0] - 2026-06-25

### Added
- `push_stores_to_hub` now batches named store configs into a single Hub commit.
- `mouse_core.__version__` now exposes the installed `mouse-core` package version from package metadata.
- Added `examples/03_train_online.ipynb`, showing live environment training with epsilon-greedy action selection, `Datastore` replay buffers, `DataLoader` sampling over appended experience, DQN updates, and task-boundary policy-context resets.
- `Augmenter` now provides the public training-time augmentation path for raw `DataLoader` batches, configured by modality specs that use `field` plus an augmentation `type`. A modality `field` may name multiple fields to share the same sampled permutation and per-step mask decision, linear numeric fields use `scale_in_low`, `scale_out_low`, `scale_in_high`, and `scale_out_high` endpoint pairs to derive the reward/value affine transform, and `DataLoader(..., augmenter=augment)` runs augmentation inside the sampling path so worker threads can prepare augmented batches.
- `StepEmbedder` modality specs may now use a tuple/list `field` to encode multiple raw fields with the same modality settings. Learnable scratch modalities no longer need a fake data field; use `{"type": "learnable", "tokens": N}`.
- Notebook 01 now collects expert demonstrations using `q_star_source={"provider": "env_q_star"}` and a curriculum policy (`oracle_prob` 0 → 0.5 → 1.0 across three collection phases).
- `Encoder.forward` now returns a third value `step_token_indices [B, S]` — the absolute flat-sequence position of the prediction token for each step. `Encoder.pool_step_reprs` now takes this tensor instead of `batch_size`, enabling future variable-token-count modalities without any change to the pooling logic.

### Changed
- `push_stores_to_hub(clear=True)` now replaces the full dataset repository contents instead of only clearing dataset shard/card files.
- `DqnObjective` now uses `gamma_step` for running transitions instead of `gamma`; the example DQN configs bootstrap through all done codes by setting episode and task boundary discounts to `0.99`.
- FrozenLake examples now use deterministic 50-step episodes inside one infinite task (`episodes_per_task=0`, `is_slippery=False`, `max_episode_steps=50`) across collection, online training, and inference.
- Example notebooks now flow from dataset collection, to offline training, to online training, to inference. Offline and online training both push to the shared `mouse-example-model` Hub repo, and `examples/04_inference.ipynb` loads whichever checkpoint is currently published there.
- Modality specs now use the public key `type` instead of `embed` in both `StepEmbedder` and `Augmenter` configs.
- Expert Q-values from `mouse_envs` are now read from the environment-produced key `info_q_star`, following the `mouse-env` output contract that forwards `info["q_star"]` as `info_q_star`. `SpObjective`, `SvObjective`, and the collection notebook (`01_collect_dataset`) now use that single field.
- Example notebooks migrated from `FrozenLake-v1` to `Procedural-FrozenLake-v1` with a step penalty (`step_penalty=-0.04`) so the expert Q-table properly distinguishes shorter from longer paths.
- Example notebook 02 trains with `DqnObjective` using separate step, episode-boundary, and task-boundary discount factors. q_star labels from the curriculum are used to guide data collection but not as a direct training signal.
- Notebook 04 now runs a 20-episode success-rate evaluation before the frame-capture animation, and carries the KV-cache across episode boundaries.
- `DqnObjective` now accepts `gamma_task_terminal` and `gamma_task_truncated` parameters to set the bootstrap discount for `done==3` (task terminated) and `done==4` (task truncated) transitions separately from within-task episode boundaries.
- Objectives are now classes (`DqnObjective`, `VecDqnObjective`, `SpObjective`, `SvObjective`) instantiated with their hyperparameters and called with `(objective_data, predictions)`. The separate `*ObjectiveConfig` dataclasses and `*_objective` functions are removed.
- `EnvConfig` now requires `episodes_per_task` (number of episodes per MOUSE task) instead of `max_episode_steps`. Examples updated accordingly.
- `env.step()` and `env.sample_random_inputs()` now return a flat `list[dict]` — one dict per slot across all configs — instead of a nested `list[(list[dict], _)]` per config.
- `done` field now carries 5 codes (`0`=running, `1`=episode terminated, `2`=episode truncated, `3`=task terminated, `4`=task truncated) instead of 3; `StepEmbedder` `done` modality updated to `vocab_size=5`.
- Moved `docs/` reference documentation into the example notebooks; each notebook now explains the relevant concepts inline as it walks through them.
- Moved project logo (`mouse-core.png`) to the repo root.
- Updated README to link to examples instead of standalone doc files.

### Removed
- Removed the old tensor-level `TokenAugmenter` public API in favor of the raw sequence augmentation module.
- `StepEmbedder` removed `num_compute_tokens`; use a `{"type": "learnable", "tokens": N}` modality placed last in the list to get the same dedicated scratch/prediction token.
- `DataLoader` no longer accepts a `sampling` parameter; random windowed sampling is always used.

### Fixed
- `Model.forward` no longer calls `TensorDict.to()` when encoder outputs are already on the target device, avoiding a spurious CUDA availability probe during CPU-only runs.
- `push_stores_to_hub` and `push_to_hub` were writing every config's parquet data to the same path (`data/train-*.parquet`) because `data_dir="data"` was passed explicitly to `DatasetDict.push_to_hub`, overriding the per-config subdirectory behaviour introduced in datasets v5. Each config now writes to its own `data/{config_name}/` directory, so multiple subsets are no longer silently overwritten by the last one pushed.
- `push_stores_to_hub` and `push_to_hub` `clear` parameter now correctly defaults to `True` (the docstring-stated default), ensuring stale parquet shards are wiped before each fresh push.
