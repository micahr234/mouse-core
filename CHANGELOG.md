# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `examples/03_train_online.ipynb`, showing live environment training with epsilon-greedy action selection, `Datastore` replay buffers, `DataLoader` sampling over appended experience, DQN updates, and task-boundary policy-context resets.
- `SequenceAugmenter` now provides the public training-time augmentation path for raw `DataLoader` batches, configured by modality specs that use `field` plus an augmentation `type`. A modality `field` may name multiple fields to share the same sampled permutation and per-step mask decision, linear numeric fields use `scale_in_low`, `scale_out_low`, `scale_in_high`, and `scale_out_high` endpoint pairs to derive the reward/value affine transform, and `DataLoader(..., augmenter=augment)` runs augmentation inside the sampling path so worker threads can prepare augmented batches.
- `StepEmbedder` modality specs may now use a tuple/list `field` to encode multiple raw fields with the same modality settings. Learnable scratch modalities no longer need a fake data field; use `{"type": "learnable", "tokens": N}`.

### Changed
- `DqnObjective` now uses `gamma_step` for running transitions instead of `gamma`; the example DQN configs bootstrap through all done codes by setting episode and task boundary discounts to `0.99`.
- FrozenLake examples now use deterministic 50-step episodes inside one infinite task (`episodes_per_task=0`, `is_slippery=False`, `max_episode_steps=50`) across collection, online training, and inference.
- Example notebooks now flow from dataset collection, to offline training, to online training, to inference. Offline and online training both push to the shared `mouse-example-model` Hub repo, and `examples/04_inference.ipynb` loads whichever checkpoint is currently published there.
- Modality specs now use the public key `type` instead of `embed` in both `StepEmbedder` and `SequenceAugmenter` configs.
- Expert Q-values from `mouse_envs` are now read from the environment-produced key `info_env_q_star`, following the `mouse-env` output contract that forwards `info["env_q_star"]` as `info_env_q_star`. `SpObjective`, `SvObjective`, the collection notebook (`01_collect_dataset`), and the offline training notebook (`02_train_offline`) now use that single field.

### Removed
- Removed the old tensor-level `TokenAugmenter` public API in favor of the raw sequence augmentation module.

### Fixed
- `Model.forward` no longer calls `TensorDict.to()` when encoder outputs are already on the target device, avoiding a spurious CUDA availability probe during CPU-only runs.
- `push_stores_to_hub` and `push_to_hub` were writing every config's parquet data to the same path (`data/train-*.parquet`) because `data_dir="data"` was passed explicitly to `DatasetDict.push_to_hub`, overriding the per-config subdirectory behaviour introduced in datasets v5. Each config now writes to its own `data/{config_name}/` directory, so multiple subsets are no longer silently overwritten by the last one pushed.
- `push_stores_to_hub` and `push_to_hub` `clear` parameter now correctly defaults to `True` (the docstring-stated default), ensuring stale parquet shards are wiped before each fresh push.

### Changed
- Example notebooks migrated from `FrozenLake-v1` to `Procedural-FrozenLake-v1` with a step penalty (`step_penalty=-0.04`) so the expert Q-table properly distinguishes shorter from longer paths.
- Example notebook 02 trains with `DqnObjective` using separate step, episode-boundary, and task-boundary discount factors. q_star labels from the curriculum are used to guide data collection but not as a direct training signal.
- Notebook 04 now runs a 20-episode success-rate evaluation before the frame-capture animation, and carries the KV-cache across episode boundaries.
- `DqnObjective` now accepts `gamma_task_terminal` and `gamma_task_truncated` parameters to set the bootstrap discount for `done==3` (task terminated) and `done==4` (task truncated) transitions separately from within-task episode boundaries.

### Added
- Notebook 01 now collects expert demonstrations using `q_star_source={"provider": "env_q_star"}` and a curriculum policy (`oracle_prob` 0 → 0.5 → 1.0 across three collection phases).


- Objectives are now classes (`DqnObjective`, `VecDqnObjective`, `SpObjective`, `SvObjective`) instantiated with their hyperparameters and called with `(objective_data, predictions)`. The separate `*ObjectiveConfig` dataclasses and `*_objective` functions are removed.
- `StepEmbedder`: removed `num_compute_tokens` parameter. Use a `{"type": "learnable", "tokens": N}` modality placed last in the list to get the same dedicated scratch/prediction token.
- `Encoder.forward` now returns a third value `step_token_indices [B, S]` — the absolute flat-sequence position of the prediction token for each step. `Encoder.pool_step_reprs` now takes this tensor instead of `batch_size`, enabling future variable-token-count modalities without any change to the pooling logic.
- `EnvConfig` now requires `episodes_per_task` (number of episodes per MOUSE task) instead of `max_episode_steps`. Examples updated accordingly.
- `env.step()` and `env.sample_random_inputs()` now return a flat `list[dict]` — one dict per slot across all configs — instead of a nested `list[(list[dict], _)]` per config.
- `done` field now carries 5 codes (`0`=running, `1`=episode terminated, `2`=episode truncated, `3`=task terminated, `4`=task truncated) instead of 3; `StepEmbedder` `done` modality updated to `vocab_size=5`.
- `DataLoader` no longer accepts a `sampling` parameter; random windowed sampling is always used.
- Moved `docs/` reference documentation into the example notebooks; each notebook now explains the relevant concepts inline as it walks through them.
- Moved project logo (`mouse-core.png`) to the repo root.
- Updated README to link to examples instead of standalone doc files.
