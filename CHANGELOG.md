# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `push_stores_to_hub` and `push_to_hub` were writing every config's parquet data to the same path (`data/train-*.parquet`) because `data_dir="data"` was passed explicitly to `DatasetDict.push_to_hub`, overriding the per-config subdirectory behaviour introduced in datasets v5. Each config now writes to its own `data/{config_name}/` directory, so multiple subsets are no longer silently overwritten by the last one pushed.
- `push_stores_to_hub` and `push_to_hub` `clear` parameter now correctly defaults to `True` (the docstring-stated default), ensuring stale parquet shards are wiped before each fresh push.

### Changed
- Example notebooks migrated from `FrozenLake-v1` to `Procedural-FrozenLake-v1` with a step penalty (`step_penalty=-0.04`) so the expert Q-table properly distinguishes shorter from longer paths.
- Example notebook 02 trains with `DqnObjective` using separate episode and task discount factors: `gamma_terminal=gamma_truncated=1.0` (bootstrap across in-task episode boundaries) and `gamma_task_terminal=gamma_task_truncated=0.0` (no bootstrap across task boundaries). q_star labels from the curriculum are used to guide data collection but not as a direct training signal.
- Notebook 03 now runs a 20-episode success-rate evaluation before the frame-capture animation, and resets the KV-cache at each episode boundary.
- `DqnObjective` now accepts `gamma_task_terminal` and `gamma_task_truncated` parameters (both default to `0.0`) to set the bootstrap discount for `done==3` (task terminated) and `done==4` (task truncated) transitions separately from within-task episode boundaries.

### Added
- Notebook 01 now collects expert demonstrations using `q_star_source={"provider": "metadata_q_star"}` and a curriculum policy (`oracle_prob` 0 → 0.5 → 1.0 across three collection phases).


- Objectives are now classes (`DqnObjective`, `VecDqnObjective`, `SpObjective`, `SvObjective`) instantiated with their hyperparameters and called with `(objective_data, predictions)`. The separate `*ObjectiveConfig` dataclasses and `*_objective` functions are removed.
- `StepEmbedder`: removed `num_compute_tokens` parameter. Use a `{"name": "scratch", "embed": "learnable", "tokens": N}` modality placed last in the list to get the same dedicated scratch/prediction token.
- `Encoder.forward` now returns a third value `step_token_indices [B, S]` — the absolute flat-sequence position of the prediction token for each step. `Encoder.pool_step_reprs` now takes this tensor instead of `batch_size`, enabling future variable-token-count modalities without any change to the pooling logic.
- `EnvConfig` now requires `episodes_per_task` (number of episodes per MOUSE task) instead of `max_episode_steps`. Examples updated accordingly.
- `env.step()` and `env.sample_random_inputs()` now return a flat `list[dict]` — one dict per slot across all configs — instead of a nested `list[(list[dict], _)]` per config.
- `done` field now carries 5 codes (`0`=running, `1`=episode terminated, `2`=episode truncated, `3`=task terminated, `4`=task truncated) instead of 3; `StepEmbedder` `done` modality updated to `vocab_size=5`.
- `DataLoader` no longer accepts a `sampling` parameter; random windowed sampling is always used.
- Moved `docs/` reference documentation into the example notebooks; each notebook now explains the relevant concepts inline as it walks through them.
- Moved project logo (`mouse-core.png`) to the repo root.
- Updated README to link to examples instead of standalone doc files.
