# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
