# mouse-env integration

[mouse-env](https://github.com/micahr234/mouse-env) builds vector Gymnasium environments and rollout metadata for this library. The cross-repo contract is documented there:

| mouse-env doc | Topic |
|---------------|--------|
| [rollout_contract.md](https://github.com/micahr234/mouse-env/blob/main/docs/rollout_contract.md) | v1 step record (`env_id`, `episode_index`, `step_index`, action/observation/reward dicts, `done`) |
| [mouse_core_alignment.md](https://github.com/micahr234/mouse-env/blob/main/docs/mouse_core_alignment.md) | Recommended `DatasetStore` / column changes in mouse-core |

## What mouse-core does today

[`DatasetStore`](../src/data/dataset_store.py) maps Hugging Face **columns** to a `TensorDict` batch. See [data.md](data.md) for the current column table (`action`, `reward`, `done`, `episode_step` → `time`, observation fields, optional `metadata_q_star`).

Legacy keys still used in many datasets (`episode_step`, `xformed_reward`, flat observations) remain supported.

## Target alignment

New datasets collected from mouse-env should follow **rollout_contract.md**. Planned improvements on the core side (see mouse-env’s alignment doc):

1. Document the contract in [data.md](data.md) (done — link above).
2. Accept `env_id`, `episode_index`, and `step_index` columns where present.
3. Map `reward["episodic"]` when using nested reward dicts (fallback: `xformed_reward`).

Until those land, flatten contract rows to the existing columns when calling `DatasetStore.append()` or when writing Parquet for Hub upload.
