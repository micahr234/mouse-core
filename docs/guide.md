# Guide

**mouse-core** is the core PyTorch library for [MOUSE](https://github.com/micahr234/mouse-core): in-context reinforcement learning with embeddings, transformer backbones, output heads, losses, and data utilities.

Install: `pip install mouse-core` (import as `mouse`).

## Quick start

```python
from mouse import load_model
from mouse.data import DatasetStore

store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)
store.append({
    "observation_discrete": [0],
    "action": 0,
    "reward": 0.0,
    "done": 0,
    "episode_step": 0,
})
print(store)  # DatasetStore(steps=1)
```

Runnable training without the Hub:

```bash
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
source .venv/bin/activate
python examples/02_train_offline.py
```

## Package layout

| Path | Purpose |
|------|---------|
| `src/models/base.py` | `Model`, `load_model`, Hub I/O |
| `src/models/embedding/` | `StepEmbedder`, per-modality token embedders |
| `src/models/backbone/` | LLaMA, Qwen3, or no backbone (`ModelNone`) |
| `src/models/heads/` | DQN, VecDQN, SP/SwiGLU heads |
| `src/data/` | `DatasetStore`, `PrefetchBatchifier`, Hub upload, augmentation |
| `src/losses/` | DQN, VecDQN, SP, SV losses |

API details live in **docstrings** (`load_model`, `DatasetStore`, `dqn_loss`, …).

## In-context meta-RL (short)

The policy is not only fixed weights: the model attends over a history of `(observation, action, reward, done, time)` steps and adapts within a single forward pass. See [architecture.md](architecture.md) for the embedder → backbone → heads pipeline.

## Documentation map

| Doc | Description |
|-----|-------------|
| [architecture.md](architecture.md) | Embedder, backbone, heads, forward pass |
| [data.md](data.md) | `DatasetStore`, batching, Hub datasets |
| [losses.md](losses.md) | DQN, VecDQN, SP, SV |
| [examples.md](examples.md) | Training loops, dataset collection, inference |
| [mouse_env.md](mouse_env.md) | Rollout schema with [mouse-env](https://github.com/micahr234/mouse-env) |

## Collecting data

For vector env rollouts and the v1 step contract (`env_id`, nested dicts), use **[mouse-env](https://github.com/micahr234/mouse-env)**. This repo focuses on ingesting datasets and training models — see [data.md](data.md) for column → `TensorDict` mapping.
