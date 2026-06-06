# Guide

**mouse-core** is the core PyTorch library for [MOUSE](https://github.com/micahr234/mouse-core): in-context reinforcement learning with embeddings, transformer backbones, output heads, losses, and data utilities.

Install: `pip install mouse-core` (import as `mouse_core`).

## Quick start

```python
from mouse_core import load_model
from mouse_core.data import DatasetStore

store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)
store.append({
    "observation": {"discrete": 0},
    "action": {"discrete": 0},
    "reward": 0.0,
    "done": 0,
    "time": 0,
})
print(store)  # DatasetStore(steps=1)
```

See [`examples/01_collect_dataset.ipynb`](../examples/01_collect_dataset.ipynb) for a full Gymnasium-to-`DatasetStore` collection loop.

Runnable training without the Hub:

```bash
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
source .venv/bin/activate
jupyter lab examples/   # then open 02_train_offline.ipynb
```

## Package layout

| Path | Purpose |
|------|---------|
| `src/mouse_core/models/base.py` | `Model`, `load_model`, Hub I/O |
| `src/mouse_core/models/embedding/` | `StepEmbedder`, per-modality token embedders |
| `src/mouse_core/models/backbone/` | LLaMA, Qwen3, or no backbone (`ModelNone`) |
| `src/mouse_core/models/heads/` | DQN, VecDQN, SP/SwiGLU heads |
| `src/mouse_core/data/` | `DatasetStore`, `PrefetchBatchifier`, Hub upload, augmentation |
| `src/mouse_core/losses/` | DQN, VecDQN, SP, SV losses |

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

Runnable notebooks live in [`examples/`](../examples/): [`01_collect_dataset.ipynb`](../examples/01_collect_dataset.ipynb), [`02_train_offline.ipynb`](../examples/02_train_offline.ipynb), [`03_inference.ipynb`](../examples/03_inference.ipynb).

## Collecting data

For vector env rollouts and the step record contract (`group_id`, typed-dict `action`/`observation`, `reward_episodic`, `q_star`), use **[mouse-env](https://github.com/micahr234/mouse-env)** (see its [docs/guide.md](https://github.com/micahr234/mouse-env/blob/main/docs/guide.md)). This repo focuses on ingesting datasets and training models — see [data.md](data.md) for the column → `TensorDict` mapping.
