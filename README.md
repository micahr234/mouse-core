# MOUSE Core 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-core/main/docs/mouse-core.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-core** is the core library for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch stack for in-context reinforcement learning. It provides embeddings, transformer backbones, output heads, objectives, and data utilities for training and deploying agents that adapt from transition history at inference time, without weight updates.

For vector environments and rollout collection, see **[mouse-env](https://github.com/micahr234/mouse-env)**.

---

## Install 📦

```bash
pip install mouse-core
```

For development:

```bash
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```

Import as **`mouse_core`** (PyPI package name is **`mouse-core`**).

---

## Core components 🧩

mouse-core gives you the building blocks for in-context RL rather than a monolithic trainer — you compose them in your own loop:

- **Data** (`mouse_core.data`) — `DatasetStore` is a sequential container whose native row format is the mouse-env contract (`MouseEnvRecord`). You load/append mouse-env style records (nested observation/action + rollout scalars). `PrefetchBatchifier` (given model sizes) turns contiguous slices into `TensorDict[B, S]` batches. Push uses plain HF `Dataset`/`DatasetDict` with `config_name` for bins.
- **Models** (`mouse_core.models`) — a MOUSE `Model` is a pipeline of per-modality step embedders, a transformer backbone (LLaMA, Qwen3, or none), and output heads (DQN, VecDQN, SwiGLU). `load_model` / `save_model` / `push_model_to_hub` handle Hub I/O, and `init_from_pretrained_backbone` bootstraps from a pretrained language model.
- **Objectives** (`mouse_core.objectives`) — composable objective functions with configs (`dqn_objective`, `vec_dqn_objective`, `sp_objective`, `sv_objective`) that you weight and sum in your training loop.

Training combines all three: batches from a `DatasetStore` flow through the model, objectives are computed on the outputs, and the result is pushed to the Hub. At inference time you only need the model — it attends over the transition history and adapts in context, with no weight updates.

---

## Quick start 🚀

A compact training loop using all three components — **Data**, **Models**, and **Objectives**:

```python
import torch
from mouse_core.data import DatasetStore, PrefetchBatchifier
from mouse_core.objectives import DqnObjectiveConfig, dqn_objective
from mouse_core.models.backbone.none import ModelNone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data: build or load a DatasetStore, then batchify
store = DatasetStore()
# ... store.append(...) or store.from_dataset(...)
# (pass target tensor sizes to PrefetchBatchifier, not the store)
bf = PrefetchBatchifier(store, sequence_length=16, batch_size=8, num_workers=0, max_action_dim=4)

# Model (see the notebook for complete embedding/head kwargs)
model = ModelNone(
    hidden_dim=32,
    backbone_kwargs={},
    embedding_kwargs=dict(max_num_actions=4, include_action_token=True, include_reward_token=True, include_done_token=True),
    dqn_head_kwargs=dict(num_layers=1, hidden_dim=32),
    action_head="dqn",
).train().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
cfg = DqnObjectiveConfig(weight=1.0, gamma=0.99, tau=0.005)

# Training: Data → Model → Objective
for step in range(100):
    batch = bf.next_batch().to(device)
    out, _ = model(batch)
    loss, metrics = dqn_objective(batch, out, cfg)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.polyak_update(dqn_tau=cfg.tau)

bf.close()
```

The complete self-contained version (synthetic data, no Hub) is in
[`examples/02_train_offline.ipynb`](examples/02_train_offline.ipynb).

For dataset collection see [`examples/01_collect_dataset.ipynb`](examples/01_collect_dataset.ipynb).
For cached inference see [`examples/03_inference.ipynb`](examples/03_inference.ipynb).

---

## Documentation 📚

| Doc | Description |
|-----|-------------|
| [models.md](docs/models.md) | Embedder, backbone, heads, forward pass |
| [dataset.md](docs/dataset.md) | `DatasetStore`, batching, Hub upload |
| [objectives.md](docs/objectives.md) | DQN, VecDQN, SP, SV objectives |

API reference lives in the Python docstrings (`load_model`, `DatasetStore`, `dqn_objective`, …).

Runnable notebooks (end-to-end collection, training, inference) are in [`examples/`](examples/).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
