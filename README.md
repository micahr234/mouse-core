# MOUSE Core

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-core/main/mouse-core.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-core** is the core library for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch stack for in-context reinforcement learning. It provides embeddings, transformer backbones, output heads, objectives, and data utilities for training and deploying agents that adapt from transition history at inference time, without weight updates.

For vector environments and rollout collection, see **[mouse-env](https://github.com/micahr234/mouse-env)**.

---

## Install

```bash
pip install mouse-core
```

For development:

```bash
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```


---

## Core Components

mouse-core gives you the building blocks for in-context RL. You compose three pieces in your own loop:

- **Data** (`mouse_core.data`) stores sequential rows in `Datastore` and batches contiguous windows with `DataLoader`.
- **Models** (`mouse_core.models`) combine an encoder, a backbone (`LlamaBackbone`, `Qwen3Backbone`, or `IdentityBackbone`), and output heads.
- **Objectives** (`mouse_core.objectives`) compute training losses such as DQN, VecDQN, SP, and SV.

Backbone loading has one public path: instantiate the backbone. For example, `LlamaBackbone(pretrained="meta-llama/Llama-3.2-1B", num_layers=2)` reads the pretrained config, loads matching transformer weights, and exposes `backbone.hidden_dim` for the encoder and heads.

---

## Quick Start

A compact training skeleton looks like this:

```python
import torch
from mouse_core.data import DataLoader, Datastore
from mouse_core.objectives import DqnObjective
from mouse_core.models import Model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import StepEmbedder
from mouse_core.models.heads import DiscreteActionValueHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

store = Datastore()
# ... store.append(...) or store.from_dataset(...)
loader = DataLoader(store, sequence_length=16, batch_size=8, num_workers=0)

hidden_dim = 32
encoder = StepEmbedder(
    hidden_dim=hidden_dim,
    modalities=[
        {"field": "action", "type": "discrete", "vocab_size": 4},
        {"field": "reward", "type": "rff"},
        {"field": "done", "type": "discrete", "vocab_size": 3},
    ],
)

backbone = IdentityBackbone(hidden_dim=hidden_dim)
heads = DiscreteActionValueHead(
    in_features=hidden_dim,
    out_features=4,
    hidden_dim=hidden_dim,
    num_layers=1,
)
model = Model(
    encoder=encoder,
    backbone=backbone,
    heads=heads,
).train().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
objective = DqnObjective(gamma_step=0.99, tau=0.005)

for step in range(100):
    batch = loader.next_batch().to(device)
    predictions, objective_data, _ = model(batch)
    loss, metrics = objective(objective_data, predictions)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.polyak_update(action_value_tau=objective.tau)

loader.close()
```

The notebook version with synthetic data and a pretrained Llama backbone is in
[`examples/02_train_offline.ipynb`](examples/02_train_offline.ipynb).

For dataset collection see [`examples/01_collect_dataset.ipynb`](examples/01_collect_dataset.ipynb).
For online training see [`examples/03_train_online.ipynb`](examples/03_train_online.ipynb).
For cached inference see [`examples/04_inference.ipynb`](examples/04_inference.ipynb).

---

## Examples

The notebooks are the primary documentation — each one explains the concepts as it walks through them:

| Notebook | What it covers |
|---|---|
| [`examples/01_collect_dataset.ipynb`](examples/01_collect_dataset.ipynb) | `Datastore`, collecting transitions, pushing to the Hub |
| [`examples/02_train_offline.ipynb`](examples/02_train_offline.ipynb) | `DataLoader`, model architecture, DQN training, all objectives |
| [`examples/03_train_online.ipynb`](examples/03_train_online.ipynb) | Online rollout collection, `Datastore` replay buffers, `DataLoader` sampling |
| [`examples/04_inference.ipynb`](examples/04_inference.ipynb) | KV-cache inference, loading the current checkpoint from the shared Hub model repo |

API reference lives in the Python docstrings (`load_model`, `Datastore`, `DqnObjective`, etc.). See [`CHANGELOG.md`](CHANGELOG.md) for release history.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
