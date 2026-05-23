# Meta-Optimization Using Sequential Experiences

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-core/main/docs/mouse-core.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for use. APIs will change without notice.

**mouse-core** is the core library for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch stack for in-context reinforcement learning. It provides embeddings, transformer backbones, output heads, losses, and data utilities for training and deploying agents that adapt from transition history at inference time, without weight updates.

For vector environments and rollout collection, see **[mouse-env](https://github.com/micahr234/mouse-env)**.

## Install

```bash
pip install mouse-core
```

Development:

```bash
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
```

Import as **`mouse`** (PyPI package name is **`mouse-core`**).

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
print(store)
```

Offline training on synthetic data (no Hub):

```bash
source .venv/bin/activate
python examples/02_train_offline.py
```

## Documentation

All docs are Markdown in [`docs/`](docs/) (read on GitHub or in the repo):

| Doc | Description |
|-----|-------------|
| [guide.md](docs/guide.md) | Overview, package layout, quick start |
| [architecture.md](docs/architecture.md) | Embedder, backbone, heads |
| [data.md](docs/data.md) | `DatasetStore`, batching, Hub upload |
| [losses.md](docs/losses.md) | DQN, VecDQN, SP, SV |
| [examples.md](docs/examples.md) | Training loops, datasets, inference |
| [mouse_env.md](docs/mouse_env.md) | **mouse-core ↔ mouse-env** rollout schema |

API reference: Python docstrings in `src/` (e.g. `load_model`, `DatasetStore`, `dqn_loss`).

Repo scripts: [`examples/`](examples/) (`01_collect_dataset.py`, `02_train_offline.py`, `03_inference.py`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
