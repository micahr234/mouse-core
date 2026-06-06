# MOUSE Core 🐭

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-core/main/docs/mouse-core.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-core** is the core library for [MOUSE](https://github.com/micahr234/mouse-core), a modular PyTorch stack for in-context reinforcement learning. It provides embeddings, transformer backbones, output heads, losses, and data utilities for training and deploying agents that adapt from transition history at inference time, without weight updates.

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

## Quick start 🚀

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
print(store)
```

See the runnable [`examples/`](examples/) notebooks for end-to-end dataset collection, training, and inference. For offline training on synthetic data (no Hub):

```bash
source .venv/bin/activate
jupyter lab examples/   # then open 02_train_offline.ipynb
```

---

## Documentation 📚

All docs are Markdown in [`docs/`](docs/) (read on GitHub or in the repo):

| Doc | Description |
|-----|-------------|
| [guide.md](docs/guide.md) | Overview, package layout, quick start |
| [architecture.md](docs/architecture.md) | Embedder, backbone, heads |
| [data.md](docs/data.md) | `DatasetStore`, batching, Hub upload |
| [losses.md](docs/losses.md) | DQN, VecDQN, SP, SV |
| [examples.md](docs/examples.md) | Training loops, datasets, inference |

API reference: Python docstrings in `src/mouse_core/` (e.g. `load_model`, `DatasetStore`, `dqn_loss`).

Example notebooks: [`examples/`](examples/) — [`01_collect_dataset.ipynb`](examples/01_collect_dataset.ipynb), [`02_train_offline.ipynb`](examples/02_train_offline.ipynb), [`03_inference.ipynb`](examples/03_inference.ipynb).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
