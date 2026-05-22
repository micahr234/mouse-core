# Meta-Optimization Using Sequential Experiences

<p align="center"><img src="docs/mouse.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**MOUSE** is a modular PyTorch library for in-context reinforcement learning. It provides embeddings, transformer backbones, output heads, losses, and data utilities for agents that adapt from transition history at inference time, without weight updates.

Install the **`mouse-core`** package on PyPI; import it as **`mouse`** in Python.

## Quick start

```bash
pip install mouse-core
```

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

For a full training loop on synthetic data (no Hub credentials required):

```bash
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
source .venv/bin/activate
python examples/02_train_offline.py
```

## Documentation

Markdown guides live in [`docs/`](docs/README.md):

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md), [data](docs/data.md), [losses](docs/losses.md), [examples](docs/examples.md)
- [API reference](docs/api/model.md) (links to source)
- [Runnable examples](examples/) in this repository

## Requirements

- Python **3.12+**
- PyTorch **2.12+** (CUDA optional)
- Optional: `gymnasium` for environment rollout examples (`pip install "mouse-core[examples]"`)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Development setup:

```bash
source scripts/install.sh
source .venv/bin/activate
pyright src/
pytest
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
