# Getting started

This page is the shortest path from install to a working training step. For full tutorials, see the [home page](index.md) and [examples](examples.md).

---

## 1. Install

```bash
pip install mouse-core
```

Development install (editable, with dev tools and docs):

```bash
git clone https://github.com/micahr234/mouse-core.git
cd mouse-core
source scripts/install.sh
source .venv/bin/activate
```

> **Package name vs import:** PyPI package is `mouse-core`; Python import is `mouse`.

---

## 2. Core concepts

| Concept | What it is |
|---------|------------|
| **Step stream** | `TensorDict[B, S]` — batch of `S` transitions per sequence |
| **DatasetStore** | Buffer backed by Hugging Face `Dataset` or in-memory rollouts |
| **PrefetchBatchifier** | Background loader that yields training batches |
| **Model** | Embedder + backbone + heads; policy emerges from context |
| **Losses** | DQN, VecDQN, SP, SV — compose in your training loop |

In-context RL means the agent conditions on its own history in one forward pass; there are no gradient steps at inference time.

---

## 3. Minimal data + import check

```python
from mouse.data import DatasetStore

store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)
store.append({
    "observation_discrete": [0],
    "action": 0,
    "reward": 0.0,
    "done": 0,
    "episode_step": 0,
})
print(len(store))
```

---

## 4. Runnable examples in the repo

| Script | Purpose |
|--------|---------|
| [`examples/01_collect_dataset.py`](https://github.com/micahr234/mouse-core/blob/main/examples/01_collect_dataset.py) | Collect rollouts with Gymnasium (optional extra) |
| [`examples/02_train_offline.py`](https://github.com/micahr234/mouse-core/blob/main/examples/02_train_offline.py) | Train on in-memory data — no Hub required |
| [`examples/03_inference.py`](https://github.com/micahr234/mouse-core/blob/main/examples/03_inference.py) | Inference loop pattern (set your model id) |

Run from the repository root after `source scripts/install.sh`.

---

## 5. Next steps

- [Model architecture](architecture.md) — embeddings, backbone, heads
- [Data pipeline](data.md) — `DatasetStore`, batching, Hub upload
- [Losses](losses.md) — DQN and auxiliary objectives
- [API reference](api/model.md) — full symbol docs

---

## FAQ

**Why does DQN loss raise “Not enough valid q values”?**  
Sequences need at least two timesteps (`S >= 2`) because actions and rewards are aligned one step ahead of observations.

**Do I need a GPU?**  
No for small smoke tests; training large transformers benefits from CUDA.

**Where are full training configs?**  
Training orchestration lives outside this core library; MOUSE supplies model, data, and loss primitives.
