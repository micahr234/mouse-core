# MOUSE Core

<p align="center"><img src="https://raw.githubusercontent.com/micahr234/mouse-core/main/mouse-core.png" width="400"/></p>

> **Warning:** MOUSE is in early development and is not yet ready for production use. APIs may change without notice.

**mouse-core** is the core library for the Meta-Optimization Using Sequential Experience (MOUSE) learning system, a modular PyTorch stack for in-context reinforcement learning (ICRL). It provides data utilities, embeddings frameworks, transformer backbones, output heads, and objective functions for training and deploying agents that adapt from transition history at inference time, without weight updates.

**[mouse-env](https://github.com/micahr234/mouse-env)** sits alongside mouse-core and handles the environment side: it wraps any Gymnasium env into a reset-free continuing interface, stitching episodes together into uninterrupted trajectories with explicit task boundaries. **mouse-core** is what you use to learn from those trajectories — data utilities, models, and objectives for training and deploying in-context RL agents.

---

## Why MOUSE exists

MOUSE is built around two observations:

1. General learning systems that scale tend to outperform hand-crafted solutions in the long run. This idea is captured in Rich Sutton's essay [The Bitter Lesson](https://web.archive.org/web/20260409023855/https://www.incompleteideas.net/IncIdeas/BitterLesson.html). MOUSE takes that lesson seriously: it meta-learns during training how to solve tasks, so that at deployment time it can adapt to new situations from experience.

2. Learning must not stop at deployment time. The [Big World Hypothesis](http://incompleteideas.net/papers/The_Big_World_Hypothesis.pdf) says that real environments are too vast to model completely ahead of time, so agents cannot be given all the information they will need before they act. MOUSE adapts by conditioning on prior history rather than updating its weights. Because the weights remain fixed at deployment, this avoids plasticity loss, a common continual-learning failure mode where repeated updates gradually reduce an agent's ability to learn.

In the video below, an agent plays FrozenLake on a map it has never seen before, with the map hidden from the agent. Without gradient updates, using only in-context learning, it tries different paths until it finds one that leads directly to the goal. You can train one yourself using the [example notebooks](examples/).

<p align="center"><img src="frozenlake.gif" width="400" alt="MOUSE agent on Procedural FrozenLake"/></p>

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

A compact online training skeleton looks like this:

```python
import torch
from mouse_envs import EnvConfig, make_env

from mouse_core.data import DataLoader, Datastore
from mouse_core.objectives import DqnObjective
from mouse_core.models import Model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import StepEmbedder
from mouse_core.models.heads import DiscreteActionValueHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_env([
    EnvConfig(
        id="Procedural-FrozenLake-v1",
        name="quickstart_frozenlake",
        reset_seed=0,
        episodes_per_task=0,
        kwargs={
            "is_slippery": False,
            "min_width": 4,
            "max_width": 4,
            "min_height": 4,
            "max_height": 4,
            "step_penalty": -0.01,
            "max_episode_steps": 50,
            "map_seed": 0,
        },
    )
])

hidden_dim = 32
encoder = StepEmbedder(
    hidden_dim=hidden_dim,
    modalities=[
        {"field": "action", "type": "discrete", "vocab_size": 4},
        {"field": "observation", "type": "discrete", "vocab_size": 64},
        {"field": "reward", "type": "rff", "in_min": 0.01, "in_max": 100.0},
        {"field": "done", "type": "discrete", "vocab_size": 5},
        {"type": "learnable", "std": 0.02, "tokens": 1},
    ],
    include_type_token=False,
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

stores = [Datastore(name=name) for name in env.names]
contexts = [[] for _ in env.names]
sequence_length = 16

for step in range(100):
    epsilon = max(0.05, 1.0 - step / 100)
    random_inputs = env.sample_random_inputs()
    inputs = []

    for i, context in enumerate(contexts):
        if not context or torch.rand(()) < epsilon:
            inputs.append(random_inputs[i])
        else:
            with torch.no_grad():
                predictions, _, _ = model([context[-sequence_length:]])
                action = model.get_action(predictions, temperature=0.0, num_actions=4)
            inputs.append({"action": action.squeeze().cpu()})

    outputs = env.step(inputs)
    for store, context, inp, out in zip(stores, contexts, inputs, outputs):
        row = {**inp, **out}
        store.append(row)
        context.append(row)

    if min(len(store) for store in stores) < sequence_length:
        continue

    loader = DataLoader(stores, sequence_length=sequence_length, batch_size=1, num_workers=0)
    batch = loader.next_batch()
    loader.close()

    predictions, objective_data, _ = model(batch)
    loss, metrics = objective(objective_data, predictions)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.polyak_update(action_value_tau=objective.tau)

env.close()
```

The tuned online version with live environment collection and a pretrained Qwen backbone is in
[`examples/03_train_online.ipynb`](examples/03_train_online.ipynb).

For dataset collection see [`examples/01_collect_dataset.ipynb`](examples/01_collect_dataset.ipynb).
For offline replay training see [`examples/02_train_offline.ipynb`](examples/02_train_offline.ipynb).
For cached inference see [`examples/04_inference.ipynb`](examples/04_inference.ipynb).

---

## Examples

The notebooks are the primary documentation — each one explains the concepts as it walks through them:

| Notebook | What it covers |
|---|---|
| [`examples/01_collect_dataset.ipynb`](examples/01_collect_dataset.ipynb) | `Datastore`, collecting transitions, pushing to the Hub |
| [`examples/03_train_online.ipynb`](examples/03_train_online.ipynb) | Online rollout collection, in-memory replay, DQN training |
| [`examples/02_train_offline.ipynb`](examples/02_train_offline.ipynb) | Offline replay baseline, model architecture, all objectives |
| [`examples/04_inference.ipynb`](examples/04_inference.ipynb) | KV-cache inference, loading the current checkpoint from the shared Hub model repo |

API reference lives in the Python docstrings (`load_model`, `Datastore`, `DqnObjective`, etc.). See [`CHANGELOG.md`](CHANGELOG.md) for release history.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
