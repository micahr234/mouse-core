# Architecture

MOUSE is organized as a small library with three main Python packages under `src/`:

| Package | Role |
|---------|------|
| `mouse.models` | `StepEmbedder`, transformer backbones, output heads, `load_model` / Hub I/O |
| `mouse.data` | `DatasetStore`, `PrefetchBatchifier`, augmentation, Hub dataset upload |
| `mouse.losses` | DQN, VecDQN, SP, SV loss functions and configs |

Each environment step is embedded into tokens, processed by a causal backbone, pooled to one vector per step, then passed to heads (policy logits, Q-values, etc.). The policy is not fixed in weights alone; it emerges from attention over the step history.

For design rationale and diagrams, see [docs/architecture.md](docs/architecture.md). API pages under [docs/api/](docs/api/) link to source files.
