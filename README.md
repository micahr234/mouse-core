# Meta-Optimization Using Sequential Experiences

![MOUSE](mouse.png)

**MOUSE** is a PyTorch library for context-conditioned sequence modeling in reinforcement learning. It ingests a history of environment transitions as a `TensorDict[B, S]` batch, embeds each step into a token sequence, runs a transformer backbone (Llama, Qwen3, or identity), and produces outputs from one or more action heads — supervised policy, DQN Q-values, vector DQN, or supervised Q* regression.

📖 **[Full documentation](https://micahr234.github.io/mouse-core/)**

---

## Install

```bash
pip install "git+https://github.com/micahr234/mouse-core.git"
```

Or for development:

```bash
source scripts/install.sh
```

This creates a Python 3.12 virtual environment via `uv` and installs the package in editable mode with dev dependencies (`pyright`, `pytest`).

For image observation support install the optional extra:

```bash
pip install "mouse[image]"
```

---

## Quick start

```python
import torch
from tensordict import TensorDict
from mouse.models.base import load_model

# Load from a local checkpoint or HuggingFace Hub
model = load_model("your-org/your-model")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

B, S = 4, 32  # batch of 4 sequences, 32 steps each
step_stream = TensorDict(
    {
        "action":         torch.zeros(B, S, dtype=torch.int64),
        "reward":         torch.zeros(B, S, dtype=torch.float32),
        "done":           torch.zeros(B, S, dtype=torch.int64),
        "time":           torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
        "obs_continuous": torch.zeros(B, S, 8, dtype=torch.float32),
    },
    batch_size=(B, S),
)

with torch.no_grad():
    out, cache = model(step_stream.to(device))

# out is TensorDict[B, S]; pick an action from the last step
action = model.get_action(out, temperature=0.0)  # [B]
```

---

## Online rollouts with KV-cache

Feed one step at a time and carry the cache forward for efficient autoregressive inference:

```python
cache = None
step_idx = 0

while not done:
    step_stream = TensorDict(
        {
            "action":         last_action.unsqueeze(1),      # [B, 1]
            "reward":         last_reward.unsqueeze(1),      # [B, 1]
            "done":           last_done.unsqueeze(1),        # [B, 1]
            "time":           torch.full((B, 1), step_idx, dtype=torch.long),
            "obs_continuous": obs.unsqueeze(1).float(),      # [B, 1, obs_dim]
        },
        batch_size=(B, 1),
    )

    with torch.no_grad():
        out, cache = model(step_stream.to(device), cache=cache, use_cache=True)

    action = model.get_action(out, temperature=0.0)  # [B]
    step_idx += 1
```

> **Cache limit.** The model was trained with sequences of a fixed length. Accuracy degrades when the cache exceeds roughly 2× that length — reset it (`cache = None`) before reaching that limit.

---

## Package layout

```
src/
├── models/
│   ├── base.py              # Model base class, load_model(), MODEL_CARD_TEMPLATE
│   ├── backbone/
│   │   ├── configs.py       # LlamaBackboneConfig, Qwen3BackboneConfig
│   │   ├── llama.py         # ModelLlama (HF Hub mixin)
│   │   ├── qwen3.py         # ModelQwen3 (HF Hub mixin)
│   │   └── none.py          # ModelNone (identity backbone)
│   ├── embedding/
│   │   ├── embedding.py     # StepEmbedder, TokenType, per-modality embedders
│   │   ├── encoding.py      # RandomFourierFeatures, NormalizedPixel
│   │   └── linear.py        # ScaledEmbedding, ScaledLinear, ScaledPosLinear
│   └── heads/
│       ├── swiglu.py        # SwiGLU, SwiGLUHead
│       ├── dqn.py           # DQNHead (online + target, Polyak)
│       └── vec_dqn.py       # VecDQNHead, rope_rotate, vec_dqn_scores
├── losses/
│   ├── dqn.py               # DqnLossConfig, dqn_loss
│   ├── vec_dqn.py           # VecDqnLossConfig, vec_dqn_loss
│   ├── sp.py                # SpLossConfig, sp_loss (CE / soft-CE / JS / KL)
│   └── sv.py                # SvLossConfig, sv_loss (MSE / MAE on q_star)
└── data/
    ├── dataset_store.py     # DatasetStore (HF Dataset-backed step buffer)
    ├── batch.py             # PrefetchBatchifier, to_tensor_dict
    └── augment.py           # TokenAugmenter, AugmentTokensConfig
```

---

## Model heads

| Head | Output shape | Use |
|---|---|---|
| `sp` | `[B, S, A]` | Supervised policy logits distilled from `q_star` |
| `dqn` / `dqn_target` | `[B, S, A]` | Q-values; DQN TD loss |
| `vec_dqn` / `vec_dqn_target` | `[B, S, A, D]` | Action vectors; angular scoring via `vec_dqn_scores` |
| `sv` | `[B, S, A]` | Direct `q_star` regression |

Each head is optional — omit or set `num_layers=0` to disable it.

---

## Losses

All losses take a `step_stream: TensorDict[B, S]` and a tensor from the model output. Each returns `(scalar_loss, metrics_dict)`.

| Loss | Function | Config |
|---|---|---|
| DQN (TD) | `dqn_loss(step_stream, q, q_target, cfg)` | `DqnLossConfig` |
| Vector DQN | `vec_dqn_loss(step_stream, online_vecs, target_vecs, cfg)` | `VecDqnLossConfig` |
| Supervised policy | `sp_loss(step_stream, logits, cfg)` | `SpLossConfig` |
| Supervised value | `sv_loss(step_stream, logits, cfg)` | `SvLossConfig` |

```python
from mouse.losses.dqn import DqnLossConfig, dqn_loss

cfg = DqnLossConfig(gamma=0.99, tau=0.005)
loss, metrics = dqn_loss(step_stream, out["dqn"], out["dqn_target"], cfg)
loss.backward()
```

---

## Data pipeline

```python
from datasets import load_dataset
from mouse.data.dataset_store import DatasetStore
from mouse.data.batch import PrefetchBatchifier

store = DatasetStore(max_action_dim=18, max_obs_continuous_dim=8)
store.from_dataset(load_dataset("your-org/your-dataset", split="train"))

with PrefetchBatchifier(store, sequence_length=64, batch_size=32) as bf:
    for _ in range(num_steps):
        step_stream = bf.next_batch()   # TensorDict[32, 64], instant once warm
        out, _ = model(step_stream.to(device))
        loss, _ = dqn_loss(step_stream, out["dqn"], out["dqn_target"], cfg)
        loss.backward()
        optimizer.step()
        model.polyak_update(dqn_tau=cfg.tau)
```

---

## Documentation

| Doc | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Model internals: embedding, backbone, heads |
| [Data pipeline](docs/data.md) | DatasetStore, PrefetchBatchifier, TensorDict layout |
| [Losses](docs/losses.md) | DQN, VecDQN, SP, SV loss functions |
| [Training guide](docs/training.md) | End-to-end offline training loop |

---

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.12
- `transformers` ≥ 5.8 (for Llama / Qwen3 backbones)
- `tensordict` ≥ 0.12
- `datasets` ≥ 4.8
- `huggingface_hub` ≥ 0.30
- `pillow` ≥ 11.0 (optional, for image observations)

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
