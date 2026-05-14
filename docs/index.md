# MOUSE

![MOUSE](mouse.png)

**Meta-Optimization Using Sequential Experiences** — a PyTorch library for in-context meta-reinforcement learning.

---

## What is in-context meta-RL?

In standard reinforcement learning, an agent's policy is encoded in its weights — learning happens through gradient updates over many environment interactions. In **meta-reinforcement learning**, the goal is to produce an agent that can adapt its behavior rapidly to new tasks, ideally without any weight updates at all.

**In-context meta-RL** takes this further: instead of adapting through gradient steps, the agent adapts purely through its *context* — the sequence of transitions it has observed so far in the current episode or trial. The policy is not a fixed mapping from state to action; it is an emergent property of the transformer's attention over the history.

At each step the model sees the full sequence of past `(observation, action, reward, done)` tuples. By attending over this history the model can recognize what has and hasn't worked, infer the reward structure of the current task, and adjust its behavior accordingly — all within a single forward pass.

## How MOUSE implements this

MOUSE treats the transition history as a token sequence and runs a standard causal transformer over it:

```
# o=obs  a=action  r=reward  d=done
[o₀, a₀, r₀, d₀]  [o₁, a₁, r₁, d₁]  …  [oₜ, aₜ, rₜ, dₜ]
         ↓ StepEmbedder (each modality → T tokens, summed)
    [B, S × T, D]  flat token sequence
         ↓ Transformer backbone (Llama / Qwen3 / identity)
    [B, S × T, D]  contextualised hidden states
         ↓ last-token pool per step
    [B, S, D]  one vector per step
         ↓ output heads
    actions  /  Q-values
```

Each step is encoded into `T` embedding tokens — one contribution per modality (observation, action, reward, done, time), all summed. The transformer attends causally over the full `S × T` token sequence, so every step representation is conditioned on everything that came before it. The last token of each step group is pooled to produce a single `[D]`-vector per step, which is then passed to whichever output heads are enabled.

Four head types are supported:

| Head | Output | Use |
|---|---|---|
| `sp` | logits `[B, S, A]` | Supervised policy — distilled from Q* annotations |
| `sv` | logits `[B, S, A]` | Supervised Q* regression |
| `dqn` | logits `[B, S, A]` | Trained with offline TD |
| `vec_dqn` | vectors `[B, S, A, D]` | Geometric Q — reward encoded as angular rotation |

---

## Install

**From GitHub:**

```bash
pip install "git+https://github.com/micahr234/mouse-core.git"
```

**For development** (creates a Python 3.12 venv via `uv`, installs all extras):

```bash
source scripts/install.sh
```

---

## Example

```python
import torch
from tensordict import TensorDict
from mouse.models.base import load_model

model = load_model("your-org/your-model").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

B, S = 4, 32
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
    out, _ = model(step_stream.to(device))

action = model.get_action(out, temperature=0.0)  # [B]
```

---

## Guides

- [Data](data.md) — downloading and processing offline RL datasets
- [Model Architecture](architecture.md) — design choices: embedding, backbone, heads
- [Losses](losses.md) — DQN, VecDQN, SP, and SV loss functions
- [Examples](examples.md) — training loop patterns

---

The full API reference is available [here](api/model.md).
