# Meta-Optimization Using Sequential Experiences

<p align="center"><img src="mouse.png" alt="MOUSE" width="400"/></p>

**MOUSE** is a modular PyTorch library for in-context reinforcement learning. It provides the building blocks — embeddings, transformer backbones, output heads, losses, and data utilities — for training and deploying agents that adapt their behaviour by attending over their own transition history, with no weight updates at inference time.

Use MOUSE when you want to build or experiment with in-context RL agents: whether training from an offline dataset, collecting online rollouts, or combining both.

---

## What is in-context meta-RL?

In standard reinforcement learning, an agent's policy is encoded in its weights — learning happens through gradient updates over many environment interactions. In **meta-reinforcement learning**, the goal is to produce an agent that can adapt its behavior rapidly to new tasks.

**In-context meta-RL** takes this further: instead of adapting through gradient steps, the agent adapts purely through its *context* — the sequence of transitions it has observed so far in the current episode or trial. The policy is not a fixed mapping from state to action; it is an emergent property of the transformer's attention over the history.

At each step the model sees the full sequence of past `(observation, action, reward, done, time)` tuples. By attending over this history the model can recognize what has and hasn't worked, infer the reward structure of the current task, and adjust its behavior accordingly — all within a single forward pass.

## How MOUSE implements this

Each environment step is embedded into a block of tokens, one per modality (observation, action, reward, done, time). All step blocks are concatenated into a flat causal sequence, passed through a transformer backbone, and pooled back to one vector per step. Those vectors are then fed to output heads that produce action logits or Q-values.

The policy is never stored in fixed weights — it emerges from the transformer attending over the full history within a single forward pass. See the [Architecture](architecture.md) page for details.

---

## Install

```bash
pip install mouse-core
```

---

## Creating and Uploading a Dataset

```python
import gymnasium as gym
import torch
from mouse.data.dataset_store import DatasetStore
from mouse.data.hub import push_stores_to_hub

env = gym.make("FrozenLake-v1", is_slippery=True)

# max_obs_discrete_dim=1 for a single integer observation (grid cell index)
store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)

for episode in range(500):
    obs, _ = env.reset()
    action    = 0
    reward    = 0.0
    done_flag = 0

    for step_idx in range(200):
        store.append({
            "observation_discrete": [obs],
            "action":               action,
            "reward":               reward,
            "done":                 done_flag,
            "episode_step":         step_idx,
        })

        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done_flag = 1 if terminated else (2 if truncated else 0)

        if terminated or truncated:
            # Append the terminal transition before moving to the next episode
            store.append({
                "observation_discrete": [obs],
                "action":               action,
                "reward":               reward,
                "done":                 done_flag,
                "episode_step":         step_idx + 1,
            })
            break

print(store)  # DatasetStore(steps=...)

# Push to the Hugging Face Hub — creates the repo if it doesn't exist yet
push_stores_to_hub(
    [store],
    repo_id="your-org/your-dataset",
    split="train",
    private=True,
)
```

> **Multiple splits** — use `push_to_hub` directly if you want separate train/eval splits:
> ```python
> from mouse.data.hub import push_to_hub
>
> push_to_hub(
>     {"train": [train_store], "eval": [eval_store]},
>     repo_id="your-org/your-dataset",
> )
> ```

---

## Offline Training Example

```python
import torch
from datasets import load_dataset
from mouse.models.base import load_model
from mouse.data.dataset_store import DatasetStore
from mouse.data.batch import PrefetchBatchifier
from mouse.losses.dqn import DqnLossConfig, dqn_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained MOUSE model and set it to training mode
model = load_model("your-org/your-model").train().to(device)

# Create the optimizer and after each optimizer step update the target Q-head
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer.register_step_post_hook(
    lambda opt, args, kwargs: model.polyak_update(dqn_tau=dqn_cfg.tau)
)

# Point DatasetStore at your offline RL dataset on HuggingFace Hub
store = DatasetStore(max_action_dim=18, max_obs_continuous_dim=8)
store.from_dataset(load_dataset("your-org/your-dataset", split="train"))

# gamma=0.99 discounts future rewards; tau=0.005 controls target head lag
dqn_cfg = DqnLossConfig(weight=1.0, gamma=0.99, tau=0.005)

# Background threads pre-fetch and encode batches so next_batch() is instant
bf = PrefetchBatchifier(store, sequence_length=64, batch_size=32,
                        sampling="random", prefetch=4, num_workers=2,
                        pin_memory=True)

for step in range(100_000):
    step_stream = bf.next_batch().to(device)

    # Forward pass — out contains Q-values for every (step, action) pair
    out, _ = model(step_stream)

    loss, _ = dqn_loss(step_stream, out, dqn_cfg)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()  # Polyak update fires automatically via post-step hook

bf.close()
```

## Inference Example

```python
import gymnasium as gym
import torch
from tensordict import TensorDict
from mouse.models.base import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_model("your-org/your-model").eval().to(device)

env    = gym.make("FrozenLake-v1", is_slippery=True)
obs, _ = env.reset()

cache     = None
step_idx  = 0
action    = 0
reward    = 0.0
done_flag = 0

while True:
    step_stream = TensorDict(
        {
            "action":         torch.tensor([[action]],     dtype=torch.long),
            "reward":         torch.tensor([[reward]],     dtype=torch.float32),
            "done":           torch.tensor([[done_flag]],  dtype=torch.long),
            "time":           torch.tensor([[step_idx]],   dtype=torch.long),
            "obs_discrete":   torch.tensor([[obs]],         dtype=torch.long),
        },
        batch_size=(1, 1),
    )

    with torch.no_grad():
        out, cache = model(step_stream.to(device), cache=cache, use_cache=True)

    action = model.get_action(out, temperature=0.0)[0].item()
    obs, reward, terminated, truncated, _ = env.step(action)
    done_flag = 1 if terminated else (2 if truncated else 0)
    step_idx += 1

    if terminated or truncated:
        obs, _ = env.reset()
        action    = 0
        reward    = 0.0
        done_flag = 0
```

---

## Guides

- [Data](data.md) — downloading and processing offline RL datasets
- [Model Architecture](architecture.md) — design choices: embedding, backbone, heads
- [Losses](losses.md) — DQN, VecDQN, SP, and SV loss functions
- [Examples](examples.md) — training loop patterns

---

The full API reference is available [here](api/model.md).
