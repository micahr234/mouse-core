# Examples

Copy-paste examples for common use cases. Training scripts live in a separate repo; this library supplies the model, losses, and data utilities.

---

## Setup

```bash
source scripts/install.sh
```

This installs `uv`, creates a Python 3.12 venv at `.venv/`, and installs the package in editable mode with dev dependencies.

---

## Minimal training loop

```python
import torch
from datasets import load_dataset
from tensordict import TensorDict

from mouse.models.base import load_model
from mouse.data.dataset_store import DatasetStore
from mouse.data.batch import PrefetchBatchifier
from mouse.losses.dqn import DqnLossConfig, dqn_loss
from mouse.losses.sp import SpLossConfig, sp_loss

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
model = load_model("your-org/your-model").to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# ── Data ──────────────────────────────────────────────────────────────────────
store = DatasetStore(max_action_dim=18, max_obs_continuous_dim=8)
store.from_dataset(load_dataset("your-org/your-dataset", split="train"))

# ── Loss configs ──────────────────────────────────────────────────────────────
dqn_cfg = DqnLossConfig(weight=1.0, gamma=0.99, tau=0.005)
sp_cfg  = SpLossConfig(weight=0.5, loss_type="ce-soft-fwd", temperature=1.0)

# ── Training ──────────────────────────────────────────────────────────────────
num_steps = 100_000

with PrefetchBatchifier(
    store,
    sequence_length=64,
    batch_size=32,
    sampling="random",
    prefetch=4,
    num_workers=2,
    pin_memory=True,
) as bf:
    for step in range(num_steps):
        step_stream = bf.next_batch().to(device)

        out, _ = model(step_stream)

        loss = torch.tensor(0.0, device=device)

        if dqn_cfg.weight > 0:
            l, _ = dqn_loss(step_stream, out["dqn"], out["dqn_target"], dqn_cfg)
            loss = loss + dqn_cfg.weight * l

        if sp_cfg.weight > 0:
            l, _ = sp_loss(step_stream, out["sp"], sp_cfg)
            loss = loss + sp_cfg.weight * l

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.polyak_update(dqn_tau=dqn_cfg.tau)
```

---

## Data-augmented loop

```python
from mouse.data.augment import TokenAugmenter, AugmentTokensConfig, AugmentScalarSpec

aug_cfg = AugmentTokensConfig(
    scale_reward=AugmentScalarSpec(mean=1.0, low=0.5, high=2.0),
    permute_action="both",
)

augmenter = TokenAugmenter(
    augment=aug_cfg,
    max_num_actions=18,
    max_num_obs_discrete=0,
    device=device,
)

with PrefetchBatchifier(store, sequence_length=64, batch_size=32) as bf:
    for step in range(num_steps):
        step_stream = bf.next_batch().to(device)

        # Sample new permutations/scalars for this batch
        augmenter.update_augmentations(step_stream)
        aug_stream = augmenter(step_stream)   # cloned; original unchanged

        out, _ = model(aug_stream)
        # ... compute losses on aug_stream ...
```

---

## Saving and loading

MOUSE models implement `PyTorchModelHubMixin` from HuggingFace Hub, so `save_pretrained` / `from_pretrained` / `push_to_hub` all work out of the box.

```python
# Save locally
model.save_pretrained("./my-checkpoint")

# Push to HuggingFace Hub
model.push_to_hub("your-org/your-model")

# Load (auto-detects backbone class)
from mouse.models.base import load_model
model = load_model("your-org/your-model")
```

---

## Model cards

`MODEL_CARD_TEMPLATE` in `mouse.models.base` is a Jinja-style template. When you call `push_to_hub`, HuggingFace Hub renders it automatically using the stored `config.json` fields, producing a formatted README with code examples tailored to your model's modalities and heads.

---

## Inference / evaluation

Use `model.eval()` and `torch.no_grad()` for evaluation. For online rollouts with minimal recomputation use the KV-cache:

```python
model.eval()
cache = None
step_idx = 0

while not done:
    step_stream = TensorDict(
        {
            "action":         last_action.unsqueeze(1),
            "reward":         last_reward.unsqueeze(1),
            "done":           last_done.unsqueeze(1),
            "time":           torch.full((B, 1), step_idx, dtype=torch.long),
            "obs_continuous": obs.unsqueeze(1).float(),
        },
        batch_size=(B, 1),
    )

    with torch.no_grad():
        out, cache = model(step_stream.to(device), cache=cache, use_cache=True)

    action = model.get_action(out, temperature=0.0)
    step_idx += 1
```

Reset the cache (`cache = None`) before the context grows to roughly 2× the training sequence length to avoid quality degradation.

---

## Tips

**Gradient clipping.** Use `clip_grad_norm_(model.parameters(), 1.0)` — transformer models are sensitive to large gradient spikes early in training.

**Polyak rate.** `tau=0.005` is a reasonable default. Higher values (`tau=0.01`) update the target faster; lower values (`tau=0.001`) give a more stable bootstrap target.

**Reward normalisation.** Enable `normalize_reward_mean=True` and/or `normalize_reward_std=True` in `DqnLossConfig` / `VecDqnLossConfig` when rewards vary widely across episodes. Normalisation is applied per sequence row inside the loss.

**Mixed precision.** MOUSE models are compatible with `torch.autocast`:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out, _ = model(step_stream)
```

**Sequence length.** Keep `sequence_length` close to the value the model was trained with. Too-short sequences under-utilise the context; too-long sequences are out-of-distribution for the positional encodings.

**`xformed_reward`.** If your dataset includes a pre-computed transformed reward (e.g. clipped or log-scaled), set `use_xformed_reward=True` in `DqnLossConfig` to use `xformed_reward` instead of `reward` as the TD signal. The raw `reward` field is still available for logging.
