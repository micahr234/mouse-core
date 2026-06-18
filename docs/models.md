# Models

MOUSE is a context-conditioned sequence model. The full forward pass has three stages:

```
TensorDict[B, S]
       │
       ▼
  StepEmbedder          → [B, S*T, D]   (multi-modal per-step tokens)
       │
       ▼
   Backbone             → [B, S*T, D]   (Llama / Qwen3 / identity)
       │
  last-token pool
       │
       ▼
  Output heads          → TensorDict[B, S]
```

`B` = batch size, `S` = sequence length (steps), `T` = tokens per step, `D` = hidden dimension.

---

## StepEmbedder (`mouse_core.models.embedding.embedding`)

Each environment step is embedded into a block of `tokens_per_step` vectors. Two modes control how modalities map to tokens:

**Sum mode** (`concat_modalities=False`, default) — every active modality contributes a flat `T*D` vector that is **added** to a shared `T`-token accumulator:

```
token[i] = Σ_modality  type_embed(modality) + content_embed(modality, i)

tokens_per_step = T + K
```

**Concat mode** (`concat_modalities=True`) — each modality gets its own dedicated block of `T` tokens laid out sequentially:

```
[time×T | action×T | obs_cont×T | ... | compute×K]

tokens_per_step = M*T + K    (M = number of active modalities)
```

**Compute tokens** (`num_compute_tokens=K`) — `K` learned scratch tokens are appended after the data tokens in every step block. They carry no input data; their embedding is a shared `[K, D]` parameter broadcast over `(B, S)`. The backbone can use them as working memory. The step representation is always pooled from the **last** token (i.e. the last compute token when `K > 0`).

After embedding, the full sequence `[B, S*tokens_per_step, D]` is passed to the backbone.

### Modalities

| Field | Type | Embedder |
|---|---|---|
| `action` | `int64` | `ActionEmbedder` — learned embedding table |
| `reward` | `float32` | `RewardEmbedder` — Random Fourier Features |
| `done` | `int64` {0,1,2} | `DoneEmbedder` — 3-entry embedding table |
| `time` | `int64` | `TimeEmbedder` — learned embedding table |
| `obs_continuous` | `float32[C]` | `ObsContinuousEmbedder` — per-dim position-indexed RFF; or `ObsContinuousLinearEmbedder` — learned per-dim linear |
| `obs_discrete` | `int64` | `ObsDiscreteEmbedder` — learned embedding table |
| `obs_image` | `int64[P]` | `ObsImageEmbedder` — per-pixel position-indexed linear on normalised pixel values |

Each modality is independently optional. Set the corresponding `include_*` flag to `False` in `embedding_kwargs` to omit it. The corresponding field is then not required in `step_stream`.

### Token types

`TokenType` (IntEnum) labels each token position. The backbone currently masks only `PAD` tokens; other token types flow through unmasked.

```python
class TokenType(IntEnum):
    PAD          = 0  # padding / unused
    ACTION       = 1
    REWARD       = 2
    DONE         = 3
    OBS_IMAGE    = 4
    OBS_CONTINUOUS = 5
    TIME         = 6
    OBS_DISCRETE = 7
    COMPUTE      = 8  # learned scratch token (no input data)
```

In sum mode all data positions carry type `1` (generic non-PAD). In concat mode each modality block carries its own type. Compute positions always carry `COMPUTE = 8`. The backbone uses these labels only to build the attention mask (non-zero = real token, zero = pad).

### Initialisation scaling

Embedding tables use `ScaledEmbedding` initialised at `std = embedding_std` (default 0.02). RFF-based embedders are scaled so each modality contributes roughly the same output variance regardless of its dimension count (sums over `max_num_obs` dims are divided by `√max_num_obs`).

---

## Backbone

The backbone runs a standard transformer over the `[B, S*tokens_per_step, D]` token sequence and returns hidden states of the same shape.

### Llama (`mouse_core.models.backbone.llama.ModelLlama`)

Uses `transformers.LlamaModel` (SDPA attention, no embedding layer, `vocab_size=1`). The final layer norm is replaced with `nn.Identity` so the model outputs raw hidden states. The cuDNN SDPA backend is disabled to avoid known numerical issues.

### Qwen3 (`mouse_core.models.backbone.qwen3.ModelQwen3`)

Same approach using `transformers.Qwen3Model`. Supports an explicit `head_dim` for grouped-query attention.

### None (`mouse_core.models.backbone.none.ModelNone`)

`nn.Identity` backbone — useful for ablations where no temporal context is needed. Does not support KV-cache.

### Auto-detection

`load_model` inspects `backbone_kwargs` in `config.json` and instantiates the right class:

- empty `backbone_kwargs` → `ModelNone`
- `head_dim` present → `ModelQwen3`
- otherwise → `ModelLlama`

### KV-cache

Pass `use_cache=True` to enable incremental decoding. The `cache` dict returned from one call is passed back as input to the next. Only meaningful for `ModelLlama` and `ModelQwen3`; `ModelNone` always returns `None`. See [`examples/03_inference.ipynb`](../examples/03_inference.ipynb) for a single-step cached forward pass.

---

## Last-token pooling

After the backbone, `Model.forward` reshapes hidden states to `[B, S, T, D]` and takes the **last token of each step**:

```python
h_step = h.view(B, S, T, D)[:, :, -1, :]   # [B, S, D]
```

This `[B, S, D]` tensor is the step-level representation fed to every output head.

---

## Output heads

All heads take `[B, S, D]` and return `[B, S, ...]`.

### SwiGLUHead (`mouse_core.models.heads.swiglu`)

Shared building block for `sp` and `sv` heads:

```
RMSNorm (optional) → [ Linear (2D) → SiLU × Linear (D) ] × num_layers → ScaledLinear
```

### DQNHead (`mouse_core.models.heads.dqn`)

Two `SwiGLUHead` copies — **online** and **target** — with the same architecture. `target_forward` runs the target head with no gradient tracking. Call `polyak_update(tau)` after each optimiser step:

```
θ_target ← τ·θ_online + (1−τ)·θ_target
```

### VecDQNHead (`mouse_core.models.heads.vec_dqn`)

Like `DQNHead` but each action produces a `vec_dim`-dimensional vector. Output shape: `[B, S, A, vec_dim]`.

**Scoring.** Convert vectors to scalar Q-scores with `vec_dqn_scores(vecs)`. For each action pair `(i, a)` the function computes the full signed angle `φ_a − φ_i` via `atan2(sin, cos)` and sums over all `i`. Using `atan2` rather than a raw dot product avoids aliasing above 90° and gives a score that is monotone over the full `(−π, +π)` range.

**Rotation.** `rope_rotate(x, theta)` rotates each consecutive pair of dimensions in `x` by `theta` (standard RoPE). For `D=2` this is geometrically exact; for `D>2` each pair of dimensions is an independent rotation plane.

---

## Model.get_action

```python
action = model.get_action(out, temperature=0.0)  # [B]
```

Reads from `model.action_head`, set at construction time and saved in `config.json`. Auto-detected from enabled heads if not specified: `vec_dqn` > `dqn` > `sp` > `sv`.

- `temperature=0.0` → greedy argmax.
- `temperature>0` → softmax sampling with numerical stability (max subtraction).
- `vec_dqn` head → `vec_dqn_scores` applied automatically before argmax/sampling.
- `num_actions` → trim score tensor to the first N actions (useful when the environment has fewer actions than the model maximum).

---

## Model.polyak_update

```python
model.polyak_update(dqn_tau=0.005, vec_dqn_tau=0.005)
```

Delegates to each enabled twin-head. Call once per optimiser step.

---

## Saving and loading models

### Local checkpoint

```python
from mouse_core.models import save_model, load_model

save_model(model, "./checkpoints/step-10000")
model = load_model("./checkpoints/step-10000")
```

### Hugging Face Hub

```python
from mouse_core.models import push_model_to_hub, load_model

# Upload weights + config + auto-generated model card
push_model_to_hub(model, "your-org/your-model")

# Download and instantiate — class is inferred automatically
model = load_model("your-org/your-model")
```

`push_model_to_hub` is a thin wrapper around `PyTorchModelHubMixin.push_to_hub` and accepts the same keyword arguments (e.g. `token`, `revision`, `private`).

---

## Initialising from a pretrained backbone

`init_from_pretrained_backbone` builds a MOUSE model whose backbone architecture and weights come from any Llama- or Qwen3-family checkpoint on the Hub. Architecture defaults (layer count, head count, hidden dim, FFN size) are read from the pretrained `config.json` automatically — you only need to specify the MOUSE-specific parts (embedding config and output heads).

```python
from mouse_core.models import init_from_pretrained_backbone

model = init_from_pretrained_backbone(
    "meta-llama/Llama-3.2-1B",
    embedding_kwargs=dict(
        max_num_actions=18,
        include_obs_continuous=True,
        max_num_obs_continuous=8,
        include_action_token=True,
        include_reward_token=True,
        include_done_token=True,
        token_data_len=4,
    ),
    dqn_head_kwargs=dict(num_layers=2, hidden_dim=256),
)
```

Individual backbone settings can be overridden via `backbone_kwargs_overrides` — for example, to use only the first 8 layers of a 16-layer model (the pretrained weights for those 8 layers are still loaded):

```python
model = init_from_pretrained_backbone(
    "Qwen/Qwen3-0.6B",
    backbone_kwargs_overrides={"num_layers": 8},
    embedding_kwargs=...,
    dqn_head_kwargs=...,
)
```

If you only need the architecture (no weight loading), pass `load_weights=False`.

### What is and isn't loaded

| Component | Loaded |
|---|---|
| Transformer layers (attention, FFN, layer norms) | ✅ |
| Embedding table (`embed_tokens`) | ❌ — vocab size mismatch; MOUSE uses its own `StepEmbedder` |
| Final norm | ❌ — replaced with `nn.Identity` |
| Output heads, `StepEmbedder` | ❌ — always randomly initialised |

### Inspecting backbone kwargs without building a model

`backbone_kwargs_from_pretrained` returns the raw kwargs dict and hidden dim, letting you inspect or further customise before passing them to a model constructor:

```python
from mouse_core.models.backbone import backbone_kwargs_from_pretrained

backbone_kwargs, hidden_dim = backbone_kwargs_from_pretrained(
    "meta-llama/Llama-3.2-1B",
    num_layers=8,   # override: use 8 layers instead of 16
)
print(hidden_dim)       # 2048
print(backbone_kwargs)  # {num_layers: 8, num_heads: 32, ...}
```

---

## Config layout (`config.json`)

When saved via `push_to_hub` or `save_pretrained` (HuggingFace Hub mixin), the model writes its constructor kwargs as JSON. `load_model` reads this file to select the right class and instantiate it.

Key fields:

```json
{
  "hidden_dim": 512,
  "backbone_kwargs": { "num_hidden_layers": 8, "num_attention_heads": 8, ... },
  "embedding_kwargs": {
    "max_num_actions": 18,
    "include_action_token": true,
    "include_reward_token": true,
    "include_done_token": true,
    "include_time_token": true,
    "include_obs_continuous": true,
    "max_num_obs_continuous": 8,
    "token_data_len": 4,
    ...
  },
  "sp_head_kwargs":      { "num_layers": 2, "hidden_dim": 256 },
  "dqn_head_kwargs":     { "num_layers": 2, "hidden_dim": 256 },
  "vec_dqn_head_kwargs": { "num_layers": 2, "hidden_dim": 256, "vec_dim": 2 },
  "sv_head_kwargs":      { "num_layers": 0 }
}
```

A head is disabled when its `num_layers` is 0 (or the key is absent).
