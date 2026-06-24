# Models

A MOUSE model has three pieces:

```text
list[list[dict]]  [B][S] of raw step records
    -> Encoder (StepEmbedder) -> token embeddings [B, T, D]
    -> Backbone (Llama, Qwen3, or Identity) -> hidden states [B, T, D]
    -> Heads -> out: TensorDict[B, S]
```

`Model.forward(batch)` returns `(out, step_stream, cache)`. `step_stream` is a `TensorDict[B, S]` of the modality tensors extracted by the encoder — use it in objectives. `out` contains one key per head.

Build those pieces directly, then compose them with `Model`:

```python
from mouse_core.models import Model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import StepEmbedder
from mouse_core.models.heads import DiscreteActionValueHead

hidden_dim = 128

encoder = StepEmbedder(
    hidden_dim=hidden_dim,
    modalities=[
        {"name": "action", "embed": "discrete", "vocab_size": 4},
        {"name": "reward", "embed": "rff"},
        {"name": "done", "embed": "discrete", "vocab_size": 3},
    ],
)
backbone = IdentityBackbone(hidden_dim=hidden_dim)
heads = DiscreteActionValueHead(
    in_features=hidden_dim,
    out_features=4,
    hidden_dim=hidden_dim,
    num_layers=1,
)

model = Model(encoder=encoder, backbone=backbone, heads=heads)
```

`B` is batch size, `S` is step sequence length, `T` is total token length, and `D` is hidden dimension.

## Pretrained Backbones

Backbone loading has one public path: instantiate the backbone you want.

```python
from mouse_core.models.backbone import LlamaBackbone

backbone = LlamaBackbone(
    pretrained="meta-llama/Llama-3.2-1B",
    num_layers=2,
)
hidden_dim = backbone.hidden_dim
```

The constructor reads the pretrained config, applies overrides such as `num_layers`, builds the MOUSE-compatible transformer, and loads matching transformer weights. `Qwen3Backbone` uses the same pattern for Qwen checkpoints.

Pass Hugging Face options through `hub_kwargs`, for example gated-model tokens or revisions:

```python
backbone = LlamaBackbone(
    pretrained="meta-llama/Llama-3.2-1B",
    num_layers=2,
    hub_kwargs={"token": "...", "revision": "main"},
)
```

Only transformer blocks are loaded. Token embeddings are skipped because MOUSE uses `StepEmbedder`; the final norm is skipped because MOUSE replaces it with `nn.Identity`. Pass `load_weights=False` when you only want the architecture.

```python
backbone = LlamaBackbone(
    pretrained="meta-llama/Llama-3.2-1B",
    num_layers=2,
    load_weights=False,
)
```

Build the encoder and heads from `backbone.hidden_dim` so dimensions stay aligned.

```python
encoder = StepEmbedder(hidden_dim=backbone.hidden_dim, modalities=[...])
heads = DiscreteActionValueHead(
    in_features=backbone.hidden_dim,
    out_features=...,
    hidden_dim=backbone.hidden_dim,
    num_layers=1,
)
model = Model(encoder=encoder, backbone=backbone, heads=heads)
```

## Encoder

`StepEmbedder` accepts `list[list[dict]]` (shape `[B][S]`) and extracts only the fields its `modalities` config declares. Each modality reads one key from the row dicts and declares how to embed it:

- `"discrete"`: integer ids through a learned embedding table.
- `"rff"`: scalar floats through random Fourier features.
- `"continuous"`: float vectors, zero-padded/truncated to `dim`.
- `"image"`: image or patch values through per-position learned projections.
- `"learnable"`: learned scratch tokens; no input key is required.

Extraction happens in a single pass through the batch, filling all modality buffers simultaneously. Only the columns the encoder needs are touched; all other fields in the dicts are ignored.

Modalities are required by default: if the declared field is absent from every row in the batch the encoder raises a `KeyError`. Set `required=False` when a missing field should use the encoder's default value. `learnable` modalities must use `required=False`.

### Building modality configs from env specs

Row dicts produced by `mouse-env` are flat: `{"action": tensor, "observation": tensor, "reward": tensor, "done": tensor, ...}`. The action and observation `FieldSpec` objects exposed by `env.input_spec` and `env.output_spec` carry the dtype that determines which embedder to use: `torch.int64` → discrete, `torch.float32` → continuous or image.

`action_modality` and `observation_modalities` turn those specs into modality config dicts automatically:

```python
from mouse_envs import make_vector_env
from mouse_core.models.embedding import action_modality, observation_modalities

env = make_vector_env(cfg)

modalities = [
    # action_modality reads dtype: int64 → "discrete", float32 → "continuous"
    action_modality(env.input_spec.action.dtype, vocab_size=4, tokens=1),
    # observation_modalities handles both single FieldSpec and Dict obs spaces
    *observation_modalities(env.output_spec.observation, vocab_sizes=16, tokens=1),
    {"name": "reward", "embed": "rff"},
    {"name": "done",   "embed": "discrete", "vocab_size": 3},
]

encoder = StepEmbedder(hidden_dim=hidden_dim, modalities=modalities)
env.close()
```

For `gym.spaces.Dict` observation spaces the subspace keys land directly on the output dict (not under `"observation"`). `observation_modalities` detects this automatically and produces one modality per subspace key; pass `vocab_sizes` as a `dict[key → int]` in that case.

`StepEmbedder.hidden_dim` is checked against `backbone.hidden_dim` by `Model`. If you need padding or variable token visibility, pass an explicit mask to the model:

```python
out, step_stream, cache = model(batch, attention_mask=attention_mask)
```

## Backbone

A backbone maps token embeddings to hidden states:

```python
h, cache = backbone(embeds, cache=cache, use_cache=True, attention_mask=attention_mask)
```

Built-in backbones are:

- `LlamaBackbone`: wraps `transformers.LlamaModel`.
- `Qwen3Backbone`: wraps `transformers.Qwen3Model`.
- `IdentityBackbone`: no temporal mixing, useful for smoke tests and ablations.

Pass `use_cache=True` for incremental inference when the backbone supports KV-cache. See [`examples/03_inference.ipynb`](../examples/03_inference.ipynb).

## Heads

For one output, pass a single head directly:

```python
heads = DiscreteActionValueHead(
    in_features=hidden_dim,
    out_features=max_num_actions,
    hidden_dim=hidden_dim,
    num_layers=1,
)
model = Model(encoder=encoder, backbone=backbone, heads=heads)
```

For multiple outputs, pass a dict and choose which one `get_action` uses:

```python
heads = {
    "action_value": DiscreteActionValueHead(...),
    "action_vector": VectorActionValueHead(...),
}
model = Model(
    encoder=encoder,
    backbone=backbone,
    heads=heads,
    action_head="action_value",
)
```

Supported head names are `"action_value"`, `"action_vector"`, `"action"`, and `"value"`.

## Actions and Target Updates

`Model.get_action` reads the configured `action_head` from the last step:

```python
out, _, cache = model(batch)
action = model.get_action(out, temperature=0.0)
```

Use `temperature=0.0` for greedy argmax and `temperature>0` for sampling. If the current environment has fewer actions than the head maximum, pass `num_actions`.

DQN-style heads with target networks are updated through the model:

```python
model.polyak_update(action_value_tau=0.005, action_vector_tau=0.005)
```

## Saving and Loading

Save and load full MOUSE models with the model helpers:

```python
from mouse_core.models import load_model, push_model_to_hub, save_model

save_model(model, "./checkpoints/step-10000")
model = load_model("./checkpoints/step-10000")

push_model_to_hub(model, "my-mouse-model", private=True, clear=True)
model = load_model("my-mouse-model")
```

`load_model` is for MOUSE checkpoints. Pretrained language-model loading stays inside the backbone constructor.
