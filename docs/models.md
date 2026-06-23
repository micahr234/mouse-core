# Models

A MOUSE model has three pieces:

```text
TensorDict[B, S]
    -> Encoder (StepEmbedder) -> token embeddings [B, T, D]
    -> Backbone (Llama, Qwen3, or Identity) -> hidden states [B, T, D]
    -> Heads -> TensorDict[B, S]
```

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

`StepEmbedder` converts a `TensorDict[B, S]` of step records into token embeddings. Each modality reads one key from the input and declares how to embed it:

- `"discrete"`: integer ids through a learned embedding table.
- `"rff"`: scalar floats through random Fourier features.
- `"continuous"`: float vectors, using RFF by default or `method="linear"`.
- `"image"`: image or patch values through per-position learned projections.
- `"learnable"`: learned scratch tokens; no input key is required.

Modalities can contribute one or more tokens via `tokens`. In the default sum mode (`concat_modalities=False`), modalities share a step block and are summed into matching positions. In concat mode (`concat_modalities=True`), modality blocks are laid out in order.

`StepEmbedder.hidden_dim` is checked against `backbone.hidden_dim` by `Model`. If you need padding or variable token visibility, pass an explicit mask to the model:

```python
out, cache = model(step_stream, attention_mask=attention_mask)
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
out, cache = model(step_stream)
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

push_model_to_hub(model, "my-mouse-model", private=True)
model = load_model("my-mouse-model")
```

`load_model` is for MOUSE checkpoints. Pretrained language-model loading stays inside the backbone constructor.
