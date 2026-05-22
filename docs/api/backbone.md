# Backbone

Transformer backbones (or a no-op pass-through) sit between the step embedder and output heads.

## Classes

| Class | Source |
|-------|--------|
| [`ModelLlama`](../../src/models/backbone/llama.py) | LLaMA-family causal backbone |
| [`ModelQwen3`](../../src/models/backbone/qwen3.py) | Qwen3 causal backbone |
| [`ModelNone`](../../src/models/backbone/none.py) | No backbone; embeddings feed heads directly |

## Config helpers

| Name | Source |
|------|--------|
| [`LlamaBackboneConfig`](../../src/models/backbone/llama.py) | LLaMA backbone kwargs |
| [`Qwen3BackboneConfig`](../../src/models/backbone/qwen3.py) | Qwen3 backbone kwargs |

Package overview: [`mouse.models.backbone`](../../src/models/backbone/__init__.py).
