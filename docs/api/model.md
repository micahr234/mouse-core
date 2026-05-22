# Base

The [`mouse.models.base`](../../src/models/base.py) module defines the abstract `Model` class and the `load_model` factory. Use `load_model` in most cases — it reads `config.json` and picks the right backbone implementation.

## Symbols

| Name | Description |
|------|-------------|
| [`load_model`](../../src/models/base.py) | Load a MOUSE model from a local path or Hugging Face Hub repo |
| [`save_model`](../../src/models/base.py) | Save weights and config to a directory |
| [`push_model_to_hub`](../../src/models/base.py) | Upload a model to the Hub |
| [`Model`](../../src/models/base.py) | Base class: embedder, backbone, heads, forward pass, `get_action` |
| [`init_from_pretrained_backbone`](../../src/models/base.py) | Initialize backbone weights from a pretrained checkpoint |

See also [`mouse.models`](../../src/models/__init__.py) for re-exports.
