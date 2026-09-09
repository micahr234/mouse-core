from __future__ import annotations
from typing import cast
import json
import pytest
import torch
from mouse_core.models import Model, load_model, save_model
from mouse_core.models.base import _write_model_card
from mouse_core.models.backbone import IdentityBackbone, Qwen3Backbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.data import NumericTokenizer
from mouse_core.models.heads import DiscreteActionValueHead
from tests._token_batch_helpers import batch_to_token_batch, tok_from_encoder

_tok = tok_from_encoder

def test_composed_model_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}, {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}, {"type": 'discrete', "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1}])
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=heads).eval()
    batch = [[{'action': 0, 'reward': 0.0, 'episode_done': 0, 'task_done': 0}, {'action': 1, 'reward': 1.0, 'episode_done': 0, 'task_done': 0}, {'action': 2, 'reward': 2.0, 'episode_done': 1, 'task_done': 0}]]
    expected = model(batch_to_token_batch(_tok(model.encoder), batch)).predictions
    save_model(model, tmp_path)
    loaded = load_model(tmp_path).eval()
    actual = loaded(batch_to_token_batch(_tok(loaded.encoder), batch)).predictions
    assert torch.allclose(actual['action_value'], expected['action_value'])
    assert loaded.hidden_dim == hidden_dim
    orig_sd = model.state_dict()
    loaded_sd = loaded.state_dict()
    assert set(orig_sd.keys()) == set(loaded_sd.keys()), f'state_dict key mismatch:\n  missing in loaded: {set(orig_sd.keys()) - set(loaded_sd.keys())}\n  extra in loaded:   {set(loaded_sd.keys()) - set(orig_sd.keys())}'
    for key in orig_sd:
        assert torch.equal(orig_sd[key], loaded_sd[key]), f'param {key!r} differs after save/load roundtrip'
    with (tmp_path / 'config.json').open() as fh:
        config = json.load(fh)
    assert config['format'] == 'mouse-core-model-v1'
    assert config['backbone']['type'] == 'identity'
    assert config['encoder']['type'] == 'numeric'
    enc_kwargs = config['encoder']['kwargs']
    for required_key in ('hidden_dim', 'modalities'):
        assert required_key in enc_kwargs, f'encoder config missing key {required_key!r}'
    assert 'std' not in enc_kwargs
    assert 'fourier_min' not in enc_kwargs
    assert 'fourier_max' not in enc_kwargs
    assert all(m['std'] == 0.02 for m in enc_kwargs['modalities'])
    fourier_mods = [m for m in enc_kwargs['modalities'] if m['type'] == 'fourier']
    assert fourier_mods
    assert all(m['fourier_min'] == 0.01 and m['fourier_max'] == 10.0 for m in fourier_mods)
    assert all('fourier_min' not in m for m in enc_kwargs['modalities'] if m['type'] != 'fourier')
    assert 'modality_fusion' not in enc_kwargs
    assert 'include_type_token' not in enc_kwargs

def test_roundtrip_multi_field_spec_before_learnable(tmp_path) -> None:
    """Learnable table names must not depend on the raw (unexpanded) spec index."""
    torch.manual_seed(0)
    hidden_dim = 8
    encoder = NumericEmbedder(
        hidden_dim=hidden_dim,
        modalities=[
            {"type": 'discrete', "field": ("action", "prev_action"), "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": 'learnable', "tokens": 2, "std": 0.02, "positions": 2},
            {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
            {"type": 'learnable', "std": 0.02, "positions": 1},
        ],
    )
    assert "encoder._tables.__learnable_0.weight" in {f"encoder.{k}" for k in encoder.state_dict()}
    assert "encoder._tables.__learnable_1.weight" in {f"encoder.{k}" for k in encoder.state_dict()}
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=heads).eval()
    tokenizer = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action"},
            {"type": "discrete", "input_field": "prev_action"},
            {"type": "learnable", "tokens": 2},
            {"type": "fourier", "input_field": "reward"},
            {"type": "learnable", "head_output": True},
        ],
        objective_fields=[],
        grouping_field="task_index",
    )
    batch = [[{'action': 0, 'prev_action': 1, 'reward': 0.5, 'task_index': 0}, {'action': 2, 'prev_action': 0, 'reward': 1.0, 'task_index': 0}]]
    expected = model(batch_to_token_batch(tokenizer, batch, grouping_field="task_index")).predictions
    save_model(model, tmp_path)
    loaded = load_model(tmp_path).eval()
    assert set(model.state_dict()) == set(loaded.state_dict())
    actual = loaded(batch_to_token_batch(tokenizer, batch, grouping_field="task_index")).predictions
    assert torch.allclose(actual['action_value'], expected['action_value'])


def test_composed_model_roundtrip_static_fourier(tmp_path) -> None:
    """Static Fourier buffers survive save/load."""
    torch.manual_seed(42)
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}, {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}])
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=heads).eval()
    batch = [[{'action': 1, 'reward': 0.5}, {'action': 2, 'reward': -0.1}]]
    expected = model(batch_to_token_batch(_tok(model.encoder), batch)).predictions
    save_model(model, tmp_path)
    loaded = load_model(tmp_path).eval()
    actual = loaded(batch_to_token_batch(_tok(loaded.encoder), batch)).predictions
    assert torch.allclose(actual['action_value'], expected['action_value'])
    enc = cast(NumericEmbedder, model.encoder)
    loaded_enc = cast(NumericEmbedder, loaded.encoder)
    assert torch.equal(enc.fourier["reward"].freqs, loaded_enc.fourier["reward"].freqs)

def test_model_card_includes_usage_and_architecture(tmp_path) -> None:
    model = Model(encoder=NumericEmbedder(hidden_dim=8, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}, {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}, {"type": 'discrete', "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1}]), backbone=IdentityBackbone(hidden_dim=8), heads=DiscreteActionValueHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1))
    path = tmp_path / 'README.md'
    _write_model_card(repo_id='user/mouse-example-model', model=model, path=path)
    text = path.read_text()
    assert 'library_name: mouse-core' in text
    assert text.index('## Architecture') < text.index('### Encoder')
    assert text.index('### Encoder') < text.index('## Install MouseCore')
    assert text.index('## Install MouseCore') < text.index('## Load The Model')
    assert text.index('## Load The Model') < text.index('## Run Inference')
    assert 'What This Contains' not in text
    assert 'pip install mouse-core' in text
    assert 'load_model("user/mouse-example-model"' in text
    assert 'NumericTokenizer' in text
    assert '| `action` | `discrete` | `[B, S]` | `torch.long` | integer ids in `[0, 3]` |' in text
    assert 'Fourier range `[0.01, 10.0]`' in text
    assert '"action": 0,' in text
    assert '"reward": 0.0,' in text
    assert 'out, step_stream, cache = model(batch)' not in text
    assert 'compose' in text
    assert 'pack_token_batch' in text
    assert 'DataLoader(transform=compose(augmenter, tokenizer))' in text
    assert 'eval_transform = tokenizer' in text
    assert 'grouping_field="task_index"' in text
    assert '{"input_field": "action"}' in text
    assert 'token_batch.grouper' not in text
    assert 'boundary_values' not in text
    assert 'Backbone: `identity`' in text
    assert 'Heads: `action_value`' in text

@pytest.mark.parametrize(
    "cast",
    [
        lambda m: m.to(torch.bfloat16),
        lambda m: m.to(dtype=torch.bfloat16),
        lambda m: m.to("cpu", torch.bfloat16),
        lambda m: m.to(device="cpu", dtype=torch.bfloat16),
        lambda m: m.to(torch.zeros(1, dtype=torch.bfloat16)),
        lambda m: m.bfloat16(),
        lambda m: m.to(dtype=torch.bfloat16).to("cpu"),
    ],
    ids=["pos-dtype", "kw-dtype", "pos-device-dtype", "kw-device-dtype", "tensor", "bfloat16()", "then-device"],
)
def test_model_to_every_form_keeps_heads_float32(cast) -> None:
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}, {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}])
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    model = cast(Model(encoder=encoder, backbone=backbone, heads=heads).eval())
    assert {p.dtype for p in model.encoder.parameters()} == {torch.bfloat16}
    assert {p.dtype for p in model.heads.parameters()} == {torch.float32}
    batch = [[{'action': 0, 'reward': 0.5, 'task_index': 0}, {'action': 1, 'reward': 1.0, 'task_index': 0}]]
    with torch.no_grad():
        preds = model(batch_to_token_batch(_tok(model.encoder), batch)).predictions
    assert preds['action_value'].dtype == torch.float32


def test_model_to_bfloat16_keeps_heads_float32() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}, {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}, {"type": 'discrete', "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1}])
    backbone = Qwen3Backbone(hidden_dim=hidden_dim, num_layers=1, num_heads=2)
    heads = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=heads).eval()
    model = model.to(device=torch.device('cuda'), dtype=torch.bfloat16)
    assert next(model.encoder.parameters()).dtype == torch.bfloat16
    assert next(model.backbone.parameters()).dtype == torch.bfloat16
    assert next(model.heads.parameters()).dtype == torch.float32
    batch = [[{'action': 0, 'reward': 0.0, 'episode_done': 0, 'task_done': 0}]]
    with torch.no_grad():
        preds = model(batch_to_token_batch(_tok(model.encoder), batch), use_cache=True).predictions
    assert preds['action_value'].dtype == torch.float32
