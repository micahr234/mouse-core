from __future__ import annotations
from typing import Any, cast
import json
import pytest
import torch
from mouse_core.models import Model, load_model, push_model_to_hub, save_model
from mouse_core.models.base import _write_model_card
from mouse_core.models.backbone import IdentityBackbone, LoRAConfig, TransformerBackbone
from mouse_core.data import Tokenizer, load_tokenizer, save_tokenizer
from mouse_core.models.heads import RegressionHead
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

_TOK = token_tokenizer("action", "episode_done")


def test_composed_model_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    backbone = IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32)
    heads = RegressionHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, use_norm=True)
    model = Model(backbone=backbone, heads=heads, action_source="action_value", reasoner=None).eval()
    batch = [[{'action': 0, 'reward': 0.0, 'episode_done': 0, 'task_done': 0}, {'action': 1, 'reward': 1.0, 'episode_done': 0, 'task_done': 0}, {'action': 2, 'reward': 2.0, 'episode_done': 1, 'task_done': 0}]]
    expected = model(batch_to_token_batch(_TOK, batch)).predictions
    save_model(model=model, path=tmp_path)
    loaded = load_model(repo_id_or_path=tmp_path, train_kernel="reference", decode_kernel="flex", dtype=torch.float32).eval()
    actual = loaded(batch_to_token_batch(_TOK, batch)).predictions
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
    assert config['heads']['action_source'] == 'action_value'
    assert config['backbone']['type'] == 'identity'
    assert config['backbone']['hidden_dim'] == hidden_dim
    assert config['backbone']['vocab_size'] == 32
    assert 'embedder' not in config['backbone']


def test_kernels_and_dtype_are_not_saved_and_come_from_the_loader(tmp_path) -> None:
    hidden_dim = 8
    backbone = TransformerBackbone(architecture="qwen3", train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, hidden_dim=hidden_dim, num_layers=1, num_heads=2, vocab_size=32)
    heads = RegressionHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, use_norm=True)
    save_model(model=Model(backbone=backbone, heads=heads, action_source="action_value", reasoner=None), path=tmp_path)
    with (tmp_path / 'config.json').open() as fh:
        cfg = json.load(fh)['backbone']
    assert 'train_kernel' not in cfg and 'decode_kernel' not in cfg and 'dtype' not in cfg
    assert cfg['type'] == 'transformer'
    assert cfg['architecture'] == 'qwen3'
    assert cfg['kwargs']['use_norm'] is True
    loaded = cast(TransformerBackbone, load_model(repo_id_or_path=tmp_path, train_kernel="flex", decode_kernel="flex", dtype=torch.float32).backbone)
    assert (loaded.train_kernel, loaded.decode_kernel) == ('flex', 'flex')
    with pytest.raises(TypeError, match="train_kernel.*decode_kernel.*dtype"):
        load_model(repo_id_or_path=tmp_path)  # type: ignore[call-arg]


def test_use_norm_false_roundtrip(tmp_path) -> None:
    hidden_dim = 8
    backbone = TransformerBackbone(architecture="qwen3",
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32,
        use_norm=False, hidden_dim=hidden_dim, num_layers=1, num_heads=2, vocab_size=32)
    heads = RegressionHead(
        in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1,
        use_norm=False,
    )
    model = Model(backbone=backbone, heads=heads, action_source="action_value", reasoner=None)
    save_model(model=model, path=tmp_path)
    with (tmp_path / 'config.json').open() as fh:
        config = json.load(fh)
    assert config['backbone']['kwargs']['use_norm'] is False
    assert config['heads']['heads'][0]['use_norm'] is False
    loaded = load_model(repo_id_or_path=tmp_path, train_kernel="reference", decode_kernel="flex", dtype=torch.float32)
    loaded_bb = cast(TransformerBackbone, loaded.backbone)
    assert isinstance(loaded_bb.model.norm, torch.nn.Identity)
    assert loaded_bb._config_kwargs['use_norm'] is False
    assert loaded._heads['action_value'].norm is None
    assert loaded._heads['action_value'].use_norm is False


def test_head_requires_use_norm() -> None:
    with pytest.raises(TypeError, match="use_norm"):
        RegressionHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1)  # type: ignore[call-arg]


def test_transformer_backbone_requires_kernels_and_dtype() -> None:
    with pytest.raises(TypeError, match="train_kernel.*decode_kernel.*dtype"):
        TransformerBackbone(architecture="qwen3", hidden_dim=8, num_layers=1, num_heads=2)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="train_kernel"):
        TransformerBackbone(architecture="qwen3", train_kernel=cast(Any, "sdpa"), decode_kernel="flex", dtype=torch.float32, use_norm=True, hidden_dim=8, num_layers=1, num_heads=2)
    with pytest.raises(ValueError, match="decode_kernel"):
        TransformerBackbone(architecture="qwen3", train_kernel="reference", decode_kernel=cast(Any, "sdpa"), dtype=torch.float32, use_norm=True, hidden_dim=8, num_layers=1, num_heads=2)
    with pytest.raises(TypeError, match="dtype"):
        TransformerBackbone(architecture="qwen3", train_kernel="reference", decode_kernel="flex", dtype=torch.int8, use_norm=True, hidden_dim=8, num_layers=1, num_heads=2)


def test_identity_embed_tokens_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    backbone = IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32)
    heads = RegressionHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, use_norm=True)
    model = Model(backbone=backbone, heads=heads, action_source="action_value", reasoner=None).eval()
    tokenizer = token_tokenizer("action", "prev_action")
    batch = [[{'action': 0, 'prev_action': 1, 'reward': 0.5, 'grouping_id': 0}, {'action': 2, 'prev_action': 0, 'reward': 1.0, 'grouping_id': 0}]]
    expected = model(batch_to_token_batch(tokenizer, batch)).predictions
    save_model(model=model, path=tmp_path)
    loaded = load_model(repo_id_or_path=tmp_path, train_kernel="reference", decode_kernel="flex", dtype=torch.float32).eval()
    assert set(model.state_dict()) == set(loaded.state_dict())
    actual = loaded(batch_to_token_batch(tokenizer, batch)).predictions
    assert torch.allclose(actual['action_value'], expected['action_value'])
    assert torch.equal(
        cast(IdentityBackbone, model.backbone).embed_tokens.weight,
        cast(IdentityBackbone, loaded.backbone).embed_tokens.weight,
    )


def test_model_card_includes_usage_and_architecture(tmp_path) -> None:
    model = Model(backbone=IdentityBackbone(hidden_dim=8, vocab_size=32), heads=(head := RegressionHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True)), action_source="action_value", reasoner=None)
    path = tmp_path / 'README.md'
    _write_model_card(repo_id='user/mouse-example-model', tokenizer_repo_id='user/mouse-example-tokenizer', model=model, path=path)
    text = path.read_text()
    assert 'library_name: mouse-core' in text
    assert text.index('## Architecture') < text.index('### Token embeddings')
    assert text.index('### Token embeddings') < text.index('## Install MouseCore')
    assert text.index('## Install MouseCore') < text.index('## Load The Model')
    assert text.index('## Load The Model') < text.index('## Run Inference')
    assert 'What This Contains' not in text
    assert 'pip install mouse-core' in text
    assert 'load_model(' in text and 'train_kernel="flex"' in text and 'decode_kernel="flex"' in text and 'dtype=preferred_dtype(device=device)' in text
    assert 'Tokenizer' in text
    assert 'embed_tokens' in text
    assert '"action": 0,' in text
    assert '"reward": 0.0,' in text
    assert 'out, step_stream, cache = model(batch)' not in text
    assert 'compose' in text
    assert 'pack_token_batch' in text
    assert 'DataLoader(transform=compose(stages=(augmenter, tokenizer)))' in text
    assert 'eval_transform = tokenizer' in text
    assert 'load_tokenizer(' in text
    assert 'user/mouse-example-tokenizer' in text
    assert 'separate' in text.lower()
    assert 'token_batch.grouper' not in text
    assert 'boundary_values' not in text
    assert 'Backbone: `identity`' in text
    assert 'Heads: `action_value`' in text


def _lora_model(dtype: torch.dtype) -> Model:
    hidden_dim = 8
    backbone = TransformerBackbone(architecture="qwen3", train_kernel="reference", decode_kernel="flex", dtype=dtype, use_norm=True, hidden_dim=hidden_dim, num_layers=1, num_heads=2, lora=LoRAConfig(rank=2), vocab_size=32)
    heads = RegressionHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, use_norm=True)
    return Model(backbone=backbone, heads=heads, action_source="action_value", reasoner=None).eval()


def test_backbone_dtype_applies_to_base_only_and_trainable_sections_stay_float32() -> None:
    model = _lora_model(torch.bfloat16)
    assert {p.dtype for p in model.heads.parameters()} == {torch.float32}
    assert {p.dtype for p in model.backbone.parameters() if not p.requires_grad} == {torch.bfloat16}
    assert {p.dtype for p in model.backbone.parameters() if p.requires_grad} == {torch.float32}
    assert model.backbone.dtype == torch.bfloat16
    batch = [[{'action': 0, 'reward': 0.5, 'episode_done': 0, 'task_done': 0}, {'action': 1, 'reward': 1.0, 'episode_done': 0, 'task_done': 0}]]
    with torch.no_grad():
        preds = model(batch_to_token_batch(_TOK, batch)).predictions
    assert preds['action_value'].dtype == torch.float32


@pytest.mark.parametrize(
    "cast_fn",
    [
        lambda m: m.to(torch.bfloat16),
        lambda m: m.to(dtype=torch.bfloat16),
        lambda m: m.to("cpu", torch.bfloat16),
        lambda m: m.to(device="cpu", dtype=torch.bfloat16),
        lambda m: m.to(torch.zeros(1, dtype=torch.bfloat16)),
        lambda m: m.bfloat16(),
        lambda m: m.half(),
        lambda m: m.double(),
        lambda m: m.float(),
        lambda m: m.backbone.to(dtype=torch.bfloat16),
        lambda m: m.backbone.bfloat16(),
    ],
    ids=["pos-dtype", "kw-dtype", "pos-device-dtype", "kw-device-dtype", "tensor", "bfloat16()", "half()", "double()", "float()", "backbone-to", "backbone-bfloat16()"],
)
def test_model_and_backbone_refuse_dtype_casts(cast_fn) -> None:
    """dtype is a backbone constructor argument; ``.to`` only moves."""
    model = _lora_model(torch.float32)
    with pytest.raises(TypeError, match="dtype"):
        cast_fn(model)
    assert model.backbone.dtype == torch.float32
    model.to("cpu")  # device-only moves still work
    model.to(device="cpu")


def test_load_model_casts_saved_weights_into_requested_dtype(tmp_path) -> None:
    model = _lora_model(torch.bfloat16)
    save_model(model=model, path=tmp_path)
    loaded = load_model(repo_id_or_path=tmp_path, train_kernel="reference", decode_kernel="flex", dtype=torch.float32)
    assert loaded.backbone.dtype == torch.float32
    assert {p.dtype for p in loaded.backbone.parameters()} == {torch.float32}
    saved: dict[str, torch.Tensor] = {n: cast(torch.Tensor, p) for n, p in model.backbone.named_parameters()}
    for n, p in loaded.backbone.named_parameters():
        if not p.requires_grad:
            assert torch.equal(cast(torch.Tensor, p), saved[n].float())


def test_model_to_cuda_moves_without_casting() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    model = _lora_model(torch.bfloat16).to(torch.device('cuda'))
    assert model.backbone.dtype == torch.bfloat16
    assert {p.dtype for p in model.backbone.parameters() if p.requires_grad} == {torch.float32}
    assert next(model.heads.parameters()).dtype == torch.float32
    batch = [[{'action': 0, 'reward': 0.0, 'episode_done': 0, 'task_done': 0}]]
    with torch.no_grad():
        preds = model(batch_to_token_batch(_TOK, batch), use_cache=True).predictions
    assert preds['action_value'].dtype == torch.float32


def test_save_model_does_not_write_tokenizer(tmp_path) -> None:
    model = Model(
        backbone=IdentityBackbone(hidden_dim=8, vocab_size=32),
        heads=(head := RegressionHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True)),
        action_source="action_value",
        reasoner=None,
    )
    save_model(model=model, path=tmp_path)
    assert not (tmp_path / "tokenizer.json").exists()


def test_tokenizer_roundtrip(tmp_path) -> None:
    """save_tokenizer writes tokenizer.json; load_tokenizer is the recall path."""
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "token", "input_field": "action"},
            {"type": "token", "input_field": "episode_done", "required": False},
            {"type": "token", "input_field": "done_code", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when_field": "step_index",
                "when_equals": 0,
            },
        ],
        grouping_field="task_index",
        objective_fields=[
            {"input_field": "action"},
            {"input_field": "reward", "output_field": "r"},
            {"input_field": "episode_done"},
            {"input_field": "task_done"},
        ],
    )
    save_tokenizer(tokenizer=tokenizer, path=tmp_path)
    assert (tmp_path / "tokenizer.json").is_file()
    loaded = load_tokenizer(repo_id_or_path=str(tmp_path))
    assert loaded.grouping_field == "task_index"
    assert loaded.group_prefix is None
    assert loaded.pretrained is None
    assert loaded.objective_fields == (
        ("action", "action"),
        ("reward", "r"),
        ("episode_done", "episode_done"),
        ("task_done", "task_done"),
    )
    assert [s.input_field for s in loaded.input_fields] == [
        "action",
        "episode_done",
        "done_code",
        "episode_index",
    ]
    assert loaded.input_fields[1].required is False
    assert loaded.input_fields[2].head_output is True
    assert loaded.input_fields[3].when_field == "step_index"
    assert loaded.input_fields[3].when_equals == 0
    assert sum(1 for s in loaded.input_fields if s.head_output) == 1


def test_load_tokenizer_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        load_tokenizer(repo_id_or_path=str(tmp_path))


def test_push_model_to_hub_requires_distinct_tokenizer_repo() -> None:
    model = Model(
        backbone=IdentityBackbone(hidden_dim=8, vocab_size=32),
        heads=(head := RegressionHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True)),
        action_source="action_value",
        reasoner=None,
    )
    tokenizer = Tokenizer(
        input_fields=[{"type": "token", "input_field": "action", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when_field": "step_index",
                "when_equals": 0,
            },
        ],
        grouping_field="task_index",
        objective_fields=[],
    )
    with pytest.raises(ValueError, match="tokenizer_repo_id"):
        push_model_to_hub(
            model=model,
            tokenizer=tokenizer,
            repo_id="same-id",
            tokenizer_repo_id="same-id",
        )
    with pytest.raises(TypeError, match="tokenizer_repo_id"):
        push_model_to_hub(  # type: ignore[call-arg]
            model=model,
            tokenizer=tokenizer,
            repo_id="my-model",
        )
