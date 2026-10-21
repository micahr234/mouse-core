from __future__ import annotations
from tests._model_helpers import fresh_model, ExampleModel as Model
from tests._model_helpers import simple_model
from typing import Any, cast
import json
import pytest
import torch
from mouse_core.models import Embedding, load_model, push_model_to_hub, save_model
from mouse_core.models.io import _write_model_card
from mouse_core.models.backbone import IdentityBackbone, LoRAConfig, Backbone
from mouse_core.data import Tokenizer, load_tokenizer, save_tokenizer
from mouse_core.models.mlp import MLP

def when_repeat_episode_done_nonzero(ctx):
    return "repeat_episode_done" in ctx and ctx["repeat_episode_done"] != 0


def when_group_start(ctx):
    return bool(ctx["group_start"])


def when_reward_nonzero(ctx):
    return "reward" in ctx and ctx["reward"] != 0.0


def when_step_index_zero(ctx):
    return "step_index" in ctx and ctx["step_index"] == 0


def when_step_index_zero_or_group_start(ctx):
    return (ctx.get("step_index") == 0) | bool(ctx["group_start"])

from tests._token_batch_helpers import (
    IntIdTokenizer,
    batch_to_token_batch,
    token_tokenizer,
)

_TOK = token_tokenizer("action", "repeat_episode_done")


def test_composed_model_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)
    model = simple_model(backbone=backbone, heads={'action_value': heads}, reasoner=None).eval()
    batch = [[{'action': 0, 'reward': 0.0, 'repeat_episode_done': 0, 'episode_done': 0}, {'action': 1, 'reward': 1.0, 'repeat_episode_done': 0, 'episode_done': 0}, {'action': 2, 'reward': 2.0, 'repeat_episode_done': 1, 'episode_done': 0}]]
    expected = model(batch_to_token_batch(_TOK, batch))["predictions"]
    save_model(model=model, path=tmp_path)
    loaded = load_model(model=fresh_model(model), repo_id_or_path=tmp_path).eval()
    actual = loaded(batch_to_token_batch(_TOK, batch))["predictions"]
    assert torch.allclose(actual['action_value'], expected['action_value'])
    assert loaded.backbone.hidden_dim == hidden_dim
    orig_sd = model.state_dict()
    loaded_sd = loaded.state_dict()
    assert set(orig_sd.keys()) == set(loaded_sd.keys()), f'state_dict key mismatch:\n  missing in loaded: {set(orig_sd.keys()) - set(loaded_sd.keys())}\n  extra in loaded:   {set(loaded_sd.keys()) - set(orig_sd.keys())}'
    for key in orig_sd:
        assert torch.equal(orig_sd[key], loaded_sd[key]), f'param {key!r} differs after save/load roundtrip'
    with (tmp_path / 'config.json').open() as fh:
        config = json.load(fh)
    assert config['format'] == 'mouse-core-model-v3'
    assert config['class'].endswith('ExampleModel')


def test_loading_requires_a_model_and_preserves_its_runtime(tmp_path) -> None:
    source = simple_model(
        backbone=Backbone(architecture='qwen3', train_kernel='reference', decode_kernel='flex', dtype=torch.float32, hidden_dim=8, num_layers=1, num_heads=2),
        heads={'action_value': MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.)},
         reasoner=None,
    )
    save_model(model=source, path=tmp_path)
    target = simple_model(
        backbone=Backbone(architecture='qwen3', train_kernel='flex', decode_kernel='flex', dtype=torch.float32, hidden_dim=8, num_layers=1, num_heads=2),
        heads={'action_value': MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.)},
         reasoner=None,
    )
    loaded = load_model(model=target, repo_id_or_path=tmp_path)
    assert loaded is target
    assert loaded.backbone.train_kernel == "flex"
    assert next(loaded.parameters()).requires_grad
    with pytest.raises(TypeError, match="model"):
        load_model(repo_id_or_path=tmp_path)  # type: ignore[call-arg]



def test_num_frequencies_roundtrip(tmp_path) -> None:
    embedding = Embedding(hidden_dim=8, vocab_size=32, num_frequencies=3)
    hidden_dim = 8
    backbone = Backbone(architecture='qwen3', train_kernel='reference', decode_kernel='flex', dtype=torch.float32, hidden_dim=hidden_dim, num_layers=1, num_heads=2)
    assert embedding.fourier is not None
    with torch.no_grad():
        embedding.fourier.proj.weight.fill_(0.25)
    heads = MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)
    model = simple_model(embedding=embedding, backbone=backbone, heads={'action_value': heads}, reasoner=None)
    save_model(model=model, path=tmp_path)
    loaded_model = load_model(model=fresh_model(model), repo_id_or_path=tmp_path)
    loaded = loaded_model.embedding
    assert loaded_model.embedding.num_frequencies == 3
    assert loaded.fourier is not None
    assert model.embedding.fourier is not None
    assert torch.equal(loaded.fourier.proj.weight, model.embedding.fourier.proj.weight)
    assert not any((key.endswith('bands') for key in model.state_dict()))


def test_fourier_stays_trainable_when_lora_freezes_the_stack() -> None:
    embedding = Embedding(hidden_dim=8, vocab_size=32, num_frequencies=2)
    backbone = Backbone(architecture='qwen3', train_kernel='reference', decode_kernel='flex', dtype=torch.float32, hidden_dim=8, num_layers=1, num_heads=2, lora=LoRAConfig(rank=2))
    assert embedding.fourier is not None
    assert embedding.fourier.proj.weight.requires_grad
    assert any(not param.requires_grad for param in backbone.model.parameters())


def test_identity_num_frequencies_roundtrip(tmp_path) -> None:
    embedding = Embedding(hidden_dim=8, vocab_size=32, num_frequencies=4)
    backbone = IdentityBackbone(hidden_dim=8)
    assert embedding.fourier is not None
    with torch.no_grad():
        embedding.fourier.proj.weight.normal_()
    heads = MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0)
    model = simple_model(embedding=embedding, backbone=backbone, heads={'action_value': heads}, reasoner=None)
    save_model(model=model, path=tmp_path)
    loaded_model = load_model(model=fresh_model(model), repo_id_or_path=tmp_path)
    loaded = loaded_model.embedding
    assert loaded_model.embedding.num_frequencies == 4
    assert loaded.fourier is not None
    assert model.embedding.fourier is not None
    assert torch.equal(loaded.fourier.proj.weight, model.embedding.fourier.proj.weight)


def test_explicit_model_checkpoints_roundtrip_without_backbone_output_norm(tmp_path) -> None:
    backbone = Backbone(architecture='qwen3', train_kernel='reference', decode_kernel='flex', dtype=torch.float32, hidden_dim=8, num_layers=1, num_heads=2)
    head = MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0)
    model = simple_model(backbone=backbone, heads={'action_value': head}, reasoner=None)
    with torch.no_grad():
        head.layers[-1].weight.fill_(3.0)
    save_model(model=model, path=tmp_path)
    loaded = load_model(model=fresh_model(model), repo_id_or_path=str(tmp_path))
    assert isinstance(cast(Backbone, loaded.backbone).model.norm, torch.nn.Identity)
    torch.testing.assert_close(cast(MLP, loaded.heads['action_value']).layers[-1].weight, torch.full((4, 8), 3.0))
    h = torch.randn(3, 8)
    torch.testing.assert_close(loaded.head(h=h)['action_value'], model.head(h=h)['action_value'])



@pytest.mark.parametrize("version", ["mouse-core-model-v1", "mouse-core-model-v2"])
def test_legacy_checkpoint_formats_are_rejected(tmp_path, version: str) -> None:
    model = _lora_model(torch.float32)
    save_model(model=model, path=tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["format"] = version
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unsupported model config format"):
        load_model(model=fresh_model(model), repo_id_or_path=tmp_path)



def test_mlp_defaults_to_full_input_gradients() -> None:
    mlp = MLP(in_features=8, out_features=1, hidden_dim=8, num_layers=1)
    h = torch.randn(2, 8, requires_grad=True)
    mlp(h).sum().backward()
    assert mlp.propagate_gradient == 1.0
    assert h.grad is not None and h.grad.abs().sum() > 0


def test_transformer_backbone_requires_kernels_and_dtype() -> None:
    with pytest.raises(TypeError, match="train_kernel.*decode_kernel.*dtype"):
        Backbone(architecture="qwen3", hidden_dim=8, num_layers=1, num_heads=2)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="train_kernel"):
        Backbone(architecture="qwen3", train_kernel=cast(Any, "sdpa"), decode_kernel="flex", dtype=torch.float32, hidden_dim=8, num_layers=1, num_heads=2)
    with pytest.raises(ValueError, match="decode_kernel"):
        Backbone(architecture="qwen3", train_kernel="reference", decode_kernel=cast(Any, "sdpa"), dtype=torch.float32, hidden_dim=8, num_layers=1, num_heads=2)
    with pytest.raises(TypeError, match="dtype"):
        Backbone(architecture="qwen3", train_kernel="reference", decode_kernel="flex", dtype=torch.int8, hidden_dim=8, num_layers=1, num_heads=2)


def test_identity_embed_tokens_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)
    model = simple_model(backbone=backbone, heads={'action_value': heads}, reasoner=None).eval()
    tokenizer = token_tokenizer('action', 'prev_action')
    batch = [[{'action': 0, 'prev_action': 1, 'reward': 0.5, 'grouping_id': 0}, {'action': 2, 'prev_action': 0, 'reward': 1.0, 'grouping_id': 0}]]
    expected = model(batch_to_token_batch(tokenizer, batch))["predictions"]
    save_model(model=model, path=tmp_path)
    loaded = load_model(model=fresh_model(model), repo_id_or_path=tmp_path).eval()
    assert set(model.state_dict()) == set(loaded.state_dict())
    actual = loaded(batch_to_token_batch(tokenizer, batch))["predictions"]
    assert torch.allclose(actual['action_value'], expected['action_value'])
    assert torch.equal(model.embedding.embed_tokens.weight, loaded.embedding.embed_tokens.weight)


def test_model_card_requires_the_application_implementation(tmp_path) -> None:
    model = _lora_model(torch.float32)
    path = tmp_path / "README.md"
    _write_model_card(repo_id="user/model", model=model, path=path)
    text = path.read_text()
    assert "library_name: mouse-core" in text
    assert "ExampleModel" in text
    assert "load_model(model=model," in text
    assert "No default architecture" in text
    assert "user/model" in text and "stored in this repository" in text
    assert "Optimizer state and delayed models" in text



def _lora_model(dtype: torch.dtype) -> Model:
    hidden_dim = 8
    backbone = Backbone(architecture='qwen3', train_kernel='reference', decode_kernel='flex', dtype=dtype, hidden_dim=hidden_dim, num_layers=1, num_heads=2, lora=LoRAConfig(rank=2))
    heads = MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)
    return simple_model(backbone=backbone, heads={'action_value': heads}, reasoner=None).eval()


def test_backbone_dtype_applies_to_base_only_and_trainable_sections_stay_float32() -> None:
    model = _lora_model(torch.bfloat16)
    assert {p.dtype for p in model.heads.parameters()} == {torch.float32}
    assert {p.dtype for p in model.backbone.parameters() if not p.requires_grad} == {torch.bfloat16}
    assert {p.dtype for p in model.backbone.parameters() if p.requires_grad} == {torch.float32}
    assert model.backbone.dtype == torch.bfloat16
    batch = [[{'action': 0, 'reward': 0.5, 'repeat_episode_done': 0, 'episode_done': 0}, {'action': 1, 'reward': 1.0, 'repeat_episode_done': 0, 'episode_done': 0}]]
    with torch.no_grad():
        preds = model(batch_to_token_batch(_TOK, batch))["predictions"]
    assert preds['action_value'].dtype == torch.float32


@pytest.mark.parametrize('cast_fn', [lambda m: m.to(torch.bfloat16), lambda m: m.to(dtype=torch.bfloat16), lambda m: m.to('cpu', torch.bfloat16), lambda m: m.to(device='cpu', dtype=torch.bfloat16), lambda m: m.to(torch.zeros(1, dtype=torch.bfloat16)), lambda m: m.bfloat16(), lambda m: m.half(), lambda m: m.double(), lambda m: m.float(), lambda m: m.backbone.to(dtype=torch.bfloat16), lambda m: m.backbone.bfloat16()], ids=['pos-dtype', 'kw-dtype', 'pos-device-dtype', 'kw-device-dtype', 'tensor', 'bfloat16()', 'half()', 'double()', 'float()', 'backbone-to', 'backbone-bfloat16()'])
def test_model_and_backbone_refuse_dtype_casts(cast_fn) -> None:
    """dtype is a backbone constructor argument; ``.to`` only moves."""
    model = _lora_model(torch.float32)
    with pytest.raises(TypeError, match='dtype'):
        cast_fn(model)
    assert model.backbone.dtype == torch.float32
    model.to('cpu')
    model.to(device='cpu')


def test_load_model_casts_saved_weights_into_the_supplied_models_dtype(tmp_path) -> None:
    model = _lora_model(torch.bfloat16)
    save_model(model=model, path=tmp_path)
    loaded = load_model(model=_lora_model(torch.float32), repo_id_or_path=tmp_path)
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
    batch = [[{'action': 0, 'reward': 0.0, 'repeat_episode_done': 0, 'episode_done': 0}]]
    with torch.no_grad():
        preds = model(batch_to_token_batch(_TOK, batch), use_cache=True)["predictions"]
    assert preds['action_value'].dtype == torch.float32


def test_save_model_does_not_write_tokenizer(tmp_path) -> None:
    model = simple_model(backbone=IdentityBackbone(hidden_dim=8), heads=({'action_value': (head := MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0))}), reasoner=None)
    save_model(model=model, path=tmp_path)
    assert not (tmp_path / 'tokenizer.json').exists()


def test_tokenizer_roundtrip(tmp_path) -> None:
    """save_tokenizer writes tokenizer.json; load_tokenizer is the recall path."""
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}"},
            {"type": "text", "input_field": "repeat_episode_done", "format": "{field}", "required": False},
            {"type": "text", "input_field": "done_code", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[
            {"input_field": "action"},
            {"input_field": "reward", "output_field": "r"},
            {"input_field": "repeat_episode_done"},
            {"input_field": "episode_done"},
        ],
        tokenizer=IntIdTokenizer(),
    )
    save_tokenizer(tokenizer=tokenizer, path=tmp_path)
    assert (tmp_path / "tokenizer.json").is_file()
    loaded = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizer=IntIdTokenizer())
    assert loaded.pretrained is None
    assert loaded.objective_fields == (
        ("action", "action"),
        ("reward", "r"),
        ("repeat_episode_done", "repeat_episode_done"),
        ("episode_done", "episode_done"),
    )
    assert [s.input_field for s in loaded.input_fields] == [
        "action",
        "repeat_episode_done",
        "done_code",
        "repeat_episode_index",
    ]
    assert loaded.input_fields[1].required is False
    assert loaded.input_fields[2].head_output is True
    assert loaded.input_fields[3].when is when_step_index_zero
    assert sum(1 for s in loaded.input_fields if s.head_output) == 1


def test_tokenizer_roundtrip_group_start_and_not_equals(tmp_path) -> None:
    class _Tok:
        def __call__(self, text: str, add_special_tokens: bool = False):
            return {"input_ids": [1, 2]}

    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "format": "hello\n",
                "when": when_group_start,
            },
            {
                "type": "text",
                "input_field": "reward",
                "format": "{field}",
                "when": when_reward_nonzero,
            },
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
        ],
        tokenizer=_Tok(),
    )
    save_tokenizer(tokenizer=tokenizer, path=tmp_path)
    loaded = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizer=_Tok())
    assert loaded.input_fields[0].when is when_group_start
    assert loaded.input_fields[0].input_field is None
    assert loaded.input_fields[0].format == "hello\n"
    assert loaded.input_fields[1].when is when_reward_nonzero
    st = loaded({"action": 1, "reward": 1.0, "episode_index": 0})
    assert st.group_start_token_ids is not None
    assert st.group_start_token_ids.tolist() == [1, 2]


def test_load_tokenizer_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        load_tokenizer(repo_id_or_path=str(tmp_path))


def test_push_model_and_tokenizer_share_repository(monkeypatch) -> None:
    model = simple_model(backbone=IdentityBackbone(hidden_dim=8), heads=({'action_value': (head := MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0))}), reasoner=None)
    tokenizer = Tokenizer(input_fields=[{'type': 'text', 'input_field': 'action', 'format': '{field}', 'head_output': True}, {'type': 'text', 'input_field': 'repeat_episode_index', 'format': '{field}', 'when': when_step_index_zero}], objective_fields=[], tokenizer=IntIdTokenizer())
    import mouse_core.models.io as io
    import huggingface_hub
    from pathlib import Path

    created, uploads = [], []
    def create(**kwargs):
        created.append(kwargs)
        return "https://huggingface.co/user/model", "user/model"
    class Api:
        def upload_folder(self, **kwargs):
            directory = Path(kwargs["folder_path"])
            assert {"config.json", "pytorch_model.bin", "tokenizer.json", "README.md"} <= {p.name for p in directory.iterdir()}
            load_model(model=model, repo_id_or_path=directory)
            loaded = load_tokenizer(repo_id_or_path=str(directory), tokenizer=IntIdTokenizer())
            assert loaded is not None
            uploads.append(kwargs)
    monkeypatch.setattr(io, "_create_hub_repo", create)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    result = push_model_to_hub(model=model, tokenizer=tokenizer, repo_id="user/model")
    assert result == "https://huggingface.co/user/model"
    assert len(created) == len(uploads) == 1
    assert uploads[0]["repo_id"] == "user/model"
