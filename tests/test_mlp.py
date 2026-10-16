from __future__ import annotations
from tests._model_helpers import fresh_model, ExampleModel as Model
from tests._model_helpers import simple_model
import json
import torch
from mouse_core.models import load_model, save_model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.mlp import MLP
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

_TOK = token_tokenizer("action")

def test_mlp_preserves_batch_dimensions() -> None:
    head = MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0)
    out = head(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 4)

def test_output_name_is_explicit() -> None:
    head = MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.)
    model = simple_model(backbone=IdentityBackbone(hidden_dim=8), heads={"custom_action": head}, reasoner=None)
    assert tuple(model.heads) == ("custom_action",)

def test_mlp_prediction_save_load_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    head = MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)
    model = simple_model(backbone=IdentityBackbone(hidden_dim=hidden_dim), heads={'action': head}, reasoner=None).eval()
    batch = [[{'action': 0, 'reward': 0.0}, {'action': 1, 'reward': 1.0}]]
    expected = model(batch_to_token_batch(_TOK, batch))["predictions"]
    save_model(model=model, path=tmp_path)
    loaded = load_model(model=fresh_model(model), repo_id_or_path=tmp_path).eval()
    actual = loaded(batch_to_token_batch(_TOK, batch))["predictions"]
    assert torch.allclose(actual['action'], expected['action'])
    with (tmp_path / 'config.json').open() as fh:
        config = json.load(fh)
    assert config['class'].endswith('ExampleModel')
