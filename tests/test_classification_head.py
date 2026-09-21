from __future__ import annotations
import json
import pytest
import torch
from mouse_core.models import Model, load_model, save_model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.base import Model as ModelClass
from mouse_core.data import Tokenizer
from mouse_core.models.heads import ClassificationHead
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

_TOK = token_tokenizer("action")

def test_classification_head_forward_shape() -> None:
    head = ClassificationHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True)
    out = head(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 4)

def test_infer_head_name_is_action() -> None:
    head = ClassificationHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True)
    assert ModelClass._infer_head_name(head) == 'action'


def test_action_source_must_be_an_enabled_name() -> None:
    hidden_dim = 8
    enabled = ClassificationHead(
        in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, use_norm=True
    )
    with pytest.raises(ValueError, match="not among heads"):
        Model(
            backbone=IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32),
            heads=enabled,
            action_source="action_value",
            reasoner=None,
        )

def test_classification_head_save_load_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    head = ClassificationHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, use_norm=True)
    model = Model(backbone=IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32), heads=head, action_source="action", reasoner=None).eval()
    batch = [[{'action': 0, 'reward': 0.0}, {'action': 1, 'reward': 1.0}]]
    expected = model(batch_to_token_batch(_TOK, batch)).predictions
    save_model(model=model, path=tmp_path)
    loaded = load_model(repo_id_or_path=tmp_path, train_kernel="reference", decode_kernel="flex", dtype=torch.float32).eval()
    actual = loaded(batch_to_token_batch(_TOK, batch)).predictions
    assert torch.allclose(actual['action'], expected['action'])
    assert loaded.action_source == 'action'
    with (tmp_path / 'config.json').open() as fh:
        config = json.load(fh)
    assert config['heads']['action_source'] == 'action'
    head_specs = config['heads']['heads']
    assert len(head_specs) == 1
    assert head_specs[0]['type'] == 'classification'
