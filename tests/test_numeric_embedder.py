from __future__ import annotations
import pytest
import torch
from mouse_core.models.embedding import NumericEmbedder

def _enc(**kwargs) -> NumericEmbedder:
    return NumericEmbedder(**kwargs)

def _batch(rows: list[dict], S: int=1) -> list[list[dict]]:
    assert len(rows) == S
    return [rows]

def test_numeric_embedder_ignores_is_seam_in_row_dicts() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}])
    with_seam = [[{'action': 0, 'is_seam': 0}, {'action': 1, 'is_seam': 1}]]
    without_seam = [[{'action': 0}, {'action': 1}]]
    embeds, col_values, _ = encoder(with_seam)
    plain_embeds, plain_col_values, _ = encoder(without_seam)
    assert 'is_seam' not in col_values
    assert 'is_seam' not in plain_col_values
    assert torch.equal(embeds, plain_embeds)

def test_numeric_embedder_faults_on_missing_required_modality() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff'}])
    batch = _batch([{'reward': 0.5}])
    with pytest.raises(KeyError, match="Required modality 'action' is missing"):
        encoder(batch)

def test_numeric_embedder_keeps_optional_missing_modality() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4, 'required': False}, {'field': 'reward', 'type': 'rff'}])
    batch = _batch([{'reward': 0.5}])
    embeds, _, prediction_indices = encoder(batch)
    assert embeds.shape == (1, 8)
    assert prediction_indices.shape == (1,)

def test_numeric_embedder_returns_col_values() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff'}])
    batch = _batch([{'action': 2, 'reward': 1.5}])
    embeds, col_values, prediction_indices = encoder(batch)
    assert embeds.shape == (2, 8)
    assert prediction_indices.tolist() == [1]
    assert col_values['action'].item() == 2
    assert col_values['reward'].item() == pytest.approx(1.5)

def test_numeric_embedder_expands_multi_field_modality_specs() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': ('action', 'prev_action'), 'type': 'discrete', 'vocab_size': 4}, {'field': ('reward', 'value'), 'type': 'rff'}])
    batch = _batch([{'action': 2, 'prev_action': 1, 'reward': 1.5, 'value': 0.25}])
    embeds, col_values, _ = encoder(batch)
    assert embeds.shape == (4, 8)
    assert [spec.field for spec in encoder.modalities] == ['action', 'prev_action', 'reward', 'value']
    assert col_values['action'].item() == 2
    assert col_values['prev_action'].item() == 1

def test_numeric_embedder_batch_shape() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff'}])
    B, S = (3, 5)
    batch = [[{'action': (b * S + s) % 4, 'reward': float(b * S + s)} for s in range(S)] for b in range(B)]
    embeds, col_values, prediction_indices = encoder(batch)
    assert embeds.shape == (B * S * 2, 8)
    assert col_values['action'].shape == (B * S,)
    assert col_values['reward'].shape == (B * S,)
    assert prediction_indices.shape == (B * S,)

def test_numeric_embedder_concat_tokens_in_order() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff'}, {'type': 'learnable', 'tokens': 1}])
    embeds, _, prediction_indices = encoder(_batch([{'action': 2, 'reward': 1.5}]))
    assert embeds.shape == (3, 8)
    assert prediction_indices.tolist() == [2]

def test_numeric_embedder_rejects_unknown_constructor_kwargs() -> None:
    with pytest.raises(TypeError):
        _enc(hidden_dim=8, modality_fusion='sum', modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}])
    with pytest.raises(TypeError):
        NumericEmbedder(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}], include_type_token=True)

def test_numeric_embedder_learnable_modality_is_allowed() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'type': 'learnable', 'tokens': 1}])
    embeds, col_values, _ = encoder([[{}]])
    assert embeds.shape == (1, 8)
    assert 'scratch' not in col_values

def test_numeric_embedder_continuous_one_token_per_scalar() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'obs', 'type': 'continuous', 'dim': 4}, {'field': 'reward', 'type': 'rff'}])
    batch = [[{'obs': [0.1, 0.2, 0.3, 0.4], 'reward': 1.0}]]
    embeds, col_values, prediction_indices = encoder(batch)
    assert embeds.shape == (5, 8)
    assert col_values['obs'].shape == (1, 4)
    assert prediction_indices.tolist() == [4]

def test_numeric_embedder_skip_shortens_step() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff', 'skip': 0.0}, {'type': 'learnable', 'tokens': 1}])
    batch = [[{'action': 1, 'reward': 0.0}, {'action': 2, 'reward': 1.5}]]
    embeds, col_values, indices = encoder(batch)
    assert col_values['reward'].tolist() == [0.0, 1.5]
    assert embeds.shape == (5, 8)
    assert indices.tolist() == [1, 4]

def test_numeric_embedder_image_requires_tokenizer() -> None:
    with pytest.raises(TypeError, match='image_tokenizer'):
        _enc(hidden_dim=8, modalities=[{'field': 'img', 'type': 'image', 'vocab_size': 32}])

def test_numeric_embedder_prepare_token_batch() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff'}])
    batch = [[{'action': 1, 'reward': 0.5}, {'action': 2, 'reward': 1.0}]]
    tb = encoder.prepare(batch)
    assert tb.B == 1 and int(tb.step_counts()[0]) == 2
    assert tb.L == 4
    assert list(tb.sequence_ids) == [0, 0, 0, 0]
    embeds, _, prediction_indices = encoder(tb)
    assert embeds.shape == (4, 8)
    assert prediction_indices.shape == (2,)

def test_static_fourier_no_parameters() -> None:
    from mouse_core.models.embedding import StaticFourierFeatures
    ff = StaticFourierFeatures(num_features=8, in_min=0.01, in_max=10.0)
    assert sum((p.numel() for p in ff.parameters())) == 0
    y = ff(torch.tensor([0.5, -0.5]))
    assert y.shape == (2, 8)
