from __future__ import annotations

from typing import cast

import pytest
import torch
import torch.nn as nn

from mouse_core.data import Tokenizer
from mouse_core.models.embedding import NumericEmbedder
from tests._token_batch_helpers import batch_to_packed, batch_to_token_batch, tok_from_encoder

_tok = tok_from_encoder


def _enc(**kwargs) -> NumericEmbedder:
    return NumericEmbedder(**kwargs)


def _batch(rows: list[dict], S: int = 1) -> list[list[dict]]:
    assert len(rows) == S
    return [rows]


def _tb(encoder, batch):
    return batch_to_token_batch(_tok(encoder), batch)


def test_numeric_embedder_ignores_is_seam_in_row_dicts() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}])
    with_seam = [[{"action": 0, "is_seam": 0}, {"action": 1, "is_seam": 1}]]
    without_seam = [[{"action": 0}, {"action": 1}]]
    embeds, _ = encoder(_tb(encoder, with_seam))
    plain_embeds, _ = encoder(_tb(encoder, without_seam))
    _, with_obj = batch_to_packed(_tok(encoder), with_seam)
    _, without_obj = batch_to_packed(_tok(encoder), without_seam)
    assert "is_seam" not in with_obj.keys()
    assert "is_seam" not in without_obj.keys()
    assert torch.equal(embeds, plain_embeds)


def test_numeric_embedder_faults_on_missing_required_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
        ],
    )
    batch = _batch([{"reward": 0.5}])
    with pytest.raises(KeyError, match="Required input field 'action' is missing"):
        _tb(encoder, batch)


def test_numeric_embedder_keeps_optional_missing_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
        ],
    )
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "required": False,
            },
            {
                "type": "fourier",
                "input_field": "reward",
                "output_field": "reward",
                "head_output": True,
            },
        ],
        grouping_field="grouping_id",
    )
    batch = _batch([{"reward": 0.5}])
    embeds, head_output_indices = encoder(batch_to_token_batch(tokenizer, batch))
    assert embeds.shape == (1, 8)
    assert head_output_indices.shape == (1,)


def test_numeric_embedder_returns_objective_fields() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
        ],
    )
    batch = _batch([{"action": 2, "reward": 1.5}])
    tb, obj = batch_to_packed(_tok(encoder), batch)
    embeds, head_output_indices = encoder(tb)
    assert embeds.shape == (2, 8)
    assert head_output_indices.tolist() == [1]
    assert obj["action"].item() == 2
    assert obj["reward"].item() == pytest.approx(1.5)


def test_numeric_embedder_expands_multi_field_modality_specs() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": ("action", "prev_action"), "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": "fourier", "field": ("reward", "value"), "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
        ],
    )
    batch = _batch(
        [{"action": 2, "prev_action": 1, "reward": 1.5, "value": 0.25}]
    )
    tb, obj = batch_to_packed(_tok(encoder), batch)
    embeds, _ = encoder(tb)
    assert embeds.shape == (4, 8)
    assert [spec.field for spec in encoder.modalities] == [
        "action",
        "prev_action",
        "reward",
        "value",
    ]
    assert obj["action"].item() == 2
    assert obj["prev_action"].item() == 1


def test_numeric_embedder_batch_shape() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
        ],
    )
    B, S = (3, 5)
    batch = [
        [{"action": (b * S + s) % 4, "reward": float(b * S + s)} for s in range(S)]
        for b in range(B)
    ]
    tb, obj = batch_to_packed(_tok(encoder), batch)
    embeds, head_output_indices = encoder(tb)
    assert embeds.shape == (B * S * 2, 8)
    assert obj["action"].shape == (B * S,)
    assert obj["reward"].shape == (B * S,)
    assert head_output_indices.shape == (B * S,)


def test_numeric_embedder_concat_tokens_in_order() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
            {"type": "learnable", "tokens": 1, "std": 0.02, "positions": 1},
        ],
    )
    embeds, head_output_indices = encoder(
        _tb(encoder, _batch([{"action": 2, "reward": 1.5}]))
    )
    assert embeds.shape == (3, 8)
    assert head_output_indices.tolist() == [2]


def test_numeric_embedder_rejects_unknown_constructor_kwargs() -> None:
    with pytest.raises(TypeError):
        _enc(
            hidden_dim=8,
            modality_fusion="sum",
            modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}],
        )
    with pytest.raises(TypeError):
        NumericEmbedder(
            hidden_dim=8,
            modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}],
            **{"include_type_token": True},
        )


def test_numeric_embedder_learnable_modality_is_allowed() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{"type": "learnable", "tokens": 1, "std": 0.02, "positions": 1}])
    tb, obj = batch_to_packed(_tok(encoder), [[{}]])
    embeds, _ = encoder(tb)
    assert embeds.shape == (1, 8)
    assert "scratch" not in obj.keys()


def test_numeric_embedder_continuous_one_token_per_scalar() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'continuous', "field": "obs", "dim": 4, "std": 0.02, "positions": 4, "fourier_min": 0.01, "fourier_max": 10.0},
            {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
        ],
    )
    batch = [[{"obs": [0.1, 0.2, 0.3, 0.4], "reward": 1.0}]]
    tb, obj = batch_to_packed(_tok(encoder), batch)
    embeds, head_output_indices = encoder(tb)
    assert embeds.shape == (5, 8)
    assert obj["obs"].shape == (1, 4)
    assert head_output_indices.tolist() == [4]


def test_numeric_embedder_skip_shortens_step() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
            {"type": "learnable", "tokens": 1, "std": 0.02, "positions": 1},
        ],
    )
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action", "output_field": "action"},
            {"type": "fourier", "input_field": "reward", "output_field": "reward", "skip": 0.0},
            {"type": "learnable", "tokens": 1, "head_output": True},
        ],
        objective_fields=[
            {"input_field": "action", "output_field": "action"},
            {"input_field": "reward", "output_field": "reward"},
        ],
        grouping_field="grouping_id",
    )
    batch = [[{"action": 1, "reward": 0.0}, {"action": 2, "reward": 1.5}]]
    tb, obj = batch_to_packed(tokenizer, batch)
    embeds, indices = encoder(tb)
    assert obj["reward"].tolist() == [0.0, 1.5]
    assert embeds.shape == (5, 8)
    assert indices.tolist() == [1, 4]


def test_numeric_tokenizer_image_requires_callable() -> None:
    with pytest.raises(TypeError, match="image_tokenizer"):
        Tokenizer(
            input_fields=[
                {"type": "image", "input_field": "img", "output_field": "img", "head_output": True}
            ],
            grouping_field="grouping_id",
        )
    enc = _enc(
        hidden_dim=8,
        modalities=[{"type": 'image', "field": "img", "vocab_size": 32, "std": 0.02, "positions": 16}],
    )
    assert enc.tokens_per_step == 16
    assert enc._type_vectors["img"].shape == (16, 8)


def test_numeric_embedder_prepare_token_batch() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
        ],
    )
    batch = [[{"action": 1, "reward": 0.5}, {"action": 2, "reward": 1.0}]]
    tb = _tb(encoder, batch)
    assert tb.B == 1 and int(tb.step_counts()[0]) == 2
    assert tb.L == 4
    assert list(tb.sequence_ids) == [0, 0, 0, 0]
    embeds, head_output_indices = encoder(tb)
    assert embeds.shape == (4, 8)
    assert head_output_indices.shape == (2,)


def test_numeric_embedder_requires_std_per_modality() -> None:
    with pytest.raises(ValueError, match="requires std="):
        _enc(hidden_dim=8, modalities=[{"type": "discrete", "field": "action", "vocab_size": 4}])
    with pytest.raises(ValueError, match="requires std="):
        _enc(hidden_dim=8, modalities=[{"type": "fourier", "field": "reward"}])
    with pytest.raises(ValueError, match="requires std="):
        _enc(hidden_dim=8, modalities=[{"type": "learnable", "tokens": 1}])
    with pytest.raises(ValueError, match="must be >= 0"):
        _enc(hidden_dim=8, modalities=[{"type": "fourier", "field": "reward", "std": -0.1, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}])


def test_numeric_embedder_requires_positions_per_modality() -> None:
    with pytest.raises(ValueError, match="requires positions="):
        _enc(hidden_dim=8, modalities=[{"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02}])
    with pytest.raises(ValueError, match="requires positions="):
        _enc(hidden_dim=8, modalities=[{"type": "learnable", "tokens": 1, "std": 0.02}])
    with pytest.raises(ValueError, match="positions must be >= 1"):
        _enc(hidden_dim=8, modalities=[{"type": "fourier", "field": "reward", "std": 0.02, "positions": 0, "fourier_min": 0.01, "fourier_max": 10.0}])
    with pytest.raises(ValueError, match="positions must be >= dim"):
        _enc(
            hidden_dim=8,
            modalities=[{"type": "continuous", "field": "obs", "dim": 3, "std": 0.02, "positions": 2, "fourier_min": 0.01, "fourier_max": 10.0}],
        )
    with pytest.raises(ValueError, match="positions must be >= tokens"):
        _enc(hidden_dim=8, modalities=[{"type": "learnable", "tokens": 2, "std": 0.02, "positions": 1}])
    # positions may exceed the emitted count (a max); the extra rows are unused.
    enc = _enc(
        hidden_dim=8,
        modalities=[{"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 3}],
    )
    assert enc._type_vectors["action"].shape == (3, 8)
    assert enc.tokens_per_step == 3
    with pytest.raises(TypeError):
        NumericEmbedder(
            hidden_dim=8,
            modalities=[{"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}],
            **{"std": 0.02},
        )
    with pytest.raises(TypeError):
        NumericEmbedder(
            hidden_dim=8,
            modalities=[{"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}],
            **{"fourier_min": 0.01},
        )


def test_numeric_embedder_requires_fourier_range_per_modality() -> None:
    with pytest.raises(ValueError, match="requires fourier_min="):
        _enc(
            hidden_dim=8,
            modalities=[{"type": "fourier", "field": "reward", "std": 0.02, "positions": 1}],
        )
    with pytest.raises(ValueError, match="requires fourier_min="):
        _enc(
            hidden_dim=8,
            modalities=[{"type": "continuous", "field": "obs", "dim": 2, "std": 0.02, "positions": 2}],
        )
    with pytest.raises(ValueError, match="must be > 0"):
        _enc(
            hidden_dim=8,
            modalities=[
                {
                    "type": "fourier",
                    "field": "reward",
                    "std": 0.02,
                    "positions": 1,
                    "fourier_min": 0.0,
                    "fourier_max": 10.0,
                }
            ],
        )
    with pytest.raises(ValueError, match="must be < fourier_max"):
        _enc(
            hidden_dim=8,
            modalities=[
                {
                    "type": "fourier",
                    "field": "reward",
                    "std": 0.02,
                    "positions": 1,
                    "fourier_min": 10.0,
                    "fourier_max": 0.01,
                }
            ],
        )
    with pytest.raises(TypeError, match="does not accept fourier_min"):
        _enc(
            hidden_dim=8,
            modalities=[
                {
                    "type": "discrete",
                    "field": "action",
                    "vocab_size": 4,
                    "std": 0.02,
                    "positions": 1,
                    "fourier_min": 0.01,
                    "fourier_max": 10.0,
                }
            ],
        )


def test_numeric_embedder_std_scales_tables_and_type_vectors() -> None:
    torch.manual_seed(0)
    encoder = _enc(
        hidden_dim=256,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 64, "std": 0.02, "positions": 1},
            {"type": "discrete", "field": "observation", "vocab_size": 64, "std": 0.5, "positions": 1},
        ],
    )
    for name, std in (("action", 0.02), ("observation", 0.5)):
        table = cast(nn.Embedding, encoder._tables[name])
        type_vec = cast(torch.Tensor, encoder._type_vectors[name])
        table_rms = float(table.weight.pow(2).mean().sqrt().item())
        type_rms = float(type_vec.pow(2).mean().sqrt().item())
        assert table_rms == pytest.approx(std, rel=0.1)
        assert type_rms == pytest.approx(std, rel=0.2)


def test_numeric_embedder_fourier_honors_per_modality_std() -> None:
    """Each Fourier field's embeddings are scaled by that field's ``std``."""
    encoder = _enc(
        hidden_dim=64,
        modalities=[
            {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
            {"type": "fourier", "field": "bonus", "std": 0.10, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
        ],
    )
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "fourier", "input_field": "reward"},
            {"type": "fourier", "input_field": "bonus", "head_output": True},
        ],
        grouping_field="grouping_id",
    )
    rewards = [float(x) for x in range(-4, 5)]
    batch = [[{"reward": r, "bonus": r} for r in rewards]]
    tb = batch_to_token_batch(tokenizer, batch)
    embeds, _ = encoder(tb)
    names = tokenizer.modality_names
    reward_emb = (
        embeds[torch.from_numpy(tb.modality_ids == names.index("reward"))]
        - encoder._type_vectors["reward"][0]
    )
    bonus_emb = (
        embeds[torch.from_numpy(tb.modality_ids == names.index("bonus"))]
        - encoder._type_vectors["bonus"][0]
    )
    reward_rms = float(reward_emb.pow(2).mean().sqrt().item())
    bonus_rms = float(bonus_emb.pow(2).mean().sqrt().item())
    assert reward_rms == pytest.approx(0.02, abs=0.008)
    assert bonus_rms == pytest.approx(0.10, abs=0.03)


def test_numeric_embedder_fourier_honors_per_modality_range() -> None:
    """Each Fourier field uses its own ``fourier_min`` / ``fourier_max`` bank."""
    from mouse_core.models.embedding import StaticFourierFeatures

    encoder = _enc(
        hidden_dim=32,
        modalities=[
            {
                "type": "fourier",
                "field": "reward",
                "std": 1.0,
                "positions": 1,
                "fourier_min": 0.01,
                "fourier_max": 10.0,
            },
            {
                "type": "fourier",
                "field": "bonus",
                "std": 1.0,
                "positions": 1,
                "fourier_min": 1.0,
                "fourier_max": 100.0,
            },
        ],
    )
    x = torch.tensor([1.5])
    idx = torch.tensor([0])
    got_reward = encoder.fourier["reward"](x, idx)
    got_bonus = encoder.fourier["bonus"](x, idx)
    scale = 1.0 / (0.5 ** 0.5)
    ref_reward = StaticFourierFeatures(32, in_min=0.01, in_max=10.0, output_scale=scale)(x, idx)
    ref_bonus = StaticFourierFeatures(32, in_min=1.0, in_max=100.0, output_scale=scale)(x, idx)
    assert torch.allclose(got_reward, ref_reward)
    assert torch.allclose(got_bonus, ref_bonus)
    assert not torch.allclose(got_reward, got_bonus)


def test_static_fourier_no_parameters() -> None:
    from mouse_core.models.embedding import StaticFourierFeatures

    ff = StaticFourierFeatures(num_features=8, in_min=0.01, in_max=10.0)
    assert sum((p.numel() for p in ff.parameters())) == 0
    y = ff(torch.tensor([0.5, -0.5]))
    assert y.shape == (2, 8)


def test_static_fourier_stays_fp32_under_bf16_cast() -> None:
    from mouse_core.models.embedding import StaticFourierFeatures

    ref = StaticFourierFeatures(num_features=64, in_min=0.01, in_max=100.0)
    cast = StaticFourierFeatures(num_features=64, in_min=0.01, in_max=100.0).to(
        dtype=torch.bfloat16
    )
    assert cast.freqs.dtype == torch.float32
    assert cast.phases.dtype == torch.float32
    x = torch.tensor([1.0, 3.7, 10.0])
    a = ref(x)
    b = cast(x)
    assert b.dtype == torch.float32
    assert torch.allclose(a, b)


def test_numeric_embedder_bf16_keeps_fourier_precision() -> None:
    torch.manual_seed(0)
    fp32 = _enc(hidden_dim=64, modalities=[{"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}])
    torch.manual_seed(0)
    bf16 = _enc(hidden_dim=64, modalities=[{"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}]).to(
        dtype=torch.bfloat16
    )
    batch = _tb(fp32, [[{"reward": 10.0, "task_index": 0}]])
    ref, _ = fp32(batch)
    out, _ = bf16(batch)
    assert out.dtype == torch.bfloat16
    err = (out.float() - ref).abs().max().item()
    assert err < 0.02, err


def test_numeric_embedder_extra_fields_in_objective_fields() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}],
    )
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "head_output": True,
            }
        ],
        objective_fields=[
            {"input_field": "action", "output_field": "action"},
            {"input_field": "old_log_prob", "output_field": "old_log_prob"},
        ],
        grouping_field="grouping_id",
    )
    batch = [
        [{"action": 1, "old_log_prob": 0.25}, {"action": 2, "old_log_prob": -1.5}]
    ]
    tb, obj = batch_to_packed(tokenizer, batch)
    assert tb.L == 2
    assert obj["old_log_prob"].tolist() == pytest.approx([0.25, -1.5])
    embeds, _ = encoder(tb)
    assert embeds.shape == (2, 8)


def test_task_done_is_objective_field_not_input_field() -> None:
    """task_done is an objective column; it is not a transformer input token."""
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": "discrete", "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1},
        ],
    )
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action", "output_field": "action"},
            {
                "type": "discrete",
                "input_field": "episode_done",
                "output_field": "episode_done",
                "head_output": True,
            },
        ],
        objective_fields=[
            {"input_field": "action", "output_field": "action"},
            {"input_field": "episode_done", "output_field": "episode_done"},
            {"input_field": "task_done", "output_field": "task_done"},
        ],
        grouping_field="grouping_id",
    )
    batch = [[{"action": 1, "episode_done": 0, "task_done": 2}]]
    tb, obj = batch_to_packed(tokenizer, batch)
    assert "task_done" not in tb.modality_names
    assert tb.L == 2
    assert obj["task_done"].tolist() == [2]
    embeds, _ = encoder(tb)
    assert embeds.shape == (2, 8)


def test_numeric_embedder_adds_type_vector_to_discrete() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[{"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}],
    )
    tb = _tb(encoder, [[{"action": 2}]])
    embeds, _ = encoder(tb)
    content = encoder._tables["action"](torch.tensor([2]))
    assert torch.allclose(embeds[0], content[0] + encoder._type_vectors["action"][0])


def test_numeric_embedder_adds_type_vector_to_fourier() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}])
    tb = _tb(encoder, [[{"reward": 1.5}]])
    embeds, _ = encoder(tb)
    feat = encoder.fourier["reward"](torch.tensor([1.5]), torch.tensor([0])) * 0.02
    assert torch.allclose(
        embeds[0].float(), feat[0] + encoder._type_vectors["reward"][0]
    )


def test_numeric_embedder_type_vectors_are_per_position_for_continuous() -> None:
    torch.manual_seed(0)
    encoder = _enc(
        hidden_dim=8,
        modalities=[{"type": "continuous", "field": "obs", "dim": 3, "std": 0.02, "positions": 3, "fourier_min": 0.01, "fourier_max": 10.0}],
    )
    assert encoder._type_vectors["obs"].shape == (3, 8)
    # Same scalar in every coordinate: content differs only by the Fourier
    # bank phase; the type vectors must be what tells coordinates apart.
    tb = _tb(encoder, [[{"obs": [0.5, 0.5, 0.5]}]])
    assert tb.positions.tolist() == [0, 1, 2]
    embeds, _ = encoder(tb)
    for i in range(3):
        feat = encoder.fourier["obs"](torch.tensor([0.5]), torch.tensor([i]))[0] * 0.02
        assert torch.allclose(embeds[i], feat + encoder._type_vectors["obs"][i])
    tv = encoder._type_vectors["obs"]
    assert not torch.allclose(tv[0], tv[1]) and not torch.allclose(tv[1], tv[2])


def test_numeric_embedder_type_vectors_are_per_position_for_learnable() -> None:
    torch.manual_seed(0)
    encoder = _enc(hidden_dim=8, modalities=[{"type": "learnable", "tokens": 3, "std": 0.02, "positions": 3}])
    name = encoder.modalities[0].field
    assert isinstance(name, str)
    assert encoder._type_vectors[name].shape == (3, 8)
    tb = _tb(encoder, [[{}]])
    assert tb.positions.tolist() == [0, 1, 2]
    embeds, _ = encoder(tb)
    content = cast(nn.Embedding, encoder._tables[name])(torch.arange(3))
    assert torch.allclose(embeds, content + encoder._type_vectors[name])


def test_numeric_embedder_rejects_more_tokens_than_declared() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[{"type": "continuous", "field": "obs", "dim": 2, "std": 0.02, "positions": 2, "fourier_min": 0.01, "fourier_max": 10.0}],
    )
    tb = _tb(encoder, [[{"obs": [0.1, 0.2]}]])
    tb.positions[:] = [0, 5]
    with pytest.raises(ValueError, match="emitted 6 tokens"):
        encoder(tb)


def test_numeric_embedder_type_vectors_are_per_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": "discrete", "field": "observation", "vocab_size": 4, "std": 0.02, "positions": 1},
        ],
    )
    assert encoder._type_vectors["action"] is not encoder._type_vectors["observation"]
    encoder._type_vectors["observation"].data.zero_()
    tb = _tb(encoder, [[{"action": 1, "observation": 1}]])
    embeds, _ = encoder(tb)
    action_content = encoder._tables["action"](torch.tensor([1]))
    obs_content = encoder._tables["observation"](torch.tensor([1]))
    assert torch.allclose(embeds[0], action_content[0] + encoder._type_vectors["action"][0])
    assert torch.allclose(embeds[1], obs_content[0])
