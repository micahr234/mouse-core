"""Explicit head-output tokens: flagging, packing, objectives, multi-token readout."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from mouse_core.data import NumericTokenizer, pack_token_batch
from mouse_core.models import LatentReasoner, Model
from mouse_core.models.backbone import LlamaBackbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import DiscreteActionValueHead
from mouse_core.models.reasoner import _plan_insertions
from mouse_core.objectives import DqnObjective
from tests._token_batch_helpers import batch_to_packed, tok_from_encoder

_HIDDEN = 32
_ACTIONS = 4

# Two learnable head-output tokens per step: every step yields two Q rows.
_MODALITIES = [
    {"type": "discrete", "field": "action", "vocab_size": _ACTIONS, "std": 0.02, "positions": 1},
    {"type": "discrete", "field": "observation", "vocab_size": 16, "std": 0.02, "positions": 1},
    {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
    {"type": "discrete", "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1},
    {"type": "learnable", "field": "value", "tokens": 2, "std": 0.02, "positions": 2},
]
_TOKENS_PER_STEP = 6


def _tiny_model(*, with_reasoner: bool = False) -> Model:
    encoder = NumericEmbedder(hidden_dim=_HIDDEN, modalities=_MODALITIES)
    backbone = LlamaBackbone(
        train_kernel="varlen", decode_kernel="flex", dtype=torch.float32,
        hidden_dim=_HIDDEN,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
    )
    heads = DiscreteActionValueHead(
        in_features=_HIDDEN,
        out_features=_ACTIONS,
        hidden_dim=_HIDDEN,
        num_layers=1,
    )
    reasoner = (
        LatentReasoner(hidden_dim=_HIDDEN, num_thoughts=2) if with_reasoner else None
    )
    return Model(encoder=encoder, backbone=backbone, heads=heads, action_head="action_value", reasoner=reasoner, recurrence=None)


def _rows(n: int, offset: int = 0) -> list[dict]:
    return [
        {
            "action": (i + offset) % _ACTIONS,
            "observation": (3 * i + offset) % 16,
            "reward": float(i),
            "episode_done": 0,
            "task_done": 0,
            "grouping_id": 0,
        }
        for i in range(n)
    ]


_BATCH = [_rows(3), _rows(2, offset=1)]


def _packed(model: Model):
    return batch_to_packed(tok_from_encoder(model.encoder), _BATCH)


# ---------------------------------------------------------------------------
# Tokenizer flag validation
# ---------------------------------------------------------------------------


def test_tokenizer_requires_exactly_one_head_output_field() -> None:
    with pytest.raises(ValueError, match="exactly one input field with"):
        NumericTokenizer(
            input_fields=[{"type": "discrete", "input_field": "action"}],
            grouping_field="task_index",
        )
    with pytest.raises(ValueError, match="exactly one input field with"):
        NumericTokenizer(
            input_fields=[
                {"type": "discrete", "input_field": "action", "head_output": True},
                {"type": "discrete", "input_field": "obs", "head_output": True},
            ],
            grouping_field="task_index",
        )


def test_step_without_head_output_token_raises() -> None:
    tok = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action"},
            {
                "type": "fourier",
                "input_field": "reward",
                "skip": 0.0,
                "head_output": True,
            },
        ],
        grouping_field="task_index",
    )
    # Head-output field present → fine.
    st = tok({"action": 1, "reward": 0.5, "task_index": 0})
    assert st.head_output_mask.tolist() == [False, True]
    # Head-output field skipped → the step has no head-output token.
    with pytest.raises(ValueError, match="no head-output tokens"):
        tok({"action": 1, "reward": 0.0, "task_index": 0})


# ---------------------------------------------------------------------------
# Packing: multiple head-output tokens per step
# ---------------------------------------------------------------------------


def test_pack_multi_head_output_layout() -> None:
    model = _tiny_model()
    batch, objective_data = _packed(model)
    N = sum(len(seq) for seq in _BATCH)
    assert batch.N == N
    assert batch.P == 2 * N
    # Each step's head-output tokens are its two trailing learnable tokens.
    expected = []
    for i in range(N):
        base = i * _TOKENS_PER_STEP
        expected += [base + 4, base + 5]
    assert batch.head_output_indices.tolist() == expected
    assert batch.head_output_steps.tolist() == [i for i in range(N) for _ in range(2)]
    assert batch.step_counts().tolist() == [3, 2]
    pred_mod = batch.modality_names.index("value")
    assert batch.modality_ids[batch.head_output_indices].tolist() == [pred_mod] * (2 * N)
    # pack stamps the row→step map for the objectives.
    assert objective_data["head_output_count"].tolist() == [2] * N


# ---------------------------------------------------------------------------
# Model forward
# ---------------------------------------------------------------------------


def test_forward_yields_one_row_per_head_output_token() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch, _ = _packed(model)
    with torch.no_grad():
        out = model(batch)
    assert out.predictions["action_value"].shape == (batch.P, _ACTIONS)
    assert out.head_output_indices is not None
    assert out.head_output_indices.shape == (batch.P,)
    action = model.get_action(out.predictions, temperature=0.0)
    assert action.shape == (1,)


def test_decode_pools_last_head_output_token_per_step() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch, _ = _packed(model)
    with torch.no_grad():
        flat = model(batch).predictions
        rect = model(batch, use_cache=True).predictions
    q_flat = flat["action_value"]  # [P, A]
    q_rect = rect["action_value"]  # [B, S, A], steps left-padded
    psteps = batch.head_output_steps
    last = np.ones(batch.P, dtype=bool)
    last[:-1] = psteps[1:] != psteps[:-1]
    last_rows = np.flatnonzero(last)
    counts = batch.step_counts().tolist()
    S = q_rect.shape[1]
    row = 0
    for b, n in enumerate(counts):
        for s in range(n):
            flat_q = q_flat[last_rows[row]]
            rect_q = q_rect[b, S - n + s]
            assert torch.allclose(flat_q, rect_q, atol=1e-4), (b, s)
            row += 1


# ---------------------------------------------------------------------------
# DQN objective with several head-output rows per step
# ---------------------------------------------------------------------------


def _objective_data(
    N: int, *, counts: list[int] | None = None, actions: list[int] | None = None
) -> TensorDict:
    data = {
        "action": torch.tensor(actions or list(range(N)), dtype=torch.int64) % _ACTIONS,
        "reward": torch.arange(N, dtype=torch.float32) / 2,
        "episode_done": torch.zeros(N, dtype=torch.int64),
        "task_done": torch.zeros(N, dtype=torch.int64),
        "sequence_id": torch.zeros(N, dtype=torch.int64),
    }
    if counts is not None:
        data["head_output_count"] = torch.tensor(counts, dtype=torch.int64)
    return TensorDict(data, batch_size=[N])


def test_dqn_duplicated_rows_match_single_head_output() -> None:
    torch.manual_seed(0)
    N, A = 5, _ACTIONS
    q = torch.randn(N, A)
    q_target = torch.randn(N, A)
    objective = DqnObjective(gamma_step=0.9)

    base_loss, base_metrics = objective(
        _objective_data(N),
        TensorDict({"action_value": q}, batch_size=[N]),
        TensorDict({"action_value": q_target}, batch_size=[N]),
    )
    # Duplicate every step's head-output row: same targets, same loss.
    q2 = q.repeat_interleave(2, dim=0)
    q2_target = q_target.repeat_interleave(2, dim=0)
    dup_loss, dup_metrics = objective(
        _objective_data(N, counts=[2] * N),
        TensorDict({"action_value": q2}, batch_size=[2 * N]),
        TensorDict({"action_value": q2_target}, batch_size=[2 * N]),
    )
    assert torch.allclose(base_loss, dup_loss, atol=1e-6)
    for key in ("q_values_mean", "q_values_min", "q_values_max"):
        assert base_metrics[key] == pytest.approx(dup_metrics[key], abs=1e-6)


def test_dqn_multi_head_output_shares_step_target() -> None:
    # N=2 steps, step 0 has two head-output rows, step 1 has one. Both rows of
    # step 0 train toward the same target, bootstrapped from step 1's last row.
    gamma = 0.9
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    q_target = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    objective = DqnObjective(gamma_step=gamma)
    data = _objective_data(2, counts=[2, 1], actions=[0, 1])
    data["reward"] = torch.tensor([0.0, 0.5])
    loss, _ = objective(
        data,
        TensorDict({"action_value": q}, batch_size=[3]),
        TensorDict({"action_value": q_target}, batch_size=[3]),
    )
    target = 0.5 + gamma * 60.0  # r_1 + gamma * max_a Q_target(s_1) (row 2)
    expected = ((2.0 - target) ** 2 + (4.0 - target) ** 2) / 2  # a_1 = 1
    assert loss.item() == pytest.approx(expected)


def test_dqn_misaligned_head_output_count_raises() -> None:
    N = 3
    q = torch.randn(2 * N, _ACTIONS)
    preds = TensorDict({"action_value": q}, batch_size=[2 * N])
    objective = DqnObjective()
    with pytest.raises(ValueError, match="misaligned"):
        objective(_objective_data(N, counts=[2, 2, 1]), preds, preds.clone())
    with pytest.raises(ValueError, match="head_output_count column"):
        objective(_objective_data(N), preds, preds.clone())


# ---------------------------------------------------------------------------
# Reasoning with several head-output tokens per step
# ---------------------------------------------------------------------------


def test_plan_anchors_on_first_head_output_token() -> None:
    model = _tiny_model(with_reasoner=True)
    batch, _ = _packed(model)
    # Seq 0: steps at tokens 0..17 with head-output pairs (4,5), (10,11),
    # (16,17); burst at step 1 → anchor 10 (the *first* head-output token).
    plan = _plan_insertions(batch, np.array([1, -1]), num_thoughts=2)
    assert plan is not None
    assert plan.anchors.tolist() == [10]
    assert plan.latent_positions.tolist() == [10, 11]
    shifted = [p if p < 10 else p + 2 for p in batch.head_output_indices.tolist()]
    assert plan.ext_head_output_indices.tolist() == shifted


def test_reasoning_forward_and_delayed_parity_multi_head_output() -> None:
    torch.manual_seed(0)
    model = _tiny_model(with_reasoner=True).eval()
    batch, _ = _packed(model)
    delayed = model.delayed_copy().eval()
    with torch.no_grad():
        out = model(batch, reasoning=[1, 0])
        delayed_out = delayed(batch, reasoning=[1, 0])
    assert out.predictions["action_value"].shape == (batch.P, _ACTIONS)
    assert torch.allclose(
        out.predictions["action_value"], delayed_out.predictions["action_value"], atol=1e-4
    )
