"""Explicit head-output tokens: flagging, packing, objectives, multi-token readout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_core.data import Tokenizer, pack_token_batch
from mouse_core.models import LatentReasoner, Model, ModelOutput
from mouse_core.models.backbone import TransformerBackbone
from mouse_core.models.heads import RegressionHead
from mouse_core.models.reasoner import _plan_insertions
from mouse_core.objectives import DqnObjective, affine_reward, affine_value, boundary_discount
from tests._token_batch_helpers import batch_to_packed


def _disc(**overrides: float):
    kwargs = dict(
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
    )
    kwargs.update(overrides)
    return boundary_discount(**kwargs)


def _rew(**overrides: object):
    kwargs: dict[str, object] = dict(scale=1.0, shift=0.0)
    kwargs.update(overrides)
    return affine_reward(**kwargs)  # type: ignore[arg-type]


def _val(**overrides: object):
    kwargs: dict[str, object] = dict(scale=1.0, shift=0.0)
    kwargs.update(overrides)
    return affine_value(**kwargs)  # type: ignore[arg-type]


_HIDDEN = 32
_ACTIONS = 4
# Three token fields + a two-id text const as the head-output field.
_TOKENS_PER_STEP = 5
_HEAD_OUTPUT_IDS = 2


class _FakeTokenizer:

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        ids = [ord(c) % 20 + 1 for c in text] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _head_output_tokenizer(
    *token_fields: str,
    value: str = "ab",
    tail: str | None = None,
    grouping_field: str = "grouping_id",
) -> Tokenizer:
    fields: list[dict] = [{"type": "token", "input_field": name} for name in token_fields]
    fields.append(
        {
            "type": "token",
            "input_field": "episode_index",
            "when_field": "step_index",
            "when_equals": 0,
        }
    )
    fields.append(
        {"type": "text", "output_field": "value", "format": value, "head_output": True}
    )
    if tail is not None:
        fields.append({"type": "token", "input_field": tail})
    obj = [{"input_field": name} for name in token_fields]
    if tail is not None:
        obj.append({"input_field": tail})
    return Tokenizer(
        input_fields=fields,
        tokenizer=_FakeTokenizer(),
        grouping_field=grouping_field,
        objective_fields=obj,
    )


_TOK = _head_output_tokenizer("action", "observation", "episode_done")


def _tiny_model(*, with_reasoner: bool = False) -> Model:
    backbone = TransformerBackbone(architecture="llama",
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True,
        hidden_dim=_HIDDEN,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
        vocab_size=32)
    heads = RegressionHead(
        in_features=_HIDDEN,
        out_features=_ACTIONS,
        hidden_dim=_HIDDEN,
        num_layers=1, use_norm=True,
        propagate_gradient=1.0,
    )
    reasoner = (
        LatentReasoner(hidden_dim=_HIDDEN, num_thoughts=2) if with_reasoner else None
    )
    return Model(backbone=backbone, heads=heads, action_source="action_value", reasoner=reasoner)


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
    return batch_to_packed(_TOK, _BATCH)


# ---------------------------------------------------------------------------
# Tokenizer flag validation
# ---------------------------------------------------------------------------


def test_tokenizer_requires_exactly_one_head_output_field() -> None:
    with pytest.raises(ValueError, match="exactly one input field with"):
        Tokenizer(
            input_fields=[{"type": "token", "input_field": "action"},
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when_field": "step_index",
                    "when_equals": 0,
                },
            ],
            grouping_field="task_index",
        )
    with pytest.raises(ValueError, match="exactly one input field with"):
        Tokenizer(
            input_fields=[
                {"type": "token", "input_field": "action", "head_output": True},
                {"type": "token", "input_field": "obs", "head_output": True},
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when_field": "step_index",
                    "when_equals": 0,
                },
            ],
            grouping_field="task_index",
        )


def test_step_without_head_output_token_raises() -> None:
    tok = Tokenizer(
        input_fields=[
            {"type": "token", "input_field": "action"},
            {
                "type": "text",
                "input_field": "reward",
                "format": "{field}",
                "skip": 0.0,
                "format_skipped": "",
                "head_output": True,
            },
            {
                "type": "token",
                "input_field": "episode_index",
                "when_field": "step_index",
                "when_equals": 0,
            },
        ],
        tokenizer=_FakeTokenizer(),
        grouping_field="task_index",
    )
    # Head-output field present → fine.
    st = tok({"action": 1, "reward": 0.5, "task_index": 0})
    assert st.head_output_mask.tolist() == [False] + [True] * len("0.5")
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
    assert batch.P == _HEAD_OUTPUT_IDS * N
    expected = []
    for i in range(N):
        base = i * _TOKENS_PER_STEP
        expected += [base + 3, base + 4]
    assert batch.head_output_indices.tolist() == expected
    assert batch.head_output_steps.tolist() == [i for i in range(N) for _ in range(_HEAD_OUTPUT_IDS)]
    assert batch.step_counts().tolist() == [3, 2]
    pred_mod = batch.modality_names.index("__text__")
    assert batch.modality_ids[batch.head_output_indices].tolist() == [pred_mod] * (_HEAD_OUTPUT_IDS * N)
    assert objective_data["head_output_count"].tolist() == [_HEAD_OUTPUT_IDS] * N


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


def test_get_action_uses_last_valid_head_output_not_last_token() -> None:
    """get_action reads Q above the last valid head-output token.

    Each step ends with a trailing non-head-output token, so the last
    sequence token is *not* a readout. Decode also left-pads a short row.
    ``get_action(ModelOutput)`` must match the value head at the last
    head-output index, not the last token and not a padded step column.
    """
    torch.manual_seed(0)
    hidden = _HIDDEN
    backbone = TransformerBackbone(architecture="llama",
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True,
        hidden_dim=hidden, num_layers=2, num_heads=2, max_position_embeddings=128, vocab_size=32)
    model = Model(
        backbone=backbone,
        heads=(head := RegressionHead(
            in_features=hidden, out_features=_ACTIONS, hidden_dim=hidden, num_layers=1, use_norm=True,
            propagate_gradient=1.0,
        )),
        action_source="action_value",
        reasoner=None,
    ).eval()
    tok = _head_output_tokenizer("action", "observation", tail="tail")
    rows = [
        [{**step, "tail": (i + 3) % 8} for i, step in enumerate(_rows(3))],
        [{**step, "tail": (i + 5) % 8} for i, step in enumerate(_rows(1, offset=2))],
    ]
    batch, _ = batch_to_packed(tok, rows)
    with torch.no_grad():
        out = model(batch, use_cache=True)
        action = model.get_action(out=out, temperature=0.0)

    assert out.head_output_valid is not None
    assert out.head_output_valid.tolist() == [[True, True, True], [False, False, True]]
    last_head = out.head_output_indices[torch.arange(2), 2]
    assert (last_head != out.last_hidden_state.shape[1] - 1).all()

    with torch.no_grad():
        h_head = out.last_hidden_state[torch.arange(2), last_head]
        q_head = model.head(h=h_head)["action_value"]
        h_tok = out.last_hidden_state[:, -1]
        q_tok = model.head(h=h_tok)["action_value"]
    assert not torch.allclose(q_head, q_tok, atol=1e-5)
    assert action.tolist() == q_head.argmax(dim=-1).tolist()


def test_get_action_model_output_uses_last_valid_step_column() -> None:
    """A scores tensor's last column is ignored when a later slot is invalid."""
    model = _tiny_model()
    preds = {
            "action_value": torch.tensor(
                [
                    [[0.0, 9.0], [5.0, 0.0], [0.0, 3.0]],
                    [[1.0, 0.0], [0.0, 1.0], [8.0, 0.0]],
                ]
            )
        }
    # Last valid columns are 1 and 0 — not the trailing pad column.
    valid = torch.tensor([[True, True, False], [True, False, False]])
    out = ModelOutput(
        predictions=preds,
        last_hidden_state=torch.zeros(2, 1, _HIDDEN),
        head_output_indices=torch.zeros(2, 3, dtype=torch.long),
        head_output_valid=valid,
    )
    action = model.get_action(out=out, temperature=0.0)
    assert action.tolist() == [0, 0]
    # scores-dict path still takes the last step axis.
    assert model.get_action(out=preds, temperature=0.0).tolist() == [1, 0]


def test_get_action_rejects_flat_multi_step_train_outputs() -> None:
    model = _tiny_model()
    preds = {"action_value": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])}
    with pytest.raises(ValueError, match="N=1"):
        model.get_action(out=preds, temperature=0.0)


def test_get_action_rejects_decode_row_with_no_valid_head_output() -> None:
    model = _tiny_model()
    preds = {"action_value": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])}
    out = ModelOutput(
        predictions=preds,
        last_hidden_state=torch.zeros(1, 1, _HIDDEN),
        head_output_indices=torch.zeros(1, 2, dtype=torch.long),
        head_output_valid=torch.tensor([[False, False]]),
    )
    with pytest.raises(ValueError, match="valid head-output"):
        model.get_action(out=out, temperature=0.0)


# ---------------------------------------------------------------------------
# DQN objective with several head-output rows per step
# ---------------------------------------------------------------------------


def _objective_data(
    N: int, *, counts: list[int] | None = None, actions: list[int] | None = None
) -> dict[str, torch.Tensor]:
    data = {
        "action": torch.tensor(actions or list(range(N)), dtype=torch.int64) % _ACTIONS,
        "reward": torch.arange(N, dtype=torch.float32) / 2,
        "episode_done": torch.zeros(N, dtype=torch.int64),
        "task_done": torch.zeros(N, dtype=torch.int64),
        "sequence_id": torch.zeros(N, dtype=torch.int64),
    }
    if counts is not None:
        data["head_output_count"] = torch.tensor(counts, dtype=torch.int64)
    return data


def test_dqn_duplicated_rows_match_single_head_output() -> None:
    torch.manual_seed(0)
    N, A = 5, _ACTIONS
    q = torch.randn(N, A)
    q_target = torch.randn(N, A)
    objective = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.9), grouping_field=None, temperature=0.0, double=False, gate=None, bootstrap_cutoff=True)

    base_loss, base_metrics = objective(
        objective_data=_objective_data(N), predictions=q,  delayed_predictions=q_target,
    )
    # Duplicate every step's head-output row: same targets, same loss.
    q2 = q.repeat_interleave(2, dim=0)
    q2_target = q_target.repeat_interleave(2, dim=0)
    dup_loss, dup_metrics = objective(
        objective_data=_objective_data(N, counts=[2] * N), predictions=q2,  delayed_predictions=q2_target,
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
    objective = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=gamma), grouping_field=None, temperature=0.0, double=False, gate=None, bootstrap_cutoff=True)
    data = _objective_data(2, counts=[2, 1], actions=[0, 1])
    data["reward"] = torch.tensor([0.0, 0.5])
    loss, _ = objective(
        objective_data=data, predictions=q,  delayed_predictions=q_target,
    )
    target = 0.5 + gamma * 60.0  # r_1 + gamma * max_a Q_target(s_1) (row 2)
    expected = ((2.0 - target) ** 2 + (4.0 - target) ** 2) / 2  # a_1 = 1
    assert loss.item() == pytest.approx(expected)


def test_dqn_misaligned_head_output_count_raises() -> None:
    N = 3
    q = torch.randn(2 * N, _ACTIONS)
    objective = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None, bootstrap_cutoff=True)
    with pytest.raises(ValueError, match="misaligned"):
        objective(objective_data=_objective_data(N, counts=[2, 2, 1]), predictions=q,  delayed_predictions=q.clone())
    with pytest.raises(ValueError, match="head_output_count column"):
        objective(objective_data=_objective_data(N), predictions=q,  delayed_predictions=q.clone())


# ---------------------------------------------------------------------------
# Reasoning with several head-output tokens per step
# ---------------------------------------------------------------------------


def test_plan_anchors_on_first_head_output_token() -> None:
    model = _tiny_model(with_reasoner=True)
    batch, _ = _packed(model)
    # Seq 0: 3 steps × 5 tokens; head-output pairs (3,4), (8,9), (13,14).
    # Burst at step 1 → anchor 8 (the *first* head-output token).
    plan = _plan_insertions(batch, np.array([1, -1]), num_thoughts=2)
    assert plan is not None
    assert plan.anchors.tolist() == [8]
    assert plan.latent_positions.tolist() == [8, 9]
    shifted = [p if p < 8 else p + 2 for p in batch.head_output_indices.tolist()]
    assert plan.ext_head_output_indices.tolist() == shifted


def test_reasoning_forward_and_delayed_parity_multi_head_output() -> None:
    torch.manual_seed(0)
    model = _tiny_model(with_reasoner=True).eval()
    batch, _ = _packed(model)
    delayed = model.copy(heads=True, backbone=True, reasoner=False).eval()
    with torch.no_grad():
        out = model(batch, reasoning=[1, 0])
        delayed_out = delayed(batch, reasoning=[1, 0])
    assert out.predictions["action_value"].shape == (batch.P, _ACTIONS)
    assert torch.allclose(
        out.predictions["action_value"], delayed_out.predictions["action_value"], atol=1e-4
    )
