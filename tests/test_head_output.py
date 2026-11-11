"""Explicit head-output tokens: flagging, packing, objectives, multi-token readout."""

from __future__ import annotations

from tests._prediction_layout import fixture_head_steps
from tests._event_fixtures import encode_measurement
from tests._model_helpers import ExampleModel as Model
from tests._model_helpers import copy_model, simple_model

import numpy as np
import pytest
import torch

from mouse_core.data import Tokenizer
from mouse_core.models import LatentReasoner
from mouse_core.models.backbone import Backbone
from mouse_core.models.mlp import MLP
from mouse_core.models.reasoner import _plan_insertions
from mouse_core.objectives import DqnObjective, affine_reward, affine_value, boundary_discount


from tests._token_batch_helpers import batch_to_packed

def _sequence_ids(data: dict[str, torch.Tensor]) -> torch.Tensor:
    if "sequence_ids" in data:
        return data["sequence_ids"]
    n = int(next(iter(data.values())).shape[0])
    return torch.zeros(n, dtype=torch.int64)


def _disc(**overrides: float):
    kwargs = dict(
        gamma_step=1.0,
        gamma_terminated=0.0,
        gamma_truncated=0.0,
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
) -> Tokenizer:
    fields: list[dict] = [{"provider": "sensor", "type": "text", "input_field": name, "format": "{" + name + "}"} for name in token_fields]
    fields.append(
        {"provider": "sensor",
            "type": "text",
            "input_field": "repeat_episode_index",
            "format": lambda ctx: f"{ctx['repeat_episode_index']}" if ('step_index' in ctx and ctx['step_index'] == 0) else "",
        }
    )
    fields.append(
        {"provider": "sensor", "type": "text", "format": value, "head_output": True}
    )
    if tail is not None:
        fields.append({"provider": "sensor", "type": "text", "input_field": tail, "format": "{" + tail + "}"})
    obj = [{"input_field": name} for name in token_fields]
    if tail is not None:
        obj.append({"input_field": tail})
    return Tokenizer(
        input_fields=fields,
        tokenizers={"default": _FakeTokenizer()},
        objective_fields=obj,
    )


_TOK = _head_output_tokenizer("action", "observation", "repeat_episode_done")


def _tiny_model(*, with_reasoner: bool=False) -> Model:
    backbone = Backbone(architecture='llama', dtype=torch.float32, hidden_dim=_HIDDEN, num_layers=2, num_heads=2, max_position_embeddings=128)
    heads = MLP(in_features=_HIDDEN, out_features=_ACTIONS, hidden_dim=_HIDDEN, num_layers=1, propagate_gradient=1.0)
    reasoner = LatentReasoner(hidden_dim=_HIDDEN, num_thoughts=2) if with_reasoner else None
    return simple_model(backbone=backbone, heads={'action_value': heads}, reasoner=reasoner)


def _rows(n: int, offset: int = 0) -> list[dict]:
    return [
        {
            "action": (i + offset) % _ACTIONS,
            "observation": (3 * i + offset) % 16,
            "reward": float(i),
            "repeat_episode_done": 0,
            "episode_done": 0,
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


@pytest.mark.parametrize('head', [False, True])
def test_tokenizer_supports_zero_or_multiple_head_fields(head) -> None:
    tok = Tokenizer(tokenizers={"default": _FakeTokenizer()}, input_fields=[
        dict(provider="sensor", type='text', input_field=name, format="{" + name + "}", head_output=head)
        for name in ('action', 'obs')
    ])
    step = encode_measurement(tok, {'action': 1, 'obs': 2})
    assert step.head_mask.tolist() == [head, head]


def test_step_without_head_output_token_is_retained() -> None:
    tok = Tokenizer(
        input_fields=[
            {"provider": "sensor", "type": "text", "input_field": "action", "format": '{action}'},
            {"provider": "sensor",
                "type": "text",
                "input_field": "reward",
                "format": '{reward}',
                "head_output": True,
            },
            {"provider": "sensor",
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": lambda ctx: f"{ctx['repeat_episode_index']}" if ('step_index' in ctx and ctx['step_index'] == 0) else "",
            },
        ],
        tokenizers={"default": _FakeTokenizer()},
    )
    # Head-output field present → fine.
    st = encode_measurement(tok, {"action": 1, "reward": 0.5, "episode_index": 0})
    assert st.head_mask.tolist() == [False] + [True] * len("0.5")
    # Head-output field missing → the step has no head-output token.
    passive = encode_measurement(tok, {"action": 1, "reward": None, "episode_index": 0})
    assert passive.token_ids.size == 1 and not passive.head_mask.any()


# ---------------------------------------------------------------------------
# Packing: multiple head-output tokens per step
# ---------------------------------------------------------------------------


def test_pack_multi_head_output_layout() -> None:
    model = _tiny_model()
    batch, objective_data = _packed(model)
    N = sum(len(seq) for seq in _BATCH)
    assert objective_data["sequence_ids"].shape[0] == N
    assert batch["head_mask"].nonzero(as_tuple=True)[0].shape[0] == _HEAD_OUTPUT_IDS * N
    expected = []
    for i in range(N):
        base = i * _TOKENS_PER_STEP
        expected += [base + 3, base + 4]
    assert batch["head_mask"].nonzero(as_tuple=True)[0].tolist() == expected
    assert objective_data["step_ids"].tolist() == [i for i in range(N) for _ in range(_HEAD_OUTPUT_IDS)]
    assert torch.bincount(objective_data["sequence_ids"], minlength=batch["num_sequences"]).tolist() == [3, 2]
    assert batch['numeric_mask'][batch["head_mask"].nonzero(as_tuple=True)[0]].tolist() == [False] * (_HEAD_OUTPUT_IDS * N)
    assert "head_output_count" not in objective_data
    assert np.bincount(objective_data["step_ids"], minlength=N).tolist() == [_HEAD_OUTPUT_IDS] * N


# ---------------------------------------------------------------------------
# Model forward
# ---------------------------------------------------------------------------


def test_forward_yields_one_row_per_head_output_token() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch, objective_data = _packed(model)
    with torch.no_grad():
        out = model(batch)
    assert out["predictions"]["action_value"].shape == (batch["head_mask"].nonzero(as_tuple=True)[0].shape[0], _ACTIONS)
    assert out["head_token_ids"] is not None
    assert out["head_token_ids"].shape == (batch["head_mask"].nonzero(as_tuple=True)[0].shape[0],)


def test_decode_preserves_every_marked_head_token() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch, objective_data = _packed(model)
    with torch.no_grad():
        flat = model(batch)["predictions"]
        rect = model(batch, use_cache=True)["predictions"]
    q_flat = flat["action_value"]  # [P, A]
    q_rect = rect["action_value"]  # [B, S, A], steps left-padded
    counts = torch.bincount(batch["sequence_ids"][batch["head_mask"]], minlength=batch["num_sequences"]).tolist()
    S = q_rect.shape[1]
    row = 0
    for b, n in enumerate(counts):
        for s in range(n):
            flat_q = q_flat[row]
            rect_q = q_rect[b, S - n + s]
            assert torch.allclose(flat_q, rect_q, atol=1e-4), (b, s)
            row += 1


def test_get_action_uses_last_valid_head_output_not_last_token() -> None:
    """get_action reads Q above the last valid head-output token.

    Each step ends with a trailing non-head-output token, so the last
    sequence token is *not* a readout. Decode also left-pads a short row.
    ``get_action(output_dict)`` must match the value head at the last
    head-output index, not the last token and not a padded step column.
    """
    torch.manual_seed(0)
    hidden = _HIDDEN
    backbone = Backbone(architecture='llama', dtype=torch.float32, hidden_dim=hidden, num_layers=2, num_heads=2, max_position_embeddings=128)
    model = simple_model(backbone=backbone, heads=({'action_value': (head := MLP(in_features=hidden, out_features=_ACTIONS, hidden_dim=hidden, num_layers=1, propagate_gradient=1.0))}), reasoner=None).eval()
    tok = _head_output_tokenizer('action', 'observation', tail='tail')
    rows = [[{**step, 'tail': (i + 3) % 8} for i, step in enumerate(_rows(3))], [{**step, 'tail': (i + 5) % 8} for i, step in enumerate(_rows(1, offset=2))]]
    batch, _ = batch_to_packed(tok, rows)
    with torch.no_grad():
        out = model(batch, use_cache=True)
        action = model.get_action(out=out, temperature=0.0)
    assert out["head_valid_mask"] is not None
    assert out["head_valid_mask"].tolist() == [[True] * 6, [False] * 4 + [True] * 2]
    last_head = out["head_token_ids"][torch.arange(2), -1]
    assert (last_head != out["last_hidden_state"].shape[1] - 1).all()
    with torch.no_grad():
        h_head = out["last_hidden_state"][torch.arange(2), last_head]
        q_head = model.head(h=h_head)['action_value']
        h_tok = out["last_hidden_state"][:, -1]
        q_tok = model.head(h=h_tok)['action_value']
    assert not torch.allclose(q_head, q_tok, atol=1e-05)
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
    out = {
        "predictions": preds,
        "last_hidden_state": torch.zeros(2, 1, _HIDDEN),
        "head_token_ids": torch.zeros(2, 3, dtype=torch.long),
        "head_valid_mask": valid,
    }
    action = model.get_action(out=out, temperature=0.0)
    assert action.tolist() == [0, 0]
    # scores-dict path still takes the last step axis.
    assert model.get_action(out={"predictions": preds}, temperature=0.0).tolist() == [1, 0]


def test_get_action_rejects_flat_multi_step_train_outputs() -> None:
    model = _tiny_model()
    preds = {"action_value": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])}
    with pytest.raises(ValueError, match="N=1"):
        model.get_action(out={"predictions": preds}, temperature=0.0)


def test_get_action_rejects_decode_row_with_no_valid_head_output() -> None:
    model = _tiny_model()
    preds = {"action_value": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])}
    out = {
        "predictions": preds,
        "last_hidden_state": torch.zeros(1, 1, _HIDDEN),
        "head_token_ids": torch.zeros(1, 2, dtype=torch.long),
        "head_valid_mask": torch.tensor([[False, False]]),
    }
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
        "terminated": torch.zeros(N, dtype=torch.int64),
        "truncated": torch.zeros(N, dtype=torch.int64),
        "sequence_ids": torch.zeros(N, dtype=torch.int64),
    }
    data["terminated"][-1] = 1
    if counts is not None:
        data["readout_counts"] = torch.tensor(counts, dtype=torch.int64)
    return data


def test_dqn_duplicated_rows_match_single_head_output() -> None:
    torch.manual_seed(0)
    N, A = 5, _ACTIONS
    q = torch.randn(N, A)
    q_target = torch.randn(N, A)
    objective = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.9), temperature=0.0, gate=None)

    base_data = _objective_data(N)
    base_loss, base_metrics = objective(
        objective_data=base_data, sequence_ids=_sequence_ids(base_data), step_ids=fixture_head_steps(base_data),  predictions=q,  backup_value_predictions=q_target, backup_action_predictions=q_target
    )
    # Duplicate every step's head-output row: same targets, same loss.
    q2 = q.repeat_interleave(2, dim=0)
    q2_target = q_target.repeat_interleave(2, dim=0)
    dup_data = _objective_data(N, counts=[2] * N)
    dup_loss, dup_metrics = objective(
        objective_data=dup_data, sequence_ids=_sequence_ids(dup_data), step_ids=fixture_head_steps(dup_data),  predictions=q2,  backup_value_predictions=q2_target, backup_action_predictions=q2_target
    )
    assert torch.allclose(base_loss["action_value"], dup_loss["action_value"], atol=1e-6)
    for key in ("q_values_mean", "q_values_min", "q_values_max"):
        torch.testing.assert_close(base_metrics[key], dup_metrics[key], rtol=1e-6, atol=1e-6)


def test_dqn_multi_head_output_shares_step_target() -> None:
    # Three steps: two readouts, one readout, then a terminal without a readout. Both rows of
    # step 0 train toward the same target, bootstrapped from step 1's last row.
    gamma = 0.9
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    q_target = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    objective = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=gamma), temperature=0.0, gate=None)
    data = _objective_data(3, counts=[2, 1, 0], actions=[0, 1, 0])
    data["reward"] = torch.tensor([0.0, 0.5, 0.0])
    loss, _ = objective(
        objective_data=data, sequence_ids=_sequence_ids(data), step_ids=fixture_head_steps(data),  predictions=q,  backup_value_predictions=q_target, backup_action_predictions=q_target
    )
    target = 0.5 + gamma * 60.0  # r_1 + gamma * max_a Q_target(s_1) (row 2)
    expected = ((2.0 - target) ** 2 + (4.0 - target) ** 2 + 5.0 ** 2) / 3  # a_1 = 1
    assert loss["action_value"].item() == pytest.approx(expected)


def test_dqn_misaligned_readout_counts_raises() -> None:
    N = 3
    q = torch.randn(2 * N, _ACTIONS)
    objective = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), temperature=0.0, gate=None)
    bad = _objective_data(N, counts=[2, 2, 1])
    with pytest.raises(ValueError, match="step_ids must have shape"):
        objective(objective_data=bad, sequence_ids=_sequence_ids(bad), step_ids=fixture_head_steps(bad),  predictions=q,  backup_value_predictions=q.clone(), backup_action_predictions=q.clone())
    plain = _objective_data(N)
    with pytest.raises(ValueError, match="step_ids must have shape"):
        objective(objective_data=plain, sequence_ids=_sequence_ids(plain), step_ids=fixture_head_steps(plain),  predictions=q,  backup_value_predictions=q.clone(), backup_action_predictions=q.clone())


# ---------------------------------------------------------------------------
# Reasoning with several head-output tokens per step
# ---------------------------------------------------------------------------


def test_plan_anchors_on_first_head_output_token() -> None:
    model = _tiny_model(with_reasoner=True)
    batch, objective_data = _packed(model)
    # Seq 0: 3 steps × 5 tokens; head-output pairs (3,4), (8,9), (13,14).
    # Burst at step 1 → anchor 8 (the *first* head-output token).
    plan = _plan_insertions(batch, objective_data, np.array([1, -1]), num_thoughts=2)
    assert plan is not None
    assert plan.anchors.tolist() == [8]
    assert plan.latent_positions.tolist() == [8, 9]
    shifted = [p if p < 8 else p + 2 for p in batch["head_mask"].nonzero(as_tuple=True)[0].tolist()]
    assert plan.ext_head_token_ids.tolist() == shifted


def test_reasoning_forward_and_delayed_parity_multi_head_output() -> None:
    torch.manual_seed(0)
    model = _tiny_model(with_reasoner=True).eval()
    batch, objective_data = _packed(model)
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False).eval()
    with torch.no_grad():
        out = model(batch, objective_data=objective_data, reasoning=[1, 0])
        delayed_out = delayed(batch, objective_data=objective_data, reasoning=[1, 0])
    assert out["predictions"]['action_value'].shape == (batch["head_mask"].nonzero(as_tuple=True)[0].shape[0], _ACTIONS)
    assert torch.allclose(out["predictions"]['action_value'], delayed_out["predictions"]['action_value'], atol=0.0001)
