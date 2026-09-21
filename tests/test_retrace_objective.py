"""Tests for the Retrace(λ) objective on synthetic tensors."""
from __future__ import annotations

import math

import pytest
import torch

from mouse_core.objectives import DqnObjective, RetraceObjective, affine_reward, affine_value, boundary_discount
from mouse_core.objectives.dqn import _policy_entropy
from mouse_core.objectives.retrace import _softmax_policy


# softmax([3, 0] / _T) = [0.75, 0.25]; softmax([0, 100] / _T) rounds to [0, 1] in fp32.
_T = 3.0 / math.log(3.0)


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




def _preds(
    online: torch.Tensor, delayed: torch.Tensor, behavior: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    n = online.shape[0]
    return (
        {"action_value": online, "behavior": behavior},
        {"action_value": delayed},
    )


class _RetraceCall:
    def __init__(self, objective: RetraceObjective) -> None:
        self.objective = objective

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
        delayed_predictions: dict[str, torch.Tensor] | None = None,
    ):
        kwargs: dict[str, object] = dict(
            objective_data=objective_data,
            predictions=predictions["action_value"],
            behavior_predictions=predictions["behavior"],
        )
        if delayed_predictions is not None:
            kwargs["delayed_predictions"] = delayed_predictions["action_value"]
        return self.objective(**kwargs)  # type: ignore[arg-type]


def _retrace(**overrides: object) -> _RetraceCall:
    kwargs: dict[str, object] = dict(
        td_lambda=1.0, temperature=_T, behavior_weight=1.0, grouping_field=None, discount=_disc(), reward=_rew(), value=_val()
    )
    kwargs.update(overrides)
    return _RetraceCall(RetraceObjective(**kwargs))  # type: ignore[arg-type]


def _fixture(mu_1: float) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Three in-run steps; ``mu_1`` is the behavior head's ``μ(a_1 = 1 | s_1)``.

    Action from s0 is 0; from s1 is 1. Delayed Q is ``[0, 0]`` at s0,
    ``[3, 0]`` at s1 and ``[0, 100]`` at s2. Rewards out of s0 / s1 are 1
    and 10. Online ``Q(s0, 0) = 5`` and ``Q(s1, 1) = 0``. Behavior logits
    are uniform at s0 (``μ(a_0 | s_0) = 0.5``) and ``log [1 - mu_1, mu_1]``
    at s1.

    With ``temperature=_T`` the softmax target policy is
    ``π(s1) = [0.75, 0.25]`` and ``π(s2) ≈ [0, 1]``,     so ``V_π(s1) = E_π Q + T H[π]`` and
    ``V_π(s2) ≈ 100``. The last pair has no continuation:
    ``G_1 = 10 + V_π(s2)``. At s0 the trace through ``a_1`` is
    ``c = λ min(1, 0.25 / mu_1)`` and
    ``G_0 = 1 + V_π(s1) + c * (G_1 - Q(s1, 1) = 0)``.
    """
    step_stream = {
            "action": torch.tensor([0, 0, 1]),
            "reward": torch.tensor([0.0, 1.0, 10.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.zeros(3, dtype=torch.int64),
        }
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 100.0]])
    behavior = torch.tensor([[0.0, 0.0], [math.log(1.0 - mu_1), math.log(mu_1)], [0.0, 0.0]])
    return step_stream, *_preds(online, delayed, behavior)


def _v_pi(q: torch.Tensor, temperature: float = _T) -> float:
    """``E_π Q + temperature H[π]`` for a 1-D Q row."""
    pi = _softmax_policy(q.unsqueeze(0), temperature=temperature)[0]
    return float((pi * q).sum() + float(temperature) * _policy_entropy(pi.unsqueeze(0)))


_V_S1 = _v_pi(torch.tensor([3.0, 0.0]))
_V_S2 = _v_pi(torch.tensor([0.0, 100.0]))
_G1 = 10.0 + _V_S2
_ONE_STEP_G0 = 1.0 + _V_S1
_FULL_G0 = _ONE_STEP_G0 + _G1
_HALF_G0 = _ONE_STEP_G0 + 0.5 * _G1
# (0 - G_1)^2 for the s1 row in every temperature=_T case.
_S1_SQ = (0.0 - _G1) ** 2


def _nll_loss(mu_1: float) -> float:
    """Behavior-head NLL over the two in-run rows: -log 0.5 at s0, -log mu_1 at s1."""
    return (math.log(2.0) - math.log(mu_1)) / 2


def test_retrace_requires_lambda_temperature_weight_and_grouping_field() -> None:
    with pytest.raises(TypeError, match="td_lambda"):
        RetraceObjective(temperature=1.0, behavior_weight=1.0, grouping_field=None, discount=_disc(), reward=_rew(), value=_val())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="temperature"):
        RetraceObjective(td_lambda=1.0, behavior_weight=1.0, grouping_field=None, discount=_disc(), reward=_rew(), value=_val())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="behavior_weight"):
        RetraceObjective(td_lambda=1.0, temperature=1.0, grouping_field=None, discount=_disc(), reward=_rew(), value=_val())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="grouping_field"):
        RetraceObjective(td_lambda=1.0, temperature=1.0, behavior_weight=1.0, discount=_disc(), reward=_rew(), value=_val())  # type: ignore[call-arg]


def test_retrace_rejects_out_of_range_hyperparameters() -> None:
    with pytest.raises(ValueError, match=r"td_lambda must be in \[0, 1\]"):
        _retrace(td_lambda=1.5)
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        _retrace(temperature=-0.1)
    with pytest.raises(ValueError, match="behavior_weight must be >= 0"):
        _retrace(behavior_weight=-0.1)


def test_retrace_objective_runs() -> None:
    n, a = 8, 3
    step_stream = {
            "action": torch.randint(0, a, (n,)),
            "reward": torch.randn(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
            "sequence_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        }
    predictions, delayed = _preds(torch.randn(n, a), torch.randn(n, a), torch.randn(n, a))
    loss, metrics = _retrace(temperature=1.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert loss.ndim == 0
    assert metrics["td_loss"] >= 0.0
    assert metrics["behavior_loss"] >= 0.0
    assert abs(metrics["retrace"] - (metrics["td_loss"] + metrics["behavior_loss"])) < 1e-5
    assert 0.0 <= metrics["retrace_ratio_mean"] <= 1.0
    assert 0.0 <= metrics["behavior_prob_mean"] <= 1.0
    assert "cql_penalty" not in metrics
    assert "entropy" in metrics


def test_softmax_policy_matches_get_action_convention() -> None:
    q = torch.tensor([[0.0, math.log(3.0)]])
    assert torch.allclose(_softmax_policy(q, temperature=1.0), torch.tensor([[0.25, 0.75]]))
    # Halving the temperature squares the odds: 3 → 9.
    assert torch.allclose(_softmax_policy(q, temperature=0.5), torch.tensor([[0.1, 0.9]]))
    # temperature=0 is greedy with argmax ties sharing the mass.
    tied = torch.tensor([[1.0, 3.0, 3.0, 0.0]])
    assert torch.allclose(_softmax_policy(tied, temperature=0.0), torch.tensor([[0.0, 0.5, 0.5, 0.0]]))


def test_retrace_lambda_zero_is_the_expected_one_step_target() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    # G_0 = 1 + V_π(s1).
    _, metrics = _retrace(td_lambda=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics["td_loss"] - ((5.0 - _ONE_STEP_G0) ** 2 + _S1_SQ) / 2) < 1e-3


def test_retrace_on_policy_action_keeps_the_full_trace() -> None:
    """π(a_1|s_1) = 0.25 ≥ μ = 0.25 → ratio 1: G_0 = V one-step + G_1."""
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    _, metrics = _retrace()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics["td_loss"] - ((5.0 - _FULL_G0) ** 2 + _S1_SQ) / 2) < 1e-2
    # Pair 0: π(s0) is a tie → 0.5 / μ 0.5 = 1. Pair 1: 0.25 / 0.25 = 1.
    assert abs(metrics["retrace_ratio_mean"] - 1.0) < 1e-6
    assert abs(metrics["behavior_prob_mean"] - (0.5 + 0.25) / 2) < 1e-6


def test_retrace_off_policy_action_truncates_the_trace() -> None:
    """π(a_1|s_1) = 0.25 < μ = 0.5 → ratio 0.5: G_0 = one-step + 0.5 G_1."""
    step_stream, predictions, delayed = _fixture(mu_1=0.5)
    _, metrics = _retrace()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics["td_loss"] - ((5.0 - _HALF_G0) ** 2 + _S1_SQ) / 2) < 1e-2
    assert abs(metrics["retrace_ratio_mean"] - 0.75) < 1e-6


def test_retrace_lambda_scales_the_ratio() -> None:
    """λ = 0.5 with ratio 1 is the same trace as λ = 1 with ratio 0.5."""
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    _, metrics = _retrace(td_lambda=0.5)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics["td_loss"] - ((5.0 - _HALF_G0) ** 2 + _S1_SQ) / 2) < 1e-2


def test_retrace_greedy_target_cuts_non_greedy_actions() -> None:
    """temperature = 0: delayed argmax at s1 is 0, taken a_1 = 1 → the trace is cut."""
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    _, metrics = _retrace(temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    one_step, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(),
        grouping_field=None,
        temperature=0.0,
    )(
        objective_data=step_stream,
        predictions=predictions["action_value"],
        delayed_predictions=delayed["action_value"],
    )
    assert abs(metrics["td_loss"] - one_step.item()) < 1e-4
    # a_0 = 0 is a greedy tie at s0 (π = 0.5 ≥ μ = 0.5); a_1 is not greedy.
    assert abs(metrics["retrace_ratio_mean"] - 0.5) < 1e-6


def test_retrace_greedy_target_on_greedy_data_is_the_full_return() -> None:
    """temperature = 0 with a greedy a_1 is Watkins's Q(λ): G_0 = 1 + 10 + 100 = 111."""
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    delayed = {key: value.clone() for key, value in delayed.items()}
    q = delayed["action_value"].clone()
    q[1] = torch.tensor([0.0, 3.0])  # greedy at s1 is the taken action
    delayed["action_value"] = q
    _, metrics = _retrace(temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # (5 - 111)^2 = 11236; s1: (0 - 110)^2 = 12100.
    assert abs(metrics["td_loss"] - (11236.0 + 12100.0) / 2) < 1e-3


def test_retrace_total_loss_adds_weighted_behavior_nll() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    loss, metrics = _retrace(behavior_weight=2.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics["behavior_loss"] - _nll_loss(0.25)) < 1e-5
    assert abs(loss.item() - (metrics["td_loss"] + 2.0 * _nll_loss(0.25))) < 1e-2


def test_retrace_zero_behavior_weight_is_td_loss_only() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    loss, metrics = _retrace(behavior_weight=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics["behavior_loss"] - _nll_loss(0.25)) < 1e-5
    assert abs(loss.item() - metrics["td_loss"]) < 1e-5


def test_retrace_behavior_head_is_trained_and_gives_no_td_gradient() -> None:
    """The NLL reaches the behavior logits; the trace (μ) is detached."""
    step_stream, _, _ = _fixture(mu_1=0.25)
    torch.manual_seed(0)
    online = torch.randn(3, 2, requires_grad=True)
    delayed = torch.randn(3, 2, requires_grad=True)
    behavior = torch.randn(3, 2, requires_grad=True)
    predictions, delayed_td = _preds(online, delayed, behavior)
    loss, _ = _retrace(behavior_weight=3.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_td)
    loss.backward()
    assert online.grad is not None
    assert delayed.grad is None
    assert behavior.grad is not None
    # Behavior's gradient is exactly that of 3 * mean NLL of log_softmax(logits)
    # over the two in-run rows (rows 0 / 1 predict a_0 = 0 / a_1 = 1; row 2 has
    # weight 0).
    ref = behavior.detach().clone().requires_grad_(True)
    log_mu = torch.nn.functional.log_softmax(ref[:2], dim=-1)
    nll = torch.nn.functional.nll_loss(log_mu, torch.tensor([0, 1]), reduction="sum") / 2
    (3.0 * nll).backward()
    assert ref.grad is not None
    assert torch.allclose(behavior.grad, ref.grad, atol=1e-6)


def test_retrace_does_not_cross_sequence_boundary() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["sequence_id"] = torch.tensor([0, 0, 1])
    _, metrics = _retrace()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # Only pair 0 is in-run and it cannot continue: G_0 = one-step V.
    assert abs(metrics["td_loss"] - (5.0 - _ONE_STEP_G0) ** 2) < 1e-4
    assert abs(metrics["behavior_loss"] - math.log(2.0)) < 1e-5


def test_retrace_terminal_gamma_zero_ends_the_trace() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["episode_done"] = torch.tensor([0, 0, 1])
    _, metrics = _retrace()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # γ_1 = 0 → G_1 = 10; G_0 = one-step + 10.
    g0 = _ONE_STEP_G0 + 10.0
    assert abs(metrics["td_loss"] - ((5.0 - g0) ** 2 + 100.0) / 2) < 1e-3


def test_retrace_truncation_gamma_carries_the_trace_discounted() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["episode_done"] = torch.tensor([0, 0, 2])
    _, metrics = _retrace(discount=_disc(gamma_episode_truncated=0.5))(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # γ_1 = 0.5 → G_1 = 10 + 0.5 V_π(s2); G_0 = one-step + G_1.
    g1 = 10.0 + 0.5 * _V_S2
    g0 = _ONE_STEP_G0 + g1
    assert abs(metrics["td_loss"] - ((5.0 - g0) ** 2 + g1**2) / 2) < 1e-2


def test_retrace_with_multiple_head_output_rows_per_step() -> None:
    """Every row of a step trains toward its target; per-step reads use the last row."""
    step_stream, _, _ = _fixture(mu_1=0.25)
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["head_output_count"] = torch.tensor([2, 1, 2])
    online = torch.tensor([[5.0, 0.0], [7.0, 0.0], [0.0, 0.0], [0.0, -9.0], [0.0, 0.0]])
    delayed_q = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-9.0, -9.0], [0.0, 100.0]])
    # First s0 row would give μ = 0.9 for a_0; the last row (uniform) is the one read.
    behavior = torch.tensor(
        [[math.log(0.9), math.log(0.1)], [0.0, 0.0], [math.log(0.75), math.log(0.25)], [0.0, 0.0], [0.0, 0.0]]
    )
    predictions, delayed = _preds(online, delayed_q, behavior)
    _, metrics = _retrace()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    expected = ((5.0 - _FULL_G0) ** 2 + (7.0 - _FULL_G0) ** 2 + _S1_SQ) / 3
    assert abs(metrics["td_loss"] - expected) < 2e-2
    assert abs(metrics["behavior_prob_mean"] - (0.5 + 0.25) / 2) < 1e-6
    # BC over the three in-run rows: -log 0.9, -log 0.5, -log 0.25.
    bc = (-math.log(0.9) + math.log(2.0) + math.log(4.0)) / 3
    assert abs(metrics["behavior_loss"] - bc) < 1e-5


def test_retrace_q_affine_applies_to_online_and_delayed() -> None:
    """``value`` doubles online and delayed Q; π (raw Q) and μ are unchanged."""
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    _, metrics = _retrace(value=_val(scale=2.0))(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # π from raw Q; affine Q is doubled, so V = 2 E_π Q + T H[π].
    v1 = _v_pi(torch.tensor([3.0, 0.0])) - 2.25 + 4.5
    v2 = _v_pi(torch.tensor([0.0, 100.0])) - 100.0 + 200.0
    g1 = 10.0 + v2
    g0 = 1.0 + v1 + g1
    assert abs(metrics["td_loss"] - ((10.0 - g0) ** 2 + g1**2) / 2) < 5e-2


def test_retrace_requires_behavior_predictions() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    with pytest.raises(TypeError, match="behavior_predictions"):
        RetraceObjective(
            td_lambda=1.0, temperature=_T, behavior_weight=1.0, grouping_field=None,
            discount=_disc(), reward=_rew(), value=_val(),
        )(
            objective_data=step_stream,
            predictions=predictions["action_value"],
            delayed_predictions=delayed["action_value"],
        )  # type: ignore[call-arg]


def test_retrace_rejects_misaligned_behavior_head() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    predictions = {key: value.clone() for key, value in predictions.items()}
    predictions["behavior"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="head shape"):
        _retrace()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    predictions["behavior"] = torch.zeros(3, 2, dtype=torch.float64)
    with pytest.raises(TypeError, match="float32"):
        _retrace()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)


def test_retrace_requires_delayed_predictions() -> None:
    step_stream, predictions, _ = _fixture(mu_1=0.25)
    with pytest.raises(TypeError, match="delayed_predictions"):
        RetraceObjective(
            td_lambda=1.0, temperature=_T, behavior_weight=1.0, grouping_field=None,
            discount=_disc(), reward=_rew(), value=_val(),
        )(
            objective_data=step_stream,
            predictions=predictions["action_value"],
            behavior_predictions=predictions["behavior"],
        )  # type: ignore[call-arg]


def test_retrace_cql_penalty_metric() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    _, plain = _retrace()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    _, with_cql = _retrace(cql_weight=1.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert "cql_penalty" in with_cql
    assert with_cql["td_loss"] > plain["td_loss"]


def test_retrace_temperature_zero_omits_entropy_metric() -> None:
    step_stream, predictions, delayed = _fixture(mu_1=0.25)
    _, metrics = _retrace(temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert "entropy" not in metrics
