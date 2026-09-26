from __future__ import annotations

import torch

from mouse_core.objectives import SpObjective, best_action
from mouse_core.objectives.sp import sp_ce


def _episode_done(*shape: int, fill: int = 0) -> torch.Tensor:
    return torch.full(shape, fill, dtype=torch.int64)


def _ce_targets(q: torch.Tensor) -> torch.Tensor:
    """Hard-CE target action ids from Q* via ``best_action``."""
    return best_action(q.reshape(-1, q.shape[-1])).reshape(q.shape[:-1])


def test_sp_objective_ce_uses_best_action_ids() -> None:
    q = torch.tensor([[[0.0, 1.0, -torch.inf]]])
    objective_data = {"episode_done": _episode_done(*q.shape[:-1])}
    predictions = torch.tensor([[[0.0, 1.0, 100.0]]])
    loss, metrics = SpObjective()(
        objective_data=objective_data,
        predictions=predictions,
        targets=_ce_targets(q),
    )
    assert loss.ndim == 0
    assert metrics["action"] >= 0.0


def test_sp_objective_ce_skips_nonzero_mask_rows() -> None:
    """Any nonzero mask (terminated=1, truncated=2, ...) drops the row."""
    q = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    logits = torch.tensor([[[0.0, 100.0], [100.0, 0.0], [0.0, 100.0]]])
    objective_data = {"episode_done": torch.tensor([[1, 2, 0]], dtype=torch.int64)}
    loss, _ = SpObjective()(
        objective_data=objective_data,
        predictions=logits,
        targets=_ce_targets(q),
    )
    assert loss.item() < 1e-4


def test_sp_objective_ce_mask_key_none_keeps_terminals() -> None:
    predictions = torch.tensor([[[0.0, 100.0]]])
    loss, _ = SpObjective(mask_key=None)(
        objective_data={},
        predictions=predictions,
        targets=torch.tensor([[0]]),
    )
    assert loss.item() > 1.0


def test_sp_objective_ce_rejects_out_of_range_actions() -> None:
    objective_data = {"episode_done": _episode_done(1, 1)}
    predictions = torch.tensor([[[0.0, 1.0]]])
    try:
        SpObjective()(
            objective_data=objective_data,
            predictions=predictions,
            targets=torch.tensor([[3]]),
        )
    except ValueError as exc:
        assert "action ids must be in" in str(exc)
    else:
        raise AssertionError("expected ValueError for out-of-range action")


def test_sp_objective_requires_targets() -> None:
    objective_data = {"episode_done": _episode_done(1, 1)}
    predictions = torch.tensor([[[0.0, 1.0]]])
    try:
        SpObjective()(objective_data=objective_data, predictions=predictions)
    except TypeError as exc:
        assert "targets" in str(exc)
    else:
        raise AssertionError("expected TypeError for missing targets")


def test_best_action_unique_max_is_deterministic() -> None:
    q = torch.tensor([[0.0, 2.0, 1.0, -torch.inf], [3.0, 1.0, 3.0 - 1e-6, -torch.inf]])
    for _ in range(20):
        assert best_action(q).tolist() == [1, 0]


def test_best_action_samples_uniformly_among_maxima() -> None:
    torch.manual_seed(0)
    q = torch.tensor([[1.0, 1.0, 0.5, -torch.inf]])
    seen = {best_action(q).item() for _ in range(80)}
    assert seen == {0, 1}


def test_best_action_never_selects_padding() -> None:
    torch.manual_seed(0)
    q = torch.tensor([[-torch.inf, 2.0, 2.0]])
    for _ in range(40):
        assert best_action(q).item() in (1, 2)


def test_sp_ce_matches_cross_entropy_on_unique_max() -> None:
    actions = torch.tensor([1])
    logits = torch.tensor([[0.5, -1.0, 2.0]])
    expected = torch.nn.functional.cross_entropy(logits, actions)
    assert torch.allclose(sp_ce(target_actions=actions, logits=logits), expected)
    smoothed = torch.nn.functional.cross_entropy(logits, actions, label_smoothing=0.2)
    assert torch.allclose(
        sp_ce(target_actions=actions, logits=logits, label_smoothing=0.2), smoothed
    )


def test_sp_ce_ignores_padded_student_logits() -> None:
    """A junk student logit at a padded slot must not affect the hard CE loss."""
    actions = torch.tensor([1])
    invalid = torch.tensor([[False, False, False, True]])
    logits = torch.tensor([[0.5, -1.0, 2.0, 100.0]], requires_grad=True)
    clean = sp_ce(target_actions=actions, logits=logits[:, :3])
    padded = sp_ce(target_actions=actions, logits=logits, invalid=invalid)
    assert torch.allclose(padded, clean)
    clean_smoothed = sp_ce(target_actions=actions, logits=logits[:, :3], label_smoothing=0.1)
    padded_smoothed = sp_ce(
        target_actions=actions, logits=logits, label_smoothing=0.1, invalid=invalid
    )
    assert torch.isfinite(padded_smoothed)
    assert torch.allclose(padded_smoothed, clean_smoothed)
    padded_smoothed.backward()
    assert logits.grad is not None
    assert logits.grad[0, -1].item() == 0.0


def test_sp_objective_accepts_direct_action_ids() -> None:
    objective_data = {"episode_done": _episode_done(1, 1)}
    predictions = torch.tensor([[[0.0, 1.0]]])
    loss, _ = SpObjective()(
        objective_data=objective_data,
        predictions=predictions,
        targets=torch.tensor([[1]]),
    )
    assert loss.item() > 0.0
