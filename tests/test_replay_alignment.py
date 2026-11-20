"""Recorded actions align to their originating predictions, not arrival readouts."""
import pytest
import torch
from mouse_core.objectives import replay_action_indices


def test_multiple_readouts_missing_requests_and_episode_boundaries():
    data = {"action": torch.tensor([float("nan"), 3., float("nan"), 1., 2., 0.])}
    groups = torch.tensor([0, 0, 0, 0, 1, 1])
    steps = torch.tensor([0, 0, 2, 3, 4, 5])
    rows, actions = replay_action_indices(data, groups, step_ids=steps, readout_shape=(6,))
    assert rows.tolist() == [0, 1, 2, 4]
    assert actions.tolist() == [3, 3, 1, 0]
    q = torch.arange(24.).reshape(6, 4).requires_grad_()
    q[rows, actions].sum().backward()
    assert q.grad.nonzero().tolist() == [[0, 3], [1, 3], [2, 1], [4, 0]]


@pytest.mark.parametrize("size", [0, 1, 3])
def test_no_predictions_and_final_rows_have_no_outgoing_action(size):
    data = {"action": torch.zeros(size, dtype=torch.int64)}
    groups = torch.zeros(size, dtype=torch.int64)
    for steps in (torch.empty(0, dtype=torch.int64), torch.arange(size)):
        rows, actions = replay_action_indices(data, groups, step_ids=steps, readout_shape=steps.shape)
        assert len(rows) == max(0, len(steps) - 1)
        assert actions.dtype == torch.int64


@pytest.mark.parametrize("action", [1.5, float("inf"), -1.])
def test_invalid_recorded_action_cannot_be_silently_cast(action):
    with pytest.raises(ValueError):
        replay_action_indices({"action": torch.tensor([0., action])}, torch.zeros(2, dtype=torch.int64),
                              step_ids=torch.tensor([0]), readout_shape=(1,))


@pytest.mark.parametrize("steps,shape", [(torch.tensor([-1]), (1,)), (torch.tensor([2]), (1,)),
                                         (torch.tensor([0]), (2,))])
def test_bad_alignment_rejected(steps, shape):
    with pytest.raises(ValueError):
        replay_action_indices({"action": torch.zeros(2)}, torch.zeros(2, dtype=torch.int64),
                              step_ids=steps, readout_shape=shape)


@pytest.mark.parametrize('heads', [None, 2])
@pytest.mark.parametrize('action_key', ['action', 'reported_action'])
def test_dqn_and_explicit_selection_share_sparse_repeated_alignment(heads, action_key):
    from mouse_core.objectives import DqnObjective, boundary_discount
    data = {
        action_key: torch.tensor([float('nan'), 3., float('nan'), 1., 2., 0.]),
        'reward': torch.tensor([0., 1., .5, 3., 0., 5.]),
        'terminated': torch.tensor([False, False, False, True, False, True]),
        'truncated': torch.zeros(6, dtype=torch.bool),
    }
    groups = torch.tensor([0, 0, 0, 0, 1, 1])
    steps = torch.tensor([0, 0, 2, 3, 4, 5])
    shape = (6, 4) if heads is None else (6, heads, 4)
    q = (torch.arange(torch.tensor(shape).prod()).reshape(shape).float() / 10).requires_grad_()
    delayed = (q.detach() + 1).requires_grad_()
    objective = DqnObjective(action_key=action_key, reward=None, value=None, gate=None,
        temperature=0., discount=boundary_discount(gamma_step=.5, gamma_terminated=0., gamma_truncated=0.))
    kwargs = dict(objective_data=data, sequence_ids=groups, step_ids=steps,
                  backup_value_predictions=delayed, backup_action_predictions=delayed)
    dense_loss, dense_metrics = objective(predictions=q, **kwargs)
    rows, actions = replay_action_indices({'action': data[action_key]}, groups,
                                         step_ids=steps, readout_shape=(6,))
    explicit_q = q.detach().clone().requires_grad_()
    index = actions[:, None] if heads is None else actions[:, None, None].expand(-1, heads, 1)
    selected = explicit_q.new_zeros(shape[:-1]).index_copy(
        0, rows, explicit_q[rows].gather(-1, index).squeeze(-1))
    selected_loss, selected_metrics = objective(predictions=selected, **kwargs)
    torch.testing.assert_close(dense_loss['action_value'], selected_loss['action_value'])
    for key in ('backup', 'backup_weight'):
        torch.testing.assert_close(dense_metrics[key], selected_metrics[key])
    # Missing readout at step 1 is an automatic transition: the backup passes
    # through its reward to step 2. The last rows of each sequence never train.
    first_target = 1 + .5 * .5 + .25 * delayed.detach()[2].amax(-1)
    targets = torch.stack([first_target, first_target, torch.full_like(first_target, 3.),
                           torch.full_like(first_target, 5.)])
    taken = q[rows].gather(-1, index).squeeze(-1)
    torch.testing.assert_close(dense_loss['action_value'], (taken - targets).square().mean(0))
    dense_loss['action_value'].sum().backward()
    selected_loss['action_value'].sum().backward()
    torch.testing.assert_close(q.grad, explicit_q.grad)
    assert q.grad[[3, 5]].count_nonzero() == 0
    assert delayed.grad is None
