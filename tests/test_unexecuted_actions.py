"""Reset placeholders must not become decisions, even inside a live episode."""
import pytest
import torch

from mouse_core.objectives import (
    DqnObjective, GrpoObjective, PpoObjective, boundary_discount,
    lambda_gate, nstep_gate, value_gap_gate, general_gate,
)


def data():
    return {
        'action': torch.tensor([-999, 0, 999, 1, 0]),
        'action_executed': torch.tensor([False, True, False, True, True]),
        'reward': torch.tensor([0., 2., 7., 3., 4.]),
        'terminated': torch.tensor([False, False, False, False, True]),
        'truncated': torch.zeros(5, dtype=torch.bool),
        'old_log_prob': torch.tensor([float('nan'), -.7, float('nan'), -.7, -.7]),
        'advantage': torch.tensor([float('nan'), 1., float('nan'), -1., 1.]),
    }


def objective(gate=None):
    return DqnObjective(
        reward=None, value=None,
        discount=boundary_discount(gamma_step=.9, gamma_terminated=0., gamma_truncated=0.),
        temperature=0., gate=gate, cross_group_backups='fault',
    )


@pytest.mark.parametrize('gate', [None, lambda_gate(td_lambda=.5), nstep_gate(n=2),
    value_gap_gate(beta=0., bias=0., normalize=False, policy_delayed=False, value_delayed=False),
    general_gate(gates=[nstep_gate(n=2), lambda_gate(td_lambda=.7)])])
def test_dead_actions_and_dead_q_cannot_change_dqn(gate):
    batch = data()
    q = torch.tensor([[1., 2.], [float('nan')]*2, [3., 4.], [5., 6.], [9., 10.]], requires_grad=True)
    target = torch.tensor([[1., 2.], [float('nan')]*2, [4., 5.], [5., 6.], [9., 10.]])
    losses, metrics = objective(gate)(objective_data=batch, group_id=torch.zeros(5,dtype=torch.long),
        predictions=q, backup_value_predictions=target, backup_action_predictions=target)
    assert torch.isfinite(losses['action_value'])
    assert metrics['backup_weight'].tolist() == [1., 0., 1., 1., 0.]
    losses['action_value'].backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad[1].count_nonzero() == 0
    changed = {**batch, 'action': torch.tensor([42, 0, -123, 1, 0])}
    other, other_metrics = objective(gate)(objective_data=changed, group_id=torch.zeros(5,dtype=torch.long),
        predictions=q.detach().nan_to_num(12345.), backup_value_predictions=target.nan_to_num(-999.),
        backup_action_predictions=target.nan_to_num(999.))
    torch.testing.assert_close(losses['action_value'], other['action_value'])
    torch.testing.assert_close(metrics['backup'], other_metrics['backup'])
    if gate is None:
        torch.testing.assert_close(metrics['backup'][[0, 2, 3]], torch.tensor([12.35, 8.4, 4.]))


def test_nstep_counts_executed_actions_only():
    batch = data()
    gate = nstep_gate(n=2)(q=torch.zeros(5,2), **batch)
    assert gate[0].tolist() == [0., 1., 1., 0., 0.]


@pytest.mark.parametrize('policy', ['grpo', 'ppo'])
def test_policy_loss_ignores_invalid_ids_nan_logits_and_stale_reset_logprobs(policy):
    batch = data()
    scores = torch.tensor([[.1,.3], [float('nan')]*2, [.2,.1], [.7,.1], [.0,.0]], requires_grad=True)
    kwargs = dict(objective_data=batch, group_id=torch.zeros(5,dtype=torch.long), predictions=scores)
    obj = GrpoObjective() if policy=='grpo' else PpoObjective(
        discount=boundary_discount(gamma_step=.9,gamma_terminated=0.,gamma_truncated=0.),
        reward=None,value=None,cross_group_backups='fault')
    if policy == 'ppo':
        kwargs['value_predictions'] = torch.arange(5,dtype=torch.float32,requires_grad=True)
    losses, metrics = obj(**kwargs)
    loss=sum(losses.values())
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.as_tensor(v)) for v in metrics.values())
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[1].count_nonzero() == 0


def test_mask_does_not_allow_invalid_executed_actions():
    batch = data(); batch['action'][1] = 999
    with pytest.raises(ValueError, match='action ids'):
        objective()(objective_data=batch,group_id=torch.zeros(5,dtype=torch.long),
            predictions=torch.zeros(5,2),backup_value_predictions=torch.zeros(5,2),backup_action_predictions=torch.zeros(5,2))


def test_automatic_transitions_never_cross_episodes_and_multi_readouts_match():
    batch = data()
    joined = {k:torch.cat([v,v]) for k,v in batch.items()}
    q = torch.arange(10,dtype=torch.float32).reshape(5,2)
    obj=objective(nstep_gate(n=20))
    _,single=obj(objective_data=batch,group_id=torch.zeros(5,dtype=torch.long),
        predictions=q,backup_value_predictions=q,backup_action_predictions=q)
    q2=torch.cat([q,q+1000])
    joined['head_output_count']=torch.full((10,),2,dtype=torch.long)
    q2=q2.repeat_interleave(2,dim=0)
    _,combined=obj(objective_data=joined,group_id=torch.tensor([0]*5+[1]*5),
        predictions=q2,backup_value_predictions=q2,backup_action_predictions=q2)
    torch.testing.assert_close(combined['backup'][:10],single['backup'].repeat_interleave(2))
    assert combined['backup_weight'][8:10].count_nonzero()==0


@pytest.mark.parametrize('policy', ['dqn','grpo'])
def test_all_unexecuted_returns_differentiable_zero(policy):
    batch={k:v[:2].clone() for k,v in data().items()}
    batch['action_executed'][:]=False
    q=torch.full((2,2),float('nan'),requires_grad=True)
    kwargs=dict(objective_data=batch,group_id=torch.zeros(2,dtype=torch.long),predictions=q)
    if policy=='dqn':
        kwargs.update(backup_value_predictions=q,backup_action_predictions=q)
        obj=objective()
    else: obj=GrpoObjective()
    losses,_=obj(**kwargs)
    loss=sum(losses.values())
    assert loss==0
    loss.backward()
    assert q.grad.count_nonzero()==0


@pytest.mark.parametrize('algorithm', ['dqn','grpo','ppo'])
def test_binary_int64_action_flags_from_packed_data(algorithm):
    batch=data();batch['action_executed']=batch['action_executed'].long()
    q=torch.zeros(5,2,requires_grad=True)
    kwargs=dict(objective_data=batch,group_id=torch.zeros(5,dtype=torch.long),predictions=q)
    if algorithm=='dqn':
        obj=objective();kwargs.update(backup_value_predictions=q,backup_action_predictions=q)
    elif algorithm=='grpo':obj=GrpoObjective()
    else:
        obj=PpoObjective(reward=None,value=None,discount=None,cross_group_backups='bootstrap')
        kwargs['value_predictions']=torch.zeros(5,requires_grad=True)
    losses,_=obj(**kwargs)
    loss=sum(losses.values());assert torch.isfinite(loss)
    loss.backward();assert q.grad[1].count_nonzero()==0
    batch['action_executed'][1]=2
    with pytest.raises(ValueError,match='binary'):
        obj(**kwargs)
