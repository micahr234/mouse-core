"""Action prompts are ordinary optional token fields; all step labels survive."""
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer as Backend
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from mouse_core.data import EventLog, Tokenizer, pack_token_batch, save_tokenizer, load_tokenizer
from mouse_core.models import Embedding, prepare_decode, sample_reasoning_splits
from mouse_core.objectives import DqnObjective, PpoObjective, GrpoObjective, boundary_discount


def backend():
    return PreTrainedTokenizerFast(tokenizer_object=Backend(WordLevel(
        {'[UNK]': 0, 'obs': 1, 'action': 2, 'reward': 3, 'prefix': 4}, unk_token='[UNK]')),
        unk_token='[UNK]')


def on_group_start(ctx):
    return ctx['group_start']


def tokenizer():
    return Tokenizer(tokenizers={"default": backend()}, input_fields=[
        dict(type='token', provider='environment', format='prefix', when=on_group_start),
        dict(type='token', provider='agent', input_field='action', format='action'),
        dict(type='token', provider='environment', input_field='observation', format='obs'),
        dict(type='token', provider='environment', input_field='reward', format='reward'),
        dict(type='token', provider='environment', input_field='action_requested',
             format='<|action_requested|>', special=True, head_output=True),
    ], objective_fields=[dict(input_field=name, missing_value=value) for name, value in
                        [('action', None), ('reward', 0.), ('terminated', False), ('truncated', False),
                         ('advantage', 1.), ('old_log_prob', 0.)]])


def episode():
    log = EventLog()
    log.event(provider='coordinator', episode_index=0, step_index=0, payload={'episode_start': True})
    for step, payload in enumerate([
        {'observation': 0, 'action_requested': True},
        {'observation': 1, 'reward': 2.},
        {'reward': 3.},
        {'observation': 2, 'action_requested': True},
        {'reward': 7., 'terminated': True},
    ]):
        if step in (1, 4):
            log.event(provider='agent', episode_index=0, step_index=step, payload={'action': 0})
        log.event(provider='environment', episode_index=0, step_index=step, payload=payload)
    return log


def test_special_tokens_sparse_steps_and_roundtrip(tmp_path):
    tok = tokenizer()
    special = tok.special_token_ids['default']['<|action_requested|>']
    assert tok.tokenizers['default']('<|action_requested|>', add_special_tokens=False)['input_ids'] == [special]
    assert tok.tokenizers['default'].decode([special], skip_special_tokens=True) == ''
    for source in (episode(), episode().to_dataset()):
        steps = tok.encode(source)
        assert [int(step.head_token_mask.sum()) for step in steps] == [1, 0, 0, 1, 0]
        packed, labels, groups = pack_token_batch(steps=steps, continuing=None)
        assert packed.N == 5 and packed.P == 2
        assert packed.head_step_ids.tolist() == [0, 3]
        assert packed.step_counts().tolist() == [5]
        assert packed.token_ids[packed.head_token_ids].tolist() == [special, special]
        assert packed.token_ids.tolist().count(4) == 1  # group prefix
        assert labels['reward'].tolist() == [0., 2., 3., 0., 7.]
        assert groups.tolist() == [0] * 5
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={"default": backend()})
    assert isinstance(restored, Tokenizer)
    assert restored.special_token_ids == tok.special_token_ids
    assert [step.token_ids.tolist() for step in restored.encode(episode())] == [step.token_ids.tolist() for step in tok.encode(episode())]
    changed = backend()
    changed.add_tokens(['another-token'])
    with pytest.raises(ValueError, match='Special token IDs differ'):
        load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={"default": changed})


def test_rewards_flow_through_steps_without_readouts():
    packed, labels, groups = pack_token_batch(steps=tokenizer().encode(episode()), continuing=None)
    q = torch.tensor([[1., 0.], [4., 0.]], requires_grad=True)
    discount = boundary_discount(gamma_step=.5, gamma_terminated=0., gamma_truncated=0.)
    dqn = DqnObjective(discount=discount, reward=None, value=None, temperature=0., gate=None,
                       cross_group_backups='fault')
    loss, metrics = dqn(objective_data=labels, group_id=groups, predictions=q,
                        backup_value_predictions=q.detach(), backup_action_predictions=q.detach())
    # 2 + .5 * 3 + .5**3 * 4; final decision receives the terminal reward.
    torch.testing.assert_close(metrics['backup'], torch.tensor([4., 7.]))
    torch.testing.assert_close(loss['action_value'], torch.tensor(9.))
    loss['action_value'].backward()
    assert q.grad is not None and (q.grad[:, 0] != 0).all()
    values = torch.tensor([1., 4.], requires_grad=True)
    logits = torch.zeros(packed.P, 2, requires_grad=True)
    ppo = PpoObjective(discount=discount, reward=None, value=None, gae_lambda=0.,
                       cross_group_backups='fault', normalize_advantage=False)
    loss, metrics = ppo(objective_data=labels, group_id=groups, predictions=logits, value_predictions=values)
    assert metrics['value_loss'] == pytest.approx(9.)
    loss['ppo'].backward()
    assert values.grad is not None and (values.grad != 0).all()
    grpo = GrpoObjective(clip_ratio_min=.8, clip_ratio_max=1.2)
    losses, _ = grpo(objective_data=labels, group_id=groups, predictions=torch.zeros(2, 2, requires_grad=True))
    assert all(torch.isfinite(value) for value in losses.values())


def test_cached_layout_keeps_passive_tokens_without_inventing_readouts():
    steps = tokenizer().encode(episode())
    tokens, _, _ = pack_token_batch(steps=[*steps, *steps[1:3]], sequence_ids=[0]*5+[1]*2, continuing=None)
    tensors = tokens.to_tensors()
    padded, lengths, ids, valid = prepare_decode(embeds=torch.ones(tokens.L, 3), tokens=tokens,
        sequence_ids=tensors['sequence_ids'], head_token_ids=tensors['head_token_ids'], head_step_ids=tensors['head_step_ids'])
    assert padded.shape[0] == 2 and min(lengths) > 0
    assert ids.shape == (2, 2) and valid.tolist() == [[True, True], [False, False]]
    for seed in range(10):
        splits = sample_reasoning_splits(batch=tokens, generator=np.random.default_rng(seed))
        assert splits[0] in (0, 3) and splits[1] == -1
    passive, _, _ = pack_token_batch(steps=steps[1:3], continuing=[True])
    t = passive.to_tensors()
    _, lengths, ids, valid = prepare_decode(embeds=torch.ones(passive.L, 3), tokens=passive,
        sequence_ids=t['sequence_ids'], head_token_ids=t['head_token_ids'], head_step_ids=t['head_step_ids'])
    assert lengths == [passive.L] and ids.shape == valid.shape == (1, 0)


def test_embedding_growth_preserves_old_rows_and_trains_special_row():
    embedding = Embedding(vocab_size=5, hidden_dim=4)
    old = embedding.embed_tokens.weight.detach().clone()
    embedding.resize_token_embeddings(6)
    torch.testing.assert_close(embedding.embed_tokens.weight[:5], old)
    embedding(torch.tensor([5])).sum().backward()
    assert embedding.embed_tokens.weight.grad is not None
    assert embedding.embed_tokens.weight.grad[5].abs().sum() > 0


def test_wildcard_fields_keep_order_and_group_prefix_once():
    tok = Tokenizer(tokenizers={"default": backend()}, input_fields=[
        dict(type='token', provider=None, format='prefix', when=on_group_start),
        dict(type='token', provider=None, input_field='value', format='obs'),
        dict(type='token', provider='agent', input_field='value', format='action', head_output=True),
    ])
    log = EventLog()
    log.event(provider='coordinator', episode_index=0, step_index=0, payload={})
    log.event(provider='agent', episode_index=0, step_index=0, payload={'value': 1})
    log.event(provider='sensor', episode_index=0, step_index=0, payload={'value': 2})
    log.event(provider='sensor', episode_index=0, step_index=1, payload={'value': 3})
    packed = tok.pack_rows(rows=[log], continuing=None)
    assert packed.token_ids.tolist() == [4, 1, 2, 1, 1]
    assert packed.head_step_ids.tolist() == [0]
    first = tok.pack_rows(rows=[log[:3]], continuing=None)
    rest = tok.pack_rows(rows=[log[3:]], continuing=[True])
    assert packed.token_ids.tolist() == first.token_ids.tolist() + rest.token_ids.tolist()


def test_event_augmentation_provider_filters_and_shared_permutations():
    from mouse_core.data import Augmenter, compose
    log = EventLog()
    log.event(provider='coordinator', episode_index=7, step_index=0, payload={'value': 2.})
    log.event(provider='environment', episode_index=7, step_index=0, payload={'q_star': [0., 1., 2., 3.], 'value': 2.})
    log.event(provider='agent', episode_index=7, step_index=1, payload={'action': 2, 'value': 2.})
    original = log.events
    augmenter = Augmenter(seed_field='episode_index', seed=12, fields=[
        dict(type='discrete', provider=None, input_field='action', input_vector_field='q_star',
             permute=True, vocab_size=4),
        dict(type='linear', provider='environment', input_field='value',
             scale_in_low=0., scale_in_high=1., scale_out_low=0., scale_out_high=2.),
    ])
    pipeline = compose(stages=(augmenter.events, list))
    pipeline.reseed(generation=9)
    events = pipeline(log)
    assert log.events == original
    assert events[0]['payload'] == original[0]['payload']
    assert events[1]['payload']['value'] == 4. and events[2]['payload']['value'] == 2.
    action = events[2]['payload']['action']
    assert events[1]['payload']['q_star'][action] == 2.
    assert 'action' not in events[1]['payload'] and 'q_star' not in events[2]['payload']
    pipeline.reseed(generation=9)
    replay = pipeline(log.to_dataset())
    for before, after in zip(events, replay, strict=True):
        assert before.keys() == after.keys()
        assert before['episode_index'] == after['episode_index']
        for key in before['payload']:
            np.testing.assert_array_equal(before['payload'][key], after['payload'][key])


def test_passive_only_step_window_has_no_prediction_loss():
    # A whole-step window can contain only automatic transitions.
    log = EventLog()
    for step in range(2):
        log.event(provider='environment', episode_index=0, step_index=step, payload={'reward': 1.})
    steps = tokenizer().encode(log)
    _, labels, groups = pack_token_batch(steps=steps, continuing=None)
    q = torch.empty(0, 2, requires_grad=True)
    objective = DqnObjective(discount=boundary_discount(gamma_step=.5, gamma_terminated=0., gamma_truncated=0.),
        reward=None, value=None, temperature=0., gate=None, cross_group_backups='ignore')
    losses, metrics = objective(objective_data=labels, group_id=groups, predictions=q,
        backup_value_predictions=q.detach(), backup_action_predictions=q.detach())
    assert isinstance(metrics['backup'], torch.Tensor)
    assert losses['action_value'].item() == 0 and metrics['backup'].numel() == 0
    losses['action_value'].backward()
    assert q.grad is not None and q.grad.numel() == 0


def test_field_order_and_independent_events_with_mergeable_text():
    from tokenizers.models import BPE

    hf = PreTrainedTokenizerFast(tokenizer_object=Backend(BPE(
        vocab={'[UNK]': 0, 'a': 1, 'b': 2, 'ab': 3},
        merges=[('a', 'b')], unk_token='[UNK]')), unk_token='[UNK]')
    tok = Tokenizer(tokenizers={"default": hf}, input_fields=[
        dict(type='text', provider=None, input_field='first', format='{field}'),
        dict(type='text', provider='environment', input_field='second', format='{field}'),
    ])
    assert hf('ab', add_special_tokens=False)['input_ids'] == [3]
    log = EventLog()
    # Payload order differs from configuration order; adjacent text could merge.
    log.event(provider='environment', episode_index=0, step_index=0,
              payload={'second': 'b', 'first': 'a'})
    log.event(provider='environment', episode_index=0, step_index=0, payload={'first': 'a'})
    log.event(provider='environment', episode_index=0, step_index=0, payload={'second': 'b'})
    log.event(provider='environment', episode_index=0, step_index=1, payload={'second': 'b'})
    independent = [tok(event)['token_ids'].tolist() for event in log]
    assert independent == [[1, 2], [1], [2], [2]]
    for source in (log, log.to_dataset()):
        assert [step.token_ids.tolist() for step in tok.encode(source)] == [[1, 2, 1, 2], [2]]
    stream = tok.stream()
    for event in log[:3]:
        assert stream.push(event) is None
    completed = stream.push(log[3])
    assert completed is not None and completed.token_ids.tolist() == [1, 2, 1, 2]
    assert tok(log[0])['token_ids'].tolist() == independent[0]


@pytest.mark.parametrize('payload', [{}, {'text': None, 'marker': None, 'number': None, 'image': None}])
def test_missing_inputs_skip_all_modalities(payload):
    def unexpected_image(value):
        raise AssertionError('Missing images must not be tokenized')

    tok = Tokenizer(tokenizers={'default': backend(), 'image': unexpected_image}, input_fields=[
        dict(type='text', input_field='text', format='{field}'),
        dict(type='token', input_field='marker', format='action', head_output=True),
        dict(type='numeric', input_field='number', format='reward', fourier_min=0., fourier_max=1.),
        dict(type='image', tokenizer='image', input_field='image'),
    ])
    assert tok({'provider': 'environment', 'payload': payload})['token_ids'].size == 0
    with pytest.raises(TypeError, match='real number'):
        tok({'provider': 'environment', 'payload': {'number': 'invalid'}})


def test_existing_action_marker_does_not_extend_vocabulary():
    hf = backend()
    size = len(hf)
    tok = Tokenizer(tokenizers={"default": hf}, input_fields=[
        dict(type='token', input_field='action_requested', format='action', head_output=True),
    ])
    assert len(hf) == size and tok.special_token_ids == {}
    event = {'provider': 'environment', 'payload': {'action_requested': True}}
    assert tok(event)['token_ids'].tolist() == [2]
    assert tok(event)['head_token_mask'].tolist() == [True]
