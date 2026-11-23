"""One decision contract for text spans, learned inputs, and both generators."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
from mouse_core.data import Tokenizer, VocabularyTokenizer, pack_token_batch, decision_positions, align_decision_objectives, save_tokenizer, load_tokenizer
from mouse_core.models import Backbone, Embedding, MultiEmbedding, SequentialReasoner
from mouse_core.models.autoregressive_reasoner import AutoregressiveReasoner
from tests.test_residual_reasoning import CausalBackbone

class Digits:
    def __call__(self, text, **kwargs):
        return {'input_ids': [int(word) for word in text.split()]}


def tokenizer(readout):
    return Tokenizer(tokenizers={'text': Digits(), 'reasoning': VocabularyTokenizer(['think', 'other'])},
        input_fields=[dict(type='text', tokenizer='text', input_field='observation', format='{observation}'),
                      dict(type='text', tokenizer='text', input_field='query', format='{query}', head_output=readout),
                      dict(type='token', tokenizer='reasoning', input_field='latent', format='{latent}', head_output='last')],
        objective_fields=[dict(input_field='action', missing_value=None)])


def events():
    return [dict(provider='environment', step_index=i, episode_index=0, payload=payload) for i,payload in enumerate([
        {'observation':'2 3', 'query':'4 5 6'}, {'observation':'7', 'query':'4 5 6', 'action':1}])]

@pytest.mark.parametrize('readout,heads', [('none',[]),('last',[4,8]),('all',[2,3,4,6,7,8])])
def test_multitoken_fields_keep_spans_and_one_decision(readout,heads):
    inputs,labels=pack_token_batch(steps=tokenizer(readout).encode(events()))
    assert inputs['head_mask'].nonzero().flatten().tolist()==heads
    assert decision_positions(inputs).tolist()==([] if readout=='none' else [4,8])
    assert inputs['field_start_mask'].nonzero().flatten().tolist()==[0,2,5,6]
    assert inputs['field_end_mask'].nonzero().flatten().tolist()==[1,4,5,8]
    aligned=align_decision_objectives(inputs,labels)
    assert aligned['step_ids'].tolist()==([] if readout=='none' else [0,1])


def test_symbolic_vocabulary_roundtrips_without_text_tokenizer_or_vocab_changes(tmp_path):
    tok=tokenizer('last')
    save_tokenizer(tokenizer=tok,path=tmp_path)
    restored=load_tokenizer(repo_id_or_path=str(tmp_path),tokenizers={'text':Digits()})
    assert isinstance(restored.tokenizers['reasoning'],VocabularyTokenizer)
    assert restored.tokenizers['reasoning'].get_vocab()=={'think':0,'other':1}
    event=dict(provider='environment',payload={'latent':'other'})
    result=restored(event)
    assert result['token_ids'].tolist()==[1]
    assert result['tokenizer_ids'].tolist()==[0]
    with pytest.raises(ValueError,match='Unknown symbolic'):
        restored(dict(provider='environment',payload={'latent':'missing'}))
    with pytest.raises(TypeError,match='autoregressive'):
        Tokenizer(tokenizers={'text':Digits()}, input_fields=[dict(type='token',format='9',autoregressive=True)])


@pytest.mark.parametrize('backend',['uncached','continuous'])
@torch.no_grad()
def test_text_reasoning_uses_final_boundary_and_replays_prior_generated_history(backend):
    torch.manual_seed(4)
    inputs,_=pack_token_batch(steps=tokenizer('all').encode(events()))
    embedding=MultiEmbedding(embeddings={'text':Embedding(vocab_size=10,hidden_dim=8),
                                       'reasoning':Embedding(vocab_size=2,hidden_dim=8)})
    backbone=Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2,
                      head_dim=4, max_position_embeddings=128, dtype=torch.float32)
    head=nn.Linear(8,10)
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_();head.bias[0]=10
    reasoner=AutoregressiveReasoner(head)
    seen=[]
    handle=head.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach().clone()))
    traces=reasoner.generate(inputs,embedding=embedding,backbone=backbone,tokenizer_name='text',
        max_new_tokens=3,end_token_id=0,backend=backend,max_active_requests=2,do_sample=False,temperature=1.)
    handle.remove()
    assert [trace.head_token_id for trace in traces]==[4,8]
    out=reasoner(inputs,traces,embedding=embedding,backbone=backbone,tokenizer_name='text',temperature=1.,score_tokens=True)
    torch.testing.assert_close(out.value_hidden,torch.cat(seen))
    # Literal full sequence: earlier generated stop remains before the next observed segment.
    ids=torch.tensor([2,3,4,5,6,0,7,4,5,6,0])
    encoded=embedding.embeddings['text'](ids)
    hidden=backbone(encoded,sequence_ids=torch.zeros(len(ids),dtype=torch.long))
    torch.testing.assert_close(out.final_hidden,hidden[[5,10]])



def test_distinct_learned_inputs_are_live_in_online_and_target_replay():
    tok=tokenizer('last')
    rows=[dict(provider='environment',step_index=i,payload={'observation':'2','latent':name})
          for i,name in enumerate(['think','other'])]
    inputs,_=pack_token_batch(steps=tok.encode(rows))
    embedding=MultiEmbedding(embeddings={'text':Embedding(vocab_size=10,hidden_dim=8),
                                       'reasoning':Embedding(vocab_size=2,hidden_dim=8)})
    backbone=CausalBackbone()
    reasoner=SequentialReasoner(hidden_dim=8,input_mlp_intermediate_dim=24,input_mlp_output_scale=3.,
        min_steps=2,max_steps=2,rms_norm_eps=1e-6,residual_connections=False,wrap_latent='none',adaptive_depth=False)
    trace=reasoner.generate(inputs,embedding=embedding,backbone=backbone,action_values=nn.Linear(8,4),backend='uncached',max_active_requests=1)
    recorded=[]
    hook=backbone.register_forward_pre_hook(lambda _,args:recorded.append(args[0].detach().clone()))
    output=reasoner(trace,embedding=embedding,backbone=backbone)
    hook.remove()
    for i in range(2):
        expected=embedding.embeddings['reasoning'].embed_tokens.weight[i].expand(2,-1)
        torch.testing.assert_close(recorded[0][trace.update_positions[i]],expected)
    output.sum().backward()
    assert (embedding.embeddings['reasoning'].embed_tokens.weight.grad.abs().sum(-1)>0).all()
    delayed=deepcopy(embedding)
    with torch.no_grad():delayed.embeddings['reasoning'].embed_tokens.weight.add_(.5)
    other=reasoner(trace,embedding=delayed,backbone=backbone)
    assert not torch.allclose(output,other)
    assert not hasattr(reasoner,'start_embedding')


@torch.no_grad()
def test_cached_text_reasoning_matches_full_sequence_and_resets():
    torch.manual_seed(9)
    tok = tokenizer('all')
    rows = events()
    full, _ = pack_token_batch(steps=tok.encode(rows))
    embedding = MultiEmbedding(embeddings={'text': Embedding(vocab_size=10, hidden_dim=8),
                                          'reasoning': Embedding(vocab_size=2, hidden_dim=8)})
    backbone = Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2,
                        head_dim=4, max_position_embeddings=128, dtype=torch.float32)
    head = nn.Linear(8, 10)
    head.weight.zero_()
    head.bias.zero_()
    head.bias[0] = 10
    reasoner = AutoregressiveReasoner(head)
    kwargs = dict(embedding=embedding, backbone=backbone, tokenizer_name='text',
                  max_new_tokens=3, end_token_id=0, backend='continuous',
                  max_active_requests=1, do_sample=False, temperature=1.)
    expected = reasoner.generate(full, **kwargs)
    for _ in range(2):
        cache = backbone.decode_session(batch_size=1)
        try:
            for i, row in enumerate(rows):
                chunk, _ = pack_token_batch(steps=tok.encode([row]))
                trace = reasoner.generate(chunk, cache=cache, **kwargs)
                assert len(trace) == 1
                assert trace[0].generated_ids == expected[i].generated_ids
                torch.testing.assert_close(trace[0].final_hidden, expected[i].final_hidden)
        finally:
            cache.close()
