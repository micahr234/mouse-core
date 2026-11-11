"""Shared Mouse-model generation, paged row reuse, and DQN-reward REINFORCE."""
from __future__ import annotations
from _token_batch_fixture import token_batch_fixture

from tests._prediction_layout import fixture_head_steps

import ast
import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn


from mouse_core.models import Backbone, Embedding, MultiEmbedding, MLP, AutoregressiveReasoner, ReasoningTrace, VocabularyHead, load_model
from mouse_core.objectives import DqnObjective, boundary_discount, reinforce_loss


def example_module():
    path = Path(__file__).parents[1] / "examples/23_train_autoregressive_reasoning_dqn.ipynb"
    notebook = json.loads(path.read_text())
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            nodes.extend(node for node in ast.parse("".join(cell["source"])).body
                         if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)))
    # encode_prompt reads notebook settings; these tests exercise the replay and model directly.
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def model():
    torch.manual_seed(7)
    embedding = MultiEmbedding(embeddings={"text": Embedding(vocab_size=16, hidden_dim=16)})
    backbone = Backbone(architecture="qwen3", hidden_dim=16, num_layers=1, num_heads=2,
                        max_position_embeddings=256, dtype=torch.float32)
    token_head = nn.Linear(16, 16)
    with torch.no_grad():
        token_head.weight.mul_(0.01)
        token_head.bias.zero_()
        token_head.bias[0] = 6  # reliable, stochastic EOS for local update tests
    value_head = MLP(in_features=16, out_features=4, hidden_dim=16, num_layers=1, scale=0.1)
    result = example_module().ReasoningDqn(embedding=embedding, backbone=backbone, value_head=value_head,
                                         token_head=token_head, tokenizer_name="text")
    assert result.embedding is embedding and result.backbone is backbone and result.value_head is value_head
    return result.to(torch.get_default_device())


def chain_model():
    result = model()
    with torch.no_grad():
        cast(Any, result.embedding.embeddings["text"]).embed_tokens.weight.copy_(torch.eye(16))
        for layer in cast(Any, result.backbone.model).layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        result.reasoner.token_head.weight.zero_()
        result.reasoner.token_head.bias.zero_()
        for current, successor in ((1, 2), (2, 3), (3, 0), (0, 0)):
            result.reasoner.token_head.weight[successor, current] = 4
    return result


def generate(online, tokens, *, backend, max_new_tokens=8, do_sample=False):
    return online.reasoner.generate(
        tokens, embedding=online.embedding, backbone=online.backbone, tokenizer_name="text",
        end_token_id=0, max_new_tokens=max_new_tokens, backend=backend, max_active_requests=2,
        do_sample=do_sample, temperature=0.8,
    )


@pytest.mark.parametrize("backend", ["uncached", "continuous"])
def test_same_model_generation_variable_lengths_recycling_and_repeated_calls(backend):
    online = chain_model().train()
    tokens = example_module().pack_prefixes([[6, 1], [5, 4, 2], [6, 3], [1]])
    assert len([m for m in online.modules() if isinstance(m, Backbone)]) == 1
    assert not any(isinstance(m, (Backbone, MultiEmbedding)) for m in online.reasoner.modules())
    for _ in range(2):
        traces = generate(online, tokens, backend=backend)
        assert [t.generated_ids for t in traces] == [(2, 3, 0), (3, 0), (0,), (2, 3, 0)]
        assert [t.head_token_id for t in traces] == tokens["head_mask"].nonzero(as_tuple=True)[0].tolist()
        assert online.backbone.training and online.embedding.training and online.reasoner.token_head.training
        assert all(p.grad is None for p in online.parameters())


def test_cache_rows_are_released_on_failed_generation(monkeypatch):
    online = chain_model()
    tokens = example_module().pack_prefixes([[1], [2], [3]])
    sessions = []
    real_create = online.backbone.decode_session

    def create(*, batch_size):
        session = real_create(batch_size=batch_size)
        sessions.append(session)
        return session

    monkeypatch.setattr(online.backbone, "decode_session", create)
    with pytest.raises(RuntimeError, match="did not emit an end token"):
        generate(online, tokens, backend="continuous", max_new_tokens=2)
    assert len(sessions) == 1 and sessions[0]._cache is None and sessions[0]._scheduler is None
    assert online.backbone.training


@pytest.mark.flex_training
def test_final_token_representation_and_token_scores_match_manual_shared_forward():
    online = model()
    tokens = example_module().pack_prefixes([[2, 4], [3]])
    traces = [ReasoningTrace(int(tokens["head_mask"].nonzero(as_tuple=True)[0][0]), (5, 6, 0), 0),
              ReasoningTrace(int(tokens["head_mask"].nonzero(as_tuple=True)[0][1]), (0,), 0)]
    out = online.reasoner(tokens, traces, embedding=online.embedding, backbone=online.backbone,
                          tokenizer_name="text", temperature=0.8, score_tokens=True)
    expected_log_probs = []
    for i, (prefix, trace) in enumerate(zip([[2, 4], [3]], traces, strict=True)):
        ids = torch.tensor(prefix + list(trace.generated_ids))
        embeds = online.embedding(ids, tokenizer_ids=torch.zeros_like(ids), tokenizer_names=("text",))
        hidden = online.backbone(embeds, sequence_ids=torch.zeros_like(ids))
        torch.testing.assert_close(out.final_hidden[i], hidden[-1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out.value_hidden[i], hidden[len(prefix) - 1], atol=1e-6, rtol=1e-5)
        logits = online.reasoner.token_head(hidden[len(prefix) - 1:-1]) / 0.8
        expected_log_probs.append(logits.log_softmax(-1).gather(1, torch.tensor(trace.generated_ids)[:, None]).sum())
    assert out.log_prob_sums is not None
    torch.testing.assert_close(out.log_prob_sums, torch.stack(expected_log_probs), atol=1e-6, rtol=1e-5)
    loss = reinforce_loss(log_prob_sums=out.log_prob_sums, rewards=torch.tensor([1., -1.]),
                          baseline=torch.zeros(2), weights=torch.ones(2))
    loss.backward()
    assert online.reasoner.token_head.weight.grad[0].abs().sum() > 0  # EOS is rewarded too
    assert online.embedding.embeddings["text"].embed_tokens.weight.grad[5].abs().sum() > 0
    assert dict(online.backbone.named_parameters())["model.layers.0.mlp.down_proj.weight"].grad.abs().sum() > 0


@torch.no_grad()
def test_baseline_and_early_readout_cannot_see_samples_future_events_or_other_sequences():
    online = model()
    tokens = token_batch_fixture(token_ids=np.array([2, 3, 4, 5, 6, 7]), numeric_values=np.zeros(6),
                        numeric_mask=np.zeros(6, dtype=bool), sequence_ids=np.array([0, 0, 0, 0, 1, 1]),
                        head_positions=np.array([1, 3, 5]),  num_sequences=2,
                        tokenizer_names=("text",))
    traces = [ReasoningTrace(i, (8, 0), 0) for i in [1, 3, 5]]
    kwargs = dict(embedding=online.embedding, backbone=online.backbone, tokenizer_name="text", temperature=1., score_tokens=False)
    before = online.reasoner(tokens, traces, **kwargs)
    changed = [ReasoningTrace(i, (9, 10, 0), 0) for i in [1, 3, 5]]
    after = online.reasoner(tokens, changed, **kwargs)
    torch.testing.assert_close(before.value_hidden, after.value_hidden)
    assert not torch.allclose(before.final_hidden, after.final_hidden)
    tokens['token_ids'][2:] = torch.tensor([12, 13, 14, 15])
    future = online.reasoner(tokens, traces, **kwargs)
    torch.testing.assert_close(before.final_hidden[0], future.final_hidden[0])


@pytest.mark.parametrize("reward, direction", [(2., -1), (-2., 1)])
def test_reinforce_gradient_sign_and_detached_reward_baseline(reward, direction):
    logits = torch.zeros(2, requires_grad=True)
    outcome = torch.tensor([reward], requires_grad=True)
    baseline = torch.zeros(1, requires_grad=True)
    loss = reinforce_loss(log_prob_sums=logits.log_softmax(0)[:1], rewards=outcome,
                          baseline=baseline, weights=torch.ones(1))
    loss.backward()
    assert logits.grad is not None and logits.grad[0] * direction > 0
    assert outcome.grad is None and baseline.grad is None


def test_reinforce_masks_nonparticipating_decisions_and_normalizes_weights():
    log_probs = torch.tensor([-1., -2., -3.], requires_grad=True)
    loss = reinforce_loss(log_prob_sums=log_probs, rewards=torch.tensor([2., -1., 100.]),
                          baseline=torch.zeros(3), weights=torch.tensor([0.1, 0.1, 0.]))
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.tensor([-1., 0.5, 0.]))


@pytest.mark.parametrize("backend", ["uncached", "continuous"])
@pytest.mark.flex_training
def test_dqn_plus_reinforce_update_shared_backbone_head_and_delayed_target(backend):
    example = example_module()
    online = model()
    target = online.copy().eval()
    head_before = online.reasoner.token_head.weight.detach().clone()
    decoder = dict(online.backbone.named_parameters())["model.layers.0.mlp.down_proj.weight"]
    decoder_before = decoder.detach().clone()
    histories = []

    def encode(history):
        histories.append(history)
        return [4, 1] if not history else [4, 5, 1]

    episodes = example.make_replay(seed=3, num_episodes=2, horizon=2, is_slippery=False)
    tokens, data, groups = example.replay_batch(episodes, encode_prompt=encode, device=torch.get_default_device())
    assert histories[0] == "State 0.\n"
    assert len(histories) == sum(len(ep) - 1 for ep in episodes)
    objective = DqnObjective(reward=None, value=None, gate=None, temperature=0.,
                            discount=boundary_discount(gamma_step=0.9, gamma_terminated=0., gamma_truncated=0.))
    metrics = example.train_step(online, target, torch.optim.AdamW(online.parameters(), lr=1e-3), objective,
        tokens=tokens, objective_data=data, sequence_ids=groups, backend=backend, end_token_id=0,
        max_new_tokens=64, max_active_requests=2, reasoning_temperature=0.8, reinforce_weight=1.,
        target_tau=0.2, max_grad_norm=1.)
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(head_before, online.reasoner.token_head.weight)
    assert not torch.equal(decoder_before, decoder)
    assert all(p.grad is None for p in target.parameters())
    torch.testing.assert_close(target.reasoner.token_head.weight, head_before.lerp(online.reasoner.token_head.weight, 0.2))


def local_checkpoint(path, *, tied=False):
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    vocabulary = {"<eos>": 0, "<unk>": 1, "<pad>": 15}
    vocabulary.update({f"word{i}": i for i in range(2, 15)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>", pad_token="<pad>",
        chat_template='{% for m in messages %}{{ m["content"] }}{% endfor %} assistant:')
    tokenizer.save_pretrained(path)
    lm = Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=1024,
        eos_token_id=0, pad_token_id=15, tie_word_embeddings=tied))
    with torch.no_grad():
        lm.model.embed_tokens.weight.add_(1.)
        if not tied:
            lm.lm_head.weight.zero_()
            lm.lm_head.weight[0].fill_(0.5)
    lm.save_pretrained(path)
    return lm


@pytest.mark.parametrize("tied", [False, True])
def test_restored_vocabulary_head_matches_original_checkpoint_without_second_model(tmp_path, monkeypatch, tied):
    from transformers import AutoModelForCausalLM
    original = local_checkpoint(tmp_path, tied=tied)
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not load another causal LM")
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    embedding = Embedding(pretrained=tmp_path)
    backbone = Backbone(pretrained=tmp_path, dtype=torch.float32)
    head = VocabularyHead.from_pretrained(tmp_path)
    ids = torch.tensor([1, 2, 3])
    with torch.no_grad():
        actual = head(backbone(embedding(ids), sequence_ids=torch.zeros_like(ids)))
        expected = original(input_ids=ids[None]).logits[0]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["uncached", "continuous"])
def test_notebook_model_checkpoint_reload(tmp_path, backend):
    from mouse_core.models import save_model
    example = example_module()
    online = model()
    output = tmp_path / "saved"
    save_model(model=online, path=output)
    restored = model()
    load_model(model=restored, repo_id_or_path=output)
    tokens = example.pack_prefixes([[1, 2]])
    traces = generate(restored, tokens, backend=backend)
    with torch.no_grad():
        q, _, _ = restored(tokens, traces, temperature=0.8, score_tokens=False)
    assert q.shape == (1, 4) and torch.isfinite(q).all()


def test_frozenlake_replay_is_fixed_and_prefixes_do_not_include_future_steps():
    example = example_module()
    episodes = example.make_replay(seed=11, num_episodes=3, horizon=1, is_slippery=False)
    assert episodes == example.make_replay(seed=11, num_episodes=3, horizon=1, is_slippery=False)
    histories = []
    def encode(history):
        histories.append(history)
        return [1, 2]
    tokens, data, groups = example.replay_batch(episodes, encode_prompt=encode, device=torch.device("cpu"))
    assert histories == ["State 0.\n"] * 3
    assert data["sequence_ids"].shape[0] == 6 and tokens["head_mask"].nonzero(as_tuple=True)[0].shape[0] == 3
    assert groups.tolist() == [0, 0, 1, 1, 2, 2]
    assert data["step_ids"].tolist() == [0, 2, 4]
    assert "head_output_count" not in data
    assert (~torch.isnan(data["action"])).tolist() == [False, True] * 3
    assert data["truncated"].tolist() == [False, True] * 3
    assert all((row["action"] is None or 0 <= row["action"] < 4) and 0 <= row["observation"] < 16 for ep in episodes for row in ep)


def test_kernel_opt_in_targets_existing_decoder(monkeypatch):
    backbone = model().backbone
    decoder = backbone.model
    calls = []
    monkeypatch.setattr(decoder, "set_use_kernels", lambda enabled: calls.append(enabled))
    backbone.enable_kernels()
    assert calls == [True] and backbone.model is decoder


@pytest.mark.flex_training
def test_numeric_and_position_features_are_retained_and_trainable():
    online = model()
    online.embedding = MultiEmbedding(embeddings={"text": Embedding(vocab_size=16, hidden_dim=16)},
        numeric_field_names=("reward",), num_frequencies=2, position_num_frequencies=2)
    tokens = token_batch_fixture(token_ids=np.array([2, 3]), numeric_values=np.array([0.25, 0.]),
        numeric_mask=np.array([True, False]), numeric_field_ids=np.array([0, -1]),
        numeric_field_names=("reward",), position_values=np.array([0., 0.5]),
        position_mask=np.array([False, True]), sequence_ids=np.array([0, 0]),
        head_positions=np.array([1]),  num_sequences=1, tokenizer_names=("text",))
    traces = [ReasoningTrace(1, (4, 0), 0)]
    q, _, _ = online(tokens, traces, temperature=1., score_tokens=False)
    q.square().sum().backward()
    parameters = dict(online.embedding.named_parameters())
    for name in ("features.numeric.0.proj.weight", "features.position.proj.weight"):
        gradient = parameters[name].grad
        assert gradient is not None and gradient.abs().sum() > 0
    assert all(p.grad is None for p in online.reasoner.token_head.parameters())


def test_zero_weight_reinforce_rows_ignore_nonfinite_placeholders():
    scores = torch.tensor([-1., float("nan")], requires_grad=True)
    loss = reinforce_loss(log_prob_sums=scores, rewards=torch.tensor([2., float("nan")]),
                          baseline=torch.zeros(2), weights=torch.tensor([1., 0.]))
    loss.backward()
    assert torch.isfinite(loss)
    torch.testing.assert_close(scores.grad, torch.tensor([-2., 0.]))


@pytest.mark.parametrize("backend", ["uncached", "continuous"])
def test_custom_end_token_replaces_model_defaults(backend):
    online = chain_model()
    online.reasoner.token_head.default_end_token_ids = (0, 2)
    tokens = example_module().pack_prefixes([[1]])
    kwargs = dict(embedding=online.embedding, backbone=online.backbone, tokenizer_name="text",
                  max_new_tokens=8, backend=backend, max_active_requests=1, do_sample=False, temperature=1.0)
    default = online.reasoner.generate(tokens, **kwargs)
    override = online.reasoner.generate(tokens, end_token_id=3, **kwargs)
    assert default[0].generated_ids == (2,) and default[0].end_token_id == 2
    assert override[0].generated_ids == (2, 3) and override[0].end_token_id == 3
    with torch.no_grad():
        out = online.reasoner(tokens, override, embedding=online.embedding, backbone=online.backbone,
                              tokenizer_name="text", temperature=1.0, score_tokens=False)
    torch.testing.assert_close(out.final_hidden[0], torch.eye(16)[3])


def test_default_end_tokens_loaded_from_generation_config_with_model_config_fallback(tmp_path):
    local_checkpoint(tmp_path)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 3]}))
    head = VocabularyHead.from_pretrained(tmp_path)
    assert head.default_end_token_ids == (2, 3)
    (tmp_path / "generation_config.json").unlink()
    head = VocabularyHead.from_pretrained(tmp_path)
    assert head.default_end_token_ids == (0,)


@pytest.mark.parametrize("end_token_id", [-1, 16, True, 1.5])
def test_end_token_must_be_an_integer_in_the_vocabulary(end_token_id):
    with pytest.raises(ValueError, match="integer inside"):
        model().reasoner.end_token_ids(end_token_id=end_token_id, vocab_size=16)


def test_missing_model_default_requires_explicit_end_token():
    with pytest.raises(ValueError, match="specify end_token_id"):
        model().reasoner.end_token_ids(end_token_id=None, vocab_size=16)


@pytest.mark.flex_training(default_device=False)
def test_complete_frozenlake_notebook_trains_evaluates_and_saves_locally(tmp_path):
    path = Path(__file__).parents[1] / "examples/23_train_autoregressive_reasoning_dqn.ipynb"
    notebook = json.loads(path.read_text())
    source = tmp_path / "source"
    local_checkpoint(source)
    output = tmp_path / "trained"
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        exec(compile("".join(cell["source"]), f"{path}:cell{index}", "exec"), namespace)
        if index == 1:
            namespace.update(PRETRAINED=str(source), REPLAY_EPISODES=2, EPISODES_PER_UPDATE=2,
                             MAX_EPISODE_STEPS=2, TRAIN_STEPS=1, MAX_NEW_TOKENS=64,
                             MAX_ACTIVE_REQUESTS=2, EVAL_EPISODES=1, OUTPUT=output)
    online = namespace["online"]
    assert sum(isinstance(module, Backbone) for module in online.modules()) == 1
    assert online.reasoner.token_head.default_end_token_ids == (0,)
    assert len(namespace["returns"]) == 1
    assert output.is_dir() and (output / "tokenizer/tokenizer.json").exists()
    assert np.isfinite(namespace["metrics"]["dqn_loss"])
    assert np.isfinite(namespace["metrics"]["reinforce_loss"])
    restored = online.copy()
    load_model(model=restored, repo_id_or_path=output)
    for name, parameter in online.named_parameters():
        torch.testing.assert_close(dict(restored.named_parameters())[name], parameter)
