"""Coconut-style latent reasoning: generation, bookkeeping, gradients, delayed copy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_core.models import (
    LatentReasoner,
    Model,
    load_model,
    sample_reasoning_splits,
    save_model,
)
from mouse_core.models.backbone import TransformerBackbone, LoRAConfig
from mouse_core.models.heads import RegressionHead
from mouse_core.models.reasoner import _plan_insertions
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

_HIDDEN = 32
_ACTIONS = 4
_TOK = token_tokenizer("action", "observation", "episode_done")
_TOKENS_PER_STEP = 3


def _tiny_model(*, num_thoughts: int = 2, with_reasoner: bool = True) -> Model:
    backbone = TransformerBackbone(architecture="llama", 
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True,
        hidden_dim=_HIDDEN,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
        vocab_size=32,
        lora=LoRAConfig(rank=4))
    heads = RegressionHead(
        in_features=_HIDDEN,
        out_features=_ACTIONS,
        hidden_dim=_HIDDEN,
        num_layers=1, use_norm=True,
    )
    reasoner = (
        LatentReasoner(hidden_dim=_HIDDEN, num_thoughts=num_thoughts)
        if with_reasoner
        else None
    )
    return Model(backbone=backbone, heads=heads, action_source="action_value", reasoner=reasoner)


def _rows(n: int, offset: int = 0, groups: list[int] | None = None) -> list[dict]:
    return [
        {
            "action": (i + offset) % _ACTIONS,
            "observation": (3 * i + offset) % 16,
            "reward": float(i),
            "episode_done": 0,
            "grouping_id": groups[i] if groups is not None else 0,
        }
        for i in range(n)
    ]


def _token_batch(model: Model, batch: list[list[dict]]):
    return batch_to_token_batch(_TOK, batch)


_BATCH = [_rows(4), _rows(3, offset=1)]


def test_head_output_is_text_stream() -> None:
    model = _tiny_model()
    batch = _token_batch(model, _BATCH)
    assert batch.modality_names == ("__text__",)
    assert batch.modality_ids[batch.head_output_indices].tolist() == [0] * batch.N


def test_reasoning_none_matches_plain_forward() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        plain = model(batch).predictions
        with_arg = model(batch, reasoning=None).predictions
    assert torch.equal(plain["action_value"], with_arg["action_value"])


def test_all_skip_splits_match_plain_forward() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        plain = model(batch)
        skipped = model(batch, reasoning=[-1, -1])
    assert torch.equal(plain.predictions["action_value"], skipped.predictions["action_value"])
    assert torch.equal(plain.last_hidden_state, skipped.last_hidden_state)


def test_plan_insertions_bookkeeping() -> None:
    model = _tiny_model()
    batch = _token_batch(model, _BATCH)
    # Seq 0: 4 steps × 3 tokens, head-outputs [2, 5, 8, 11]; seq 1: 3 steps
    # × 3 tokens. Burst at seq 0 step 1 (first head-output token 5), R = 2.
    plan = _plan_insertions(batch, np.array([1, -1]), num_thoughts=2)
    assert plan is not None
    assert plan.ext_length == batch.L + 2
    assert plan.anchors.tolist() == [5]
    assert plan.prefix_starts.tolist() == [0]
    assert plan.latent_positions.tolist() == [5, 6]
    expected_tokens = [i if i < 5 else i + 2 for i in range(batch.L)]
    assert plan.token_positions.tolist() == expected_tokens
    assert plan.ext_head_output_indices.tolist() == [2, 7, 10, 13, 16, 19, 22]
    assert plan.ext_sequence_ids[5:7].tolist() == [0, 0]
    assert plan.ext_grouping_ids[5:7].tolist() == [0, 0]
    # Original ids land at their shifted positions.
    assert plan.ext_sequence_ids[plan.token_positions].tolist() == (
        batch.sequence_ids.tolist()
    )


def test_reasoning_extends_stream_and_shifts_predictions() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_thoughts=3).eval()
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        out = model(batch, reasoning=[1, 0])
    assert out.predictions["action_value"].shape == (batch.N, _ACTIONS)
    assert out.last_hidden_state is not None
    assert out.last_hidden_state.shape[0] == batch.L + 2 * 3
    assert out.head_output_indices is not None
    assert out.head_output_indices.shape[0] == batch.N


def test_burst_changes_only_burst_and_later_steps() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        plain = model(batch).predictions
        reasoned = model(batch, reasoning=[1, -1]).predictions
    q_plain = plain["action_value"]
    q_reasoned = reasoned["action_value"]
    # Steps of sequence 1 (no burst) are isolated by the attention mask.
    assert torch.allclose(q_plain[4:], q_reasoned[4:], atol=1e-5)
    # Step 0 of sequence 0 precedes the burst: causal attention keeps it equal.
    assert torch.allclose(q_plain[0], q_reasoned[0], atol=1e-5)
    # The burst step's prompt attends to the latents; later steps see them too.
    assert not torch.allclose(q_plain[1], q_reasoned[1], atol=1e-5)
    assert not torch.allclose(q_plain[2], q_reasoned[2], atol=1e-5)


def test_gradients_flow_through_latent_chain() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    batch = _token_batch(model, _BATCH)
    predictions = model(batch, reasoning=[1, -1]).predictions
    loss = predictions["action_value"][2].sum()  # post-burst step of sequence 0
    loss.backward()
    proj_grad = model.reasoner.proj.weight.grad  # type: ignore[union-attr]
    assert proj_grad is not None and float(proj_grad.abs().sum()) > 0.0
    q_proj = model.backbone.model.layers[0].self_attn.q_proj  # type: ignore[union-attr]
    assert q_proj.base.weight.grad is None  # frozen base
    lora_grad = q_proj.lora_B.weight.grad
    assert lora_grad is not None and float(lora_grad.abs().sum()) > 0.0


def test_pre_burst_loss_gives_no_reasoner_gradient() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    batch = _token_batch(model, _BATCH)
    predictions = model(batch, reasoning=[1, -1]).predictions
    loss = predictions["action_value"][0].sum()  # step before the burst
    loss.backward()
    proj_grad = model.reasoner.proj.weight.grad  # type: ignore[union-attr]
    assert proj_grad is None or float(proj_grad.abs().sum()) == 0.0


def test_sample_reasoning_splits_eligibility() -> None:
    model = _tiny_model()
    # Sequence 0 has a grouping change after step 1: eligible bursts are the
    # steps whose next step shares the grouping, i.e. {0, 2}. Sequence 1 has
    # a single step: no eligible burst.
    batch = _token_batch(
        model, [_rows(4, groups=[0, 0, 1, 1]), _rows(1, offset=1)]
    )
    rng = np.random.default_rng(0)
    seen: set[int] = set()
    for _ in range(50):
        splits = sample_reasoning_splits(batch=batch, generator=rng)
        assert splits.shape == (2,)
        assert splits[1] == -1
        assert int(splits[0]) in (0, 2)
        seen.add(int(splits[0]))
    assert seen == {0, 2}


def test_reasoning_errors() -> None:
    model = _tiny_model().eval()
    no_reasoner = _tiny_model(with_reasoner=False).eval()
    batch = _token_batch(model, _BATCH)
    with pytest.raises(ValueError, match="requires a reasoner"):
        no_reasoner(batch, reasoning=[1, -1])
    with pytest.raises(ValueError, match="use_cache"):
        model(batch, use_cache=True, reasoning=[1, -1])
    with pytest.raises(ValueError, match="out of range"):
        model(batch, reasoning=[4, -1])
    with pytest.raises(ValueError, match="shape"):
        model(batch, reasoning=[1])


def test_delayed_model_parity_with_reasoning() -> None:
    """At construction the delayed model equals the online model, so the
    delayed reasoning forward on the same bursts matches the online one."""
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.copy(heads=True, backbone=True, reasoner=True).eval()
    assert delayed.reasoner is not None and delayed.reasoner is not model.reasoner
    batch = _token_batch(model, _BATCH)
    out = model(batch, reasoning=[1, 0])
    with torch.no_grad():
        delayed_out = delayed(batch, reasoning=[1, 0])
    assert torch.equal(out.head_output_indices, delayed_out.head_output_indices)
    assert torch.allclose(
        out.predictions["action_value"], delayed_out.predictions["action_value"], atol=1e-5
    )
    assert not delayed_out.predictions["action_value"].requires_grad


def test_delayed_reasoning_builds_no_autograd_graph() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    delayed = model.copy(heads=True, backbone=True, reasoner=True)
    batch = _token_batch(model, _BATCH)
    saved = {"n": 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved["n"] += 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.no_grad():
            delayed_out = delayed(batch, reasoning=[1, 0])
    assert saved["n"] == 0
    assert delayed_out.predictions["action_value"].grad_fn is None


def test_reasoning_states_stay_on_the_tape() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    out = model(_token_batch(model, _BATCH), reasoning=[1, 0])
    assert out.predictions["action_value"].requires_grad
    assert out.last_hidden_state.requires_grad


def test_save_load_roundtrip_with_reasoner(tmp_path) -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_thoughts=3).eval()
    save_model(model=model, path=tmp_path)
    loaded = load_model(repo_id_or_path=str(tmp_path), train_kernel="reference", decode_kernel="flex", dtype=torch.float32, map_location="cpu").eval()
    assert loaded.reasoner is not None
    assert loaded.reasoner.num_thoughts == 3
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        original = model(batch, reasoning=[1, 0]).predictions
        reloaded = loaded(batch, reasoning=[1, 0]).predictions
    assert torch.allclose(
        original["action_value"], reloaded["action_value"], atol=1e-6
    )
