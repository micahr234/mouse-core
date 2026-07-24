"""Incremental cached decoding must match the full-sequence forward pass.

``Model.forward`` accepts ragged batches when decoding: on every call each row
may contribute any number of new steps (including zero). Decoding runs through
a :class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession` (FlexAttention
block-sparse attention with per-sequence KV slots and RoPE positions), carried
inside the returned ``cache``. These tests verify that feeding sequences in
chunks — alone, or batched with other sequences of different lengths — yields
the same per-step predictions as one full unbatched pass.
"""
from __future__ import annotations
from typing import Any, cast
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict
from mouse_core.models import Model
from mouse_core.models.backbone import LlamaBackbone, Qwen3Backbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import DiscreteActionValueHead

def _as_rect(preds: torch.Tensor) -> torch.Tensor:
    """Normalize train-flat ``[N, A]`` vs decode ``[B, S, A]`` for comparisons."""
    if preds.ndim == 2:
        return preds.unsqueeze(0)
    return preds

def _tiny_model(backbone_cls, tokens: int=1) -> Model:
    hidden_dim = 16
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff'}, {'field': 'done', 'type': 'discrete', 'vocab_size': 5}])
    backbone = backbone_cls(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    head = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    return Model(encoder=encoder, backbone=backbone, heads=head).eval()

def _steps(n: int, start: int=0) -> list[dict]:
    return [{'action': i % 4, 'reward': float(i), 'done': int(i % 7 == 6)} for i in range(start, start + n)]

@pytest.mark.parametrize('backbone_cls', [Qwen3Backbone, LlamaBackbone])
def test_chunked_cached_forward_matches_full_forward(backbone_cls) -> None:
    torch.manual_seed(0)
    model = _tiny_model(backbone_cls)
    steps = _steps(6)
    with torch.no_grad():
        full, _, _ = model([steps])
        cache = None
        chunk_preds = []
        for lo, hi in ((0, 3), (3, 4), (4, 6)):
            preds, _, cache = model([steps[lo:hi]], cache=cache, use_cache=True)
            chunk_preds.append(preds['action_value'])
        incremental = torch.cat(chunk_preds, dim=1)
    full_q = _as_rect(full['action_value'])
    assert incremental.shape == full_q.shape
    assert torch.allclose(incremental, full_q, atol=1e-05), 'cached incremental decode diverged from full forward — RoPE cache positions are not being inferred correctly'

def test_step_by_step_cached_rollout_matches_full_forward() -> None:
    """One step at a time, as in the inference notebooks."""
    torch.manual_seed(1)
    model = _tiny_model(Qwen3Backbone)
    steps = _steps(5)
    with torch.no_grad():
        full, _, _ = model([steps])
        cache = None
        last_step_preds = []
        for step in steps:
            preds, _, cache = model([[step]], cache=cache, use_cache=True)
            last_step_preds.append(preds['action_value'][:, -1])
        incremental = torch.stack(last_step_preds, dim=1)
    assert torch.allclose(incremental, _as_rect(full['action_value']), atol=1e-05)

def test_batched_cached_rollout_matches_per_row_rollout() -> None:
    """Batched one-step decode with a shared cache == N separate B=1 decodes.

    This is the batched inference-notebook pattern: all rows step in lockstep,
    so no mask is needed.
    """
    torch.manual_seed(2)
    model = _tiny_model(Qwen3Backbone)
    rows = [_steps(6, start=b * 10) for b in range(3)]
    with torch.no_grad():
        per_row = []
        for row in rows:
            cache = None
            preds_per_step = []
            for step in row:
                preds, _, cache = model([[step]], cache=cache, use_cache=True)
                preds_per_step.append(preds['action_value'][0, -1])
            per_row.append(torch.stack(preds_per_step))
        reference = torch.stack(per_row)
        cache = None
        batched_per_step = []
        for s in range(6):
            preds, _, cache = model([[row[s]] for row in rows], cache=cache, use_cache=True)
            batched_per_step.append(preds['action_value'][:, -1])
        batched = torch.stack(batched_per_step, dim=1)
    assert torch.allclose(batched, reference, atol=1e-05)

@pytest.mark.parametrize('backbone_cls', [Qwen3Backbone, LlamaBackbone])
@pytest.mark.parametrize('tokens', [1, 2])
def test_ragged_batched_chunks_match_unbatched(backbone_cls, tokens) -> None:
    """Batched decode of ragged (variable-size) chunks == unbatched decode.

    Each call, every row contributes a different number of new steps — including
    zero — as envs emitting variable-sized responses would. ``Model.forward``
    decodes through a FlexAttention session; every real step's prediction must
    match the full unbatched forward, including when steps span multiple tokens.
    """
    torch.manual_seed(3)
    model = _tiny_model(backbone_cls, tokens=tokens)
    chunk_lengths = [[2, 4, 3], [3, 0, 2], [1, 3, 1]]
    totals = [sum((call[b] for call in chunk_lengths)) for b in range(3)]
    rows = [_steps(totals[b], start=b * 10) for b in range(3)]
    with torch.no_grad():
        reference = [model([row])[0]['action_value'] for row in rows]
        preds: TensorDict | None = None
        cache = None
        consumed = [0] * len(rows)
        collected: list[list[torch.Tensor]] = [[] for _ in rows]
        for lengths in chunk_lengths:
            batch = []
            for b, n in enumerate(lengths):
                batch.append(rows[b][consumed[b]:consumed[b] + n])
                consumed[b] += n
            step_preds, _, cache = model(batch, cache=cache, use_cache=True)
            preds = step_preds
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(step_preds['action_value'][b, padded_len - n:])
        assert preds is not None
        action = model.get_action(preds, temperature=0.0)
    for b, row_preds in enumerate(collected):
        batched = torch.cat(row_preds, dim=0)
        assert batched.shape == reference[b].shape
        assert torch.allclose(batched, reference[b], atol=1e-05), f'row {b}: ragged batched decode diverged from unbatched decode'
        assert action[b] == reference[b][-1].argmax()

@pytest.mark.parametrize('backbone_cls', [Qwen3Backbone, LlamaBackbone])
def test_empty_first_chunk_then_real_rows_match_unbatched(backbone_cls) -> None:
    """A row that is empty on the very first cached call must still decode exactly.

    The empty row's slots enter the shared cache as fully-masked padding (its
    queries attend to nothing on call one); once real rows arrive they must
    start at position 0 and match a fresh unbatched decode, with no NaN leakage
    from the fully-masked prefill.
    """
    torch.manual_seed(5)
    model = _tiny_model(backbone_cls)
    chunk_lengths = [[3, 0], [1, 2], [2, 3]]
    totals = [sum((call[b] for call in chunk_lengths)) for b in range(2)]
    rows = [_steps(totals[b], start=b * 10) for b in range(2)]
    with torch.no_grad():
        reference = [model([row])[0]['action_value'] for row in rows]
        cache = None
        consumed = [0, 0]
        collected: list[list[torch.Tensor]] = [[], []]
        for lengths in chunk_lengths:
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, _, cache = model(batch, cache=cache, use_cache=True)
            assert torch.isfinite(preds['action_value']).all(), 'NaN/inf leaked from masked padding'
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(preds['action_value'][b, padded_len - n:])
    for b in range(2):
        batched = torch.cat(collected[b], dim=0)
        assert batched.shape == reference[b].shape
        assert torch.allclose(batched, reference[b], atol=1e-05), f'row {b} diverged'

def test_concat_fusion_ragged_chunks_match_unbatched() -> None:
    """Ragged decode with concat fusion, per-modality token counts, and a
    learnable prediction token — tokens_per_step comes from summing modality
    blocks, and the mask must expand to exactly that many tokens per step."""
    torch.manual_seed(6)
    hidden_dim = 16
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff'}, {'field': 'done', 'type': 'discrete', 'vocab_size': 5}, {'type': 'learnable', 'tokens': 1}])
    backbone = Qwen3Backbone(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    head = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=head).eval()
    assert model.encoder.tokens_per_step == 4
    chunk_lengths = [[1, 4, 2], [3, 0, 1], [2, 2, 3]]
    totals = [sum((call[b] for call in chunk_lengths)) for b in range(3)]
    rows = [_steps(totals[b], start=b * 10) for b in range(3)]
    with torch.no_grad():
        reference = [model([row])[0]['action_value'] for row in rows]
        cache = None
        consumed = [0] * 3
        collected: list[list[torch.Tensor]] = [[] for _ in rows]
        for lengths in chunk_lengths:
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, _, cache = model(batch, cache=cache, use_cache=True)
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(preds['action_value'][b, padded_len - n:])
    for b in range(3):
        batched = torch.cat(collected[b], dim=0)
        assert torch.allclose(batched, reference[b], atol=1e-05), f'row {b} diverged'

@pytest.mark.parametrize('backbone_cls', [Qwen3Backbone, LlamaBackbone])
@pytest.mark.parametrize('seed', range(5))
def test_ragged_decode_fuzz(backbone_cls, seed) -> None:
    """Randomized chunk schedules: any split of any batch through the cache
    must reproduce the unbatched full forward at every real step."""
    torch.manual_seed(100 + seed)
    model = _tiny_model(backbone_cls)
    B = int(torch.randint(2, 5, (1,)).item())
    num_calls = int(torch.randint(2, 6, (1,)).item())
    schedule = torch.randint(0, 5, (num_calls, B))
    schedule[0, torch.argmax(schedule.sum(0))] += 1
    for call in range(num_calls):
        if schedule[call].sum() == 0:
            schedule[call, call % B] = 1
    totals = schedule.sum(0)
    for b in range(B):
        if totals[b] == 0:
            schedule[-1, b] = 1
    totals = schedule.sum(0).tolist()
    rows = [_steps(totals[b], start=b * 20) for b in range(B)]
    with torch.no_grad():
        reference = [model([row])[0]['action_value'] for row in rows]
        cache = None
        consumed = [0] * B
        collected: list[list[torch.Tensor]] = [[] for _ in range(B)]
        for call in range(num_calls):
            lengths = [int(schedule[call, b].item()) for b in range(B)]
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, _, cache = model(batch, cache=cache, use_cache=True)
            assert torch.isfinite(preds['action_value']).all()
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(preds['action_value'][b, padded_len - n:])
    for b in range(B):
        batched = torch.cat(collected[b], dim=0)
        assert batched.shape == reference[b].shape
        assert torch.allclose(batched, reference[b], atol=1e-05), f'seed {seed} row {b} schedule {schedule.tolist()}: ragged decode diverged'

def test_cache_without_use_cache_raises() -> None:
    """Passing cache= without use_cache=True must raise — the decode session is
    mutated by every call, so a read-only pass over an existing cache cannot exist."""
    model = _tiny_model(Qwen3Backbone)
    with torch.no_grad():
        _, _, cache = model([_steps(2)], use_cache=True)
        with pytest.raises(ValueError, match='use_cache'):
            model([_steps(1, start=2)], cache=cache)

def test_uniform_then_ragged_cached_decode() -> None:
    """A cache started with uniform rows can continue with ragged chunks."""
    torch.manual_seed(4)
    model = _tiny_model(Qwen3Backbone)
    rows = [_steps(6, start=b * 10) for b in range(2)]
    with torch.no_grad():
        reference = [model([row])[0]['action_value'] for row in rows]
        cache = None
        preds, _, cache = model([row[:2] for row in rows], cache=cache, use_cache=True)
        collected = [[preds['action_value'][b]] for b in range(2)]
        consumed = [2, 2]
        for lengths in ([3, 1], [1, 3]):
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, _, cache = model(batch, cache=cache, use_cache=True)
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(preds['action_value'][b, padded_len - n:])
    for b in range(2):
        batched = torch.cat(collected[b], dim=0)
        assert torch.allclose(batched, reference[b], atol=1e-05)

def test_flex_decode_session_drops_without_cyclic_gc() -> None:
    """mask_mod must not close over the session (that kept KV alive until gc)."""
    import gc
    import weakref
    from mouse_core.models.backbone.flex_decode import FlexDecodeSession
    model = _tiny_model(Qwen3Backbone)
    inner = cast(nn.Module, cast(Any, model.backbone).model)
    session = FlexDecodeSession(inner, batch_size=2, capacity=64)
    wr = weakref.ref(session)
    gc.disable()
    try:
        del session
        assert wr() is None, 'FlexDecodeSession stayed alive without cyclic GC'
    finally:
        gc.enable()
