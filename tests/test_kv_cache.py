from __future__ import annotations

"""Incremental cached decoding must match the full-sequence forward pass.

``Model.forward`` accepts ragged batches when decoding: on every call each row
may contribute any number of new steps (including zero). Decoding runs through
a :class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession` (FlexAttention
block-sparse attention with per-sequence KV slots and RoPE positions), carried
inside the returned ``cache``. These tests verify that feeding sequences in
chunks — alone, or batched with other sequences of different lengths — yields
the same per-step predictions as one full unbatched pass.
"""
from typing import Any, cast
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict
from mouse_core.models import Model
from mouse_core.models.backbone import LlamaBackbone, Qwen3Backbone
from mouse_core.models.backbone.flex_decode import _decode_rope_positions
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.data import NumericTokenizer
from mouse_core.models.heads import DiscreteActionValueHead
from mouse_core.data import Grouper
from tests._token_batch_helpers import batch_to_token_batch, tok_from_encoder


def _loop_decode_rope_positions(
    chunk_grouping_ids: torch.Tensor,
    real: torch.Tensor,
    cached_grouping_ids: torch.Tensor,
    prior_lengths: torch.Tensor,
) -> torch.Tensor:
    """Reference: the previous per-row / per-token Python loop."""
    B, S = chunk_grouping_ids.shape
    rope_pos = torch.zeros(B, S, dtype=torch.long, device=chunk_grouping_ids.device)
    n = real.sum(dim=1)
    for b in range(B):
        nb = int(n[b].item())
        if nb == 0:
            continue
        start = S - nb
        row_mids = chunk_grouping_ids[b, start:S]
        pl = int(prior_lengths[b].item())
        if pl > 0:
            cached = cached_grouping_ids[b, :pl]
            bases = torch.zeros(nb, dtype=torch.long, device=chunk_grouping_ids.device)
            for i in range(nb):
                bases[i] = (cached == row_mids[i]).sum()
        else:
            bases = torch.zeros(nb, dtype=torch.long, device=chunk_grouping_ids.device)
        local = torch.zeros(nb, dtype=torch.long, device=chunk_grouping_ids.device)
        for i in range(nb):
            local[i] = (row_mids[:i] == row_mids[i]).sum()
        rope_pos[b, start:S] = bases + local
    return rope_pos

_tok = tok_from_encoder

def _as_rect(preds: torch.Tensor) -> torch.Tensor:
    """Normalize train-flat ``[N, A]`` vs decode ``[B, S, A]`` for comparisons."""
    if preds.ndim == 2:
        return preds.unsqueeze(0)
    return preds

def _tiny_model(backbone_cls, tokens: int=1) -> Model:
    hidden_dim = 16
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4}, {"type": 'fourier', "field": "reward"}, {"type": 'discrete', "field": "episode_done", "vocab_size": 3}, {"type": 'discrete', "field": "task_done", "vocab_size": 3}])
    backbone = backbone_cls(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    head = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    return Model(encoder=encoder, backbone=backbone, heads=head).eval()

def _steps(n: int, start: int=0) -> list[dict]:
    return [{'action': i % 4, 'reward': float(i), 'episode_done': int(i % 7 == 6), 'task_done': 0} for i in range(start, start + n)]


def _fwd(model: Model, rows: list[list[dict]], **kwargs):
    grouper = Grouper(fields=[{"input_field": "task_index", "output_field": "grouping_id"}])
    # Ensure absolute grouping ids exist on every step.
    patched = [
        [{**step, "task_index": step.get("task_index", 0)} for step in seq]
        for seq in rows
    ]
    tb = batch_to_token_batch(_tok(model.encoder), patched, grouper=grouper)
    return model(tb, **kwargs)

@pytest.mark.parametrize('backbone_cls', [Qwen3Backbone, LlamaBackbone])
def test_chunked_cached_forward_matches_full_forward(backbone_cls) -> None:
    torch.manual_seed(0)
    model = _tiny_model(backbone_cls)
    steps = _steps(6)
    with torch.no_grad():
        full, _, _ = _fwd(model, [steps])
        cache = None
        chunk_preds = []
        for lo, hi in ((0, 3), (3, 4), (4, 6)):
            preds, _, cache = _fwd(model, [steps[lo:hi]], cache=cache, use_cache=True)
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
        full, _, _ = _fwd(model, [steps])
        cache = None
        last_step_preds = []
        for step in steps:
            preds, _, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
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
                preds, _, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
                preds_per_step.append(preds['action_value'][0, -1])
            per_row.append(torch.stack(preds_per_step))
        reference = torch.stack(per_row)
        cache = None
        batched_per_step = []
        for s in range(6):
            preds, _, cache = _fwd(model, [[row[s]] for row in rows], cache=cache, use_cache=True)
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
        reference = [_fwd(model, [row])[0]['action_value'] for row in rows]
        preds: TensorDict | None = None
        cache = None
        consumed = [0] * len(rows)
        collected: list[list[torch.Tensor]] = [[] for _ in rows]
        for lengths in chunk_lengths:
            batch = []
            for b, n in enumerate(lengths):
                batch.append(rows[b][consumed[b]:consumed[b] + n])
                consumed[b] += n
            step_preds, _, cache = _fwd(model, batch, cache=cache, use_cache=True)
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
        reference = [_fwd(model, [row])[0]['action_value'] for row in rows]
        cache = None
        consumed = [0, 0]
        collected: list[list[torch.Tensor]] = [[], []]
        for lengths in chunk_lengths:
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, _, cache = _fwd(model, batch, cache=cache, use_cache=True)
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
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4}, {"type": 'fourier', "field": "reward"}, {"type": 'discrete', "field": "episode_done", "vocab_size": 3}, {"type": 'discrete', "field": "task_done", "vocab_size": 3}, {'type': 'learnable', 'tokens': 1}])
    backbone = Qwen3Backbone(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    head = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=head).eval()
    assert model.encoder.tokens_per_step == 5
    chunk_lengths = [[1, 4, 2], [3, 0, 1], [2, 2, 3]]
    totals = [sum((call[b] for call in chunk_lengths)) for b in range(3)]
    rows = [_steps(totals[b], start=b * 10) for b in range(3)]
    with torch.no_grad():
        reference = [_fwd(model, [row])[0]['action_value'] for row in rows]
        cache = None
        consumed = [0] * 3
        collected: list[list[torch.Tensor]] = [[] for _ in rows]
        for lengths in chunk_lengths:
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, _, cache = _fwd(model, batch, cache=cache, use_cache=True)
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
        reference = [_fwd(model, [row])[0]['action_value'] for row in rows]
        cache = None
        consumed = [0] * B
        collected: list[list[torch.Tensor]] = [[] for _ in range(B)]
        for call in range(num_calls):
            lengths = [int(schedule[call, b].item()) for b in range(B)]
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, _, cache = _fwd(model, batch, cache=cache, use_cache=True)
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
        _, _, cache = _fwd(model, [_steps(2)], use_cache=True)
        with pytest.raises(ValueError, match='use_cache'):
            _fwd(model, [_steps(1, start=2)], cache=cache)

def test_uniform_then_ragged_cached_decode() -> None:
    """A cache started with uniform rows can continue with ragged chunks."""
    torch.manual_seed(4)
    model = _tiny_model(Qwen3Backbone)
    rows = [_steps(6, start=b * 10) for b in range(2)]
    with torch.no_grad():
        reference = [_fwd(model, [row])[0]['action_value'] for row in rows]
        cache = None
        preds, _, cache = _fwd(model, [row[:2] for row in rows], cache=cache, use_cache=True)
        collected = [[preds['action_value'][b]] for b in range(2)]
        consumed = [2, 2]
        for lengths in ([3, 1], [1, 3]):
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, _, cache = _fwd(model, batch, cache=cache, use_cache=True)
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(preds['action_value'][b, padded_len - n:])
    for b in range(2):
        batched = torch.cat(collected[b], dim=0)
        assert torch.allclose(batched, reference[b], atol=1e-05)

@pytest.mark.parametrize('backbone_cls', [Qwen3Backbone, LlamaBackbone])
def test_reset_rows_restarts_one_sequence_without_rebuild(backbone_cls) -> None:
    """Clearing one row's cache length must restart that row; others keep decoding.

    Mimics a single env hitting a task boundary: reset that stream and continue
    the batch without dropping the shared session.
    """
    torch.manual_seed(7)
    model = _tiny_model(backbone_cls)
    row0_a = _steps(4, start=0)
    row0_b = _steps(3, start=100)
    row1 = _steps(7, start=20)
    with torch.no_grad():
        ref0 = _fwd(model, [row0_b])[0]['action_value']
        ref1 = _fwd(model, [row1])[0]['action_value']
        cache = None
        preds, _, cache = _fwd(model, [row0_a, row1[:4]], use_cache=True)
        assert cache is not None
        cache['session'].reset_rows([0])
        collected0: list[torch.Tensor] = []
        collected1 = [preds['action_value'][1]]
        # Continue: row0 starts fresh from row0_b; row1 appends the rest.
        for lengths, batch in (
            ([2, 2], [row0_b[:2], row1[4:6]]),
            ([1, 1], [row0_b[2:], row1[6:]]),
        ):
            preds, _, cache = _fwd(model, batch, cache=cache, use_cache=True)
            padded = max(lengths)
            collected0.append(preds['action_value'][0, padded - lengths[0] :])
            collected1.append(preds['action_value'][1, padded - lengths[1] :])
        got0 = torch.cat(collected0, dim=0)
        got1 = torch.cat(collected1, dim=0)
    assert torch.allclose(got0, ref0, atol=1e-05)
    assert torch.allclose(got1, ref1, atol=1e-05)

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

@pytest.mark.parametrize('backbone_cls', [Qwen3Backbone, LlamaBackbone])
def test_decode_task_mask_isolates_without_reset(backbone_cls) -> None:
    """Continuing past a task boundary without reset_rows matches a fresh task forward.

    Older grouping-id-run KV slots remain in the shared session; grouping-id masking + per-run
    RoPE make the new-task predictions match an unbatched forward of only that
    task (no full-batch rebuild).
    """
    torch.manual_seed(8)
    model = _tiny_model(backbone_cls)
    task0 = [
        {'action': 0, 'reward': 0.0, 'episode_done': 0, 'task_done': 0, 'task_index': 0},
        {'action': 1, 'reward': 1.0, 'episode_done': 0, 'task_done': 0, 'task_index': 0},
        {'action': 2, 'reward': 2.0, 'episode_done': 1, 'task_done': 2, 'task_index': 0},
    ]
    task1 = [
        {'action': 3, 'reward': 3.0, 'episode_done': 0, 'task_done': 0, 'task_index': 1},
        {'action': 1, 'reward': 4.0, 'episode_done': 0, 'task_done': 0, 'task_index': 1},
        {'action': 0, 'reward': 5.0, 'episode_done': 0, 'task_done': 0, 'task_index': 1},
    ]
    with torch.no_grad():
        ref = _fwd(model, [task1])[0]['action_value']
        cache = None
        _, _, cache = _fwd(model, [task0], use_cache=True)
        assert cache is not None
        # Deliberately do NOT call reset_rows — isolation comes from the mask.
        collected: list[torch.Tensor] = []
        for step in task1:
            preds, _, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
            collected.append(preds['action_value'][:, -1])
        got = torch.stack(collected, dim=1)
    assert torch.allclose(got, _as_rect(ref), atol=1e-05)


def test_decode_rope_positions_matches_loop_and_ignores_pads() -> None:
    """Vectorized RoPE positions must match the old loop, including empty rows
    and pad columns whose grouping id collides with a real token (must not count)."""
    cached = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0],
            [2, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    prior = torch.tensor([4, 3, 0], dtype=torch.long)
    # Left-padded chunk: row0 two new gid-1 tokens; row1 one gid-2; row2 empty.
    chunk = torch.tensor(
        [
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )
    real = torch.tensor(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    got = _decode_rope_positions(
        chunk_grouping_ids=chunk,
        real=real,
        cached_grouping_ids=cached,
        prior_lengths=prior,
    )
    ref = _loop_decode_rope_positions(chunk, real, cached, prior)
    assert torch.equal(got, ref)
    # row0: cache has two gid-1 slots; chunk adds 0 then 1 → positions 2, 3.
    # Pad gid 0 must not contribute. row1: three cached gid-2 → position 3.
    assert got.tolist() == [[0, 2, 3], [0, 0, 3], [0, 0, 0]]
