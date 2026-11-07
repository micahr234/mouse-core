from __future__ import annotations
from tests._model_helpers import ExampleModel as Model

"""Incremental cached decoding must match the full-sequence forward pass.

``Model.forward`` accepts ragged batches when decoding: on every call each row
may contribute any number of new steps (including zero). Decoding runs through
a :class:`~mouse_core.models.backbone.continuous_decode.ContinuousDecodeSession` (FlexAttention
block-sparse attention with per-sequence KV slots and RoPE positions), carried
inside the returned ``cache``. These tests verify that feeding sequences in
chunks — alone, or batched with other sequences of different lengths — yields
the same per-step predictions as one full unbatched pass.
"""
from tests._model_helpers import simple_model
from typing import Any, Literal, cast
import pytest
import torch
import torch.nn as nn

from mouse_core.models.backbone import Backbone
from mouse_core.models.backbone import packed_train as packed_train_mod
from mouse_core.models.backbone.packed_train import install_compiled_decoder
from mouse_core.models.lora import LoRAConfig
from mouse_core.models.mlp import MLP
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer


def _as_rect(preds: torch.Tensor) -> torch.Tensor:
    """Normalize train-flat ``[N, A]`` vs decode ``[B, S, A]`` for comparisons."""
    if preds.ndim == 2:
        return preds.unsqueeze(0)
    return preds

_TOK = token_tokenizer("action", "repeat_episode_done")

def _tiny_model(architecture: Literal['llama', 'qwen3']='qwen3', tokens: int=1, dtype: torch.dtype=torch.float32, **backbone_kwargs) -> Model:
    hidden_dim = 32
    backbone = Backbone(architecture=architecture, dtype=dtype, hidden_dim=hidden_dim, num_layers=2, num_heads=2, **backbone_kwargs)
    head = MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)
    return simple_model(backbone=backbone, heads={'action_value': head}, reasoner=None).eval()

def _steps(n: int, start: int=0) -> list[dict]:
    return [{'action': i % 4, 'reward': float(i), 'repeat_episode_done': int(i % 7 == 6), 'episode_done': 0} for i in range(start, start + n)]


def _fwd(model: Model, rows: list[list[dict]], **kwargs):
    patched = [
        [{**step, "episode_index": step.get("episode_index", 0)} for step in seq]
        for seq in rows
    ]
    tb = batch_to_token_batch(
        _TOK,
        patched,
    )
    out = model(tb, **kwargs)
    return out["predictions"], out["cache"]

@pytest.mark.parametrize('backbone_cls', ["qwen3", "llama"])
def test_chunked_cached_forward_matches_full_forward(backbone_cls: Literal["llama", "qwen3"]) -> None:
    torch.manual_seed(0)
    model = _tiny_model(backbone_cls)
    steps = _steps(6)
    with torch.no_grad():
        full, _ = _fwd(model, [steps])
        cache = None
        chunk_preds = []
        for lo, hi in ((0, 3), (3, 4), (4, 6)):
            preds, cache = _fwd(model, [steps[lo:hi]], cache=cache, use_cache=True)
            chunk_preds.append(preds['action_value'])
        incremental = torch.cat(chunk_preds, dim=1)
    full_q = _as_rect(full['action_value'])
    assert incremental.shape == full_q.shape
    assert torch.allclose(incremental, full_q, atol=1e-05), 'cached incremental decode diverged from full forward — RoPE cache positions are not being inferred correctly'

@pytest.mark.parametrize('backbone_cls', ["qwen3", "llama"])
def test_recurring_grouping_id_matches_between_full_and_cached(backbone_cls: Literal["llama", "qwen3"]) -> None:
    """One sequence, cached step by step, matches a full forward of that sequence."""
    torch.manual_seed(3)
    model = _tiny_model(backbone_cls)
    steps = _steps(6)
    for step, episode in zip(steps, (0, 0, 1, 1, 0, 0)):
        step["episode_index"] = episode
    with torch.no_grad():
        full, _ = _fwd(model, [steps])
        cache = None
        last_step_preds = []
        for step in steps:
            preds, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
            last_step_preds.append(preds['action_value'][:, -1])
        incremental = torch.stack(last_step_preds, dim=1)
    assert torch.allclose(incremental, _as_rect(full['action_value']), atol=1e-05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FlexAttention train / compiled Flex decode are CUDA bf16')
def test_cuda_bf16_packed_train_matches_cached_decode_with_recurring_ids() -> None:
    """Recommended config end to end: FlexAttention packed forward == Flex cached decode."""
    torch.manual_seed(4)
    model = _tiny_model("qwen3", dtype=torch.bfloat16).to(torch.device('cuda'))
    steps = _steps(8)
    for step, episode in zip(steps, (0, 0, 1, 1, 0, 0, 2, 0)):
        step["episode_index"] = episode
    with torch.no_grad():
        full, _ = _fwd(model, [steps])
        cache = None
        last_step_preds = []
        for step in steps:
            preds, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
            last_step_preds.append(preds['action_value'][:, -1])
        incremental = torch.stack(last_step_preds, dim=1)
    full_q = _as_rect(full['action_value'])
    assert full_q.dtype == torch.float32
    assert torch.allclose(incremental, full_q, atol=0.05), (incremental - full_q).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='compiled fp32 Flex decode is CUDA-only')
def test_cuda_fp32_decode_is_compiled_and_matches_full_forward() -> None:
    """fp32 on CUDA compiles the Flex decode kernel (no eager fallback) and stays exact."""
    torch.manual_seed(5)
    model = _tiny_model('qwen3', head_dim=16).to(device=torch.device('cuda'))
    steps = _steps(200)
    for step, episode in zip(steps, [0] * 100 + [1] * 60 + [0] * 40):
        step['episode_index'] = episode
    with torch.no_grad():
        full, _ = _fwd(model, [steps])
        preds_a, cache = _fwd(model, [steps[:150]], use_cache=True)
        preds_b, cache = _fwd(model, [steps[150:]], cache=cache, use_cache=True)
        incremental = torch.cat([preds_a['action_value'], preds_b['action_value']], dim=1)
    session = cache.sessions[0]
    assert hasattr(session._layer, "_torchdynamo_orig_callable")
    from mouse_core.models.backbone import continuous_decode as decode_mod
    assert decode_mod._compiled_layer is not None
    full_q = _as_rect(full['action_value'])
    assert torch.allclose(incremental, full_q, atol=0.0001), (incremental - full_q).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FlexAttention train / compiled Flex decode are CUDA bf16')
def test_cuda_bf16_lora_compiled_train_matches_cached_decode() -> None:
    """LoRA + compiled decoder body: packed forward == Flex cached decode."""
    torch.manual_seed(5)
    hidden_dim = 64
    backbone = Backbone(architecture='qwen3', dtype=torch.bfloat16, hidden_dim=hidden_dim, num_layers=2, num_heads=4, lora=LoRAConfig(rank=4, alpha=8.0))
    head = MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)
    model = simple_model(backbone=backbone, heads={'action_value': head}, reasoner=None).eval().to(torch.device('cuda'))
    steps = _steps(8)
    for step, episode in zip(steps, (0, 0, 1, 1, 0, 0, 2, 0)):
        step['episode_index'] = episode
    was = packed_train_mod._compiled_layer
    try:
        packed_train_mod._compiled_layer = None
        install_compiled_decoder()
        with torch.no_grad():
            full, _ = _fwd(model, [steps])
            cache = None
            last_step_preds = []
            for step in steps:
                preds, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
                last_step_preds.append(preds['action_value'][:, -1])
            incremental = torch.stack(last_step_preds, dim=1)
        full_q = _as_rect(full['action_value'])
        assert full_q.dtype == torch.float32
        assert torch.allclose(incremental, full_q, atol=0.05), (incremental - full_q).abs().max().item()
        assert cache is not None
        from mouse_core.models.backbone import continuous_decode as decode_mod
        assert decode_mod._compiled_layer is not None
    finally:
        packed_train_mod._compiled_layer = was


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FlexAttention train / bf16 KV pool are CUDA-only')
def test_cuda_fp32_autocast_train_matches_cached_decode() -> None:
    """bf16 declared on both paths: FlexAttention packed train == Flex cached decode on a bf16 KV pool."""
    torch.manual_seed(6)
    model = _tiny_model('qwen3', train_autocast_dtype=torch.bfloat16, decode_autocast_dtype=torch.bfloat16).to(torch.device('cuda'))
    steps = _steps(8)
    for step, episode in zip(steps, (0, 0, 1, 1, 0, 0, 2, 0)):
        step['episode_index'] = episode
    with torch.no_grad():
        full, _ = _fwd(model, [steps])
        cache = None
        last_step_preds = []
        for step in steps:
            preds, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
            last_step_preds.append(preds['action_value'][:, -1])
        incremental = torch.stack(last_step_preds, dim=1)
    assert cache is not None
    session = cache.sessions[0]
    assert session.autocast_dtype is torch.bfloat16
    assert session._cache.dtype == torch.bfloat16
    full_q = _as_rect(full['action_value'])
    assert full_q.dtype == torch.float32
    assert torch.allclose(incremental, full_q, atol=0.05), (incremental - full_q).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FlexAttention train is CUDA-only')
def test_cuda_split_autocast_bf16_train_fp32_decode() -> None:
    """train_autocast_dtype=bf16 with decode_autocast_dtype=None: fp32 KV pool, decode matches bf16 train."""
    torch.manual_seed(7)
    model = _tiny_model('qwen3', train_autocast_dtype=torch.bfloat16).to(torch.device('cuda'))
    steps = _steps(6)
    with torch.no_grad():
        full, _ = _fwd(model, [steps])
        preds_a, cache = _fwd(model, [steps[:4]], use_cache=True)
        preds_b, cache = _fwd(model, [steps[4:]], cache=cache, use_cache=True)
        incremental = torch.cat([preds_a['action_value'], preds_b['action_value']], dim=1)
    session = cache.sessions[0]
    assert session.autocast_dtype is None
    assert session._cache.dtype == torch.float32
    full_q = _as_rect(full['action_value'])
    assert torch.allclose(incremental, full_q, atol=0.05), (incremental - full_q).abs().max().item()


def test_decode_rejects_ambient_autocast() -> None:
    """Precision is declared at construction; a decode inside torch.autocast raises."""
    model = _tiny_model("qwen3")
    steps = _steps(2)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(RuntimeError, match="torch.autocast"):
        _fwd(model, [steps], use_cache=True)














def test_packed_rope_positions_count_same_sequence_tokens() -> None:
    from mouse_core.models.backbone.flex_utils import packed_rope_positions

    seq = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    pos = packed_rope_positions(sequence_ids=seq)
    assert pos.tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2]
    assert packed_rope_positions(
        sequence_ids=torch.zeros(0, dtype=torch.long),
    ).shape == (0,)


def test_step_by_step_cached_rollout_matches_full_forward() -> None:
    """One step at a time, as in the inference notebooks."""
    torch.manual_seed(1)
    model = _tiny_model("qwen3")
    steps = _steps(5)
    with torch.no_grad():
        full, _ = _fwd(model, [steps])
        cache = None
        last_step_preds = []
        for step in steps:
            preds, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
            last_step_preds.append(preds['action_value'][:, -1])
        incremental = torch.stack(last_step_preds, dim=1)
    assert torch.allclose(incremental, _as_rect(full['action_value']), atol=1e-05)

def test_batched_cached_rollout_matches_per_row_rollout() -> None:
    """Batched one-step decode with a shared cache == N separate B=1 decodes.

    This is the batched inference-notebook pattern: all rows step in lockstep,
    so no mask is needed.
    """
    torch.manual_seed(2)
    model = _tiny_model("qwen3")
    rows = [_steps(6, start=b * 10) for b in range(3)]
    with torch.no_grad():
        per_row = []
        for row in rows:
            cache = None
            preds_per_step = []
            for step in row:
                preds, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
                preds_per_step.append(preds['action_value'][0, -1])
            per_row.append(torch.stack(preds_per_step))
        reference = torch.stack(per_row)
        cache = None
        batched_per_step = []
        for s in range(6):
            preds, cache = _fwd(model, [[row[s]] for row in rows], cache=cache, use_cache=True)
            batched_per_step.append(preds['action_value'][:, -1])
        batched = torch.stack(batched_per_step, dim=1)
    assert torch.allclose(batched, reference, atol=1e-05)

def test_eval_loop_get_action_model_output_matches_full_forward() -> None:
    """09 ``run_eval``: lockstep incremental decode + ``get_action(output_dict)``.

    Each call packs one new step per stream, carries the KV cache, and reads
    the last valid head-output. The final greedy actions must match a full
    uncached forward's last-step argmax.
    """
    torch.manual_seed(4)
    model = _tiny_model("qwen3")
    rows = [_steps(5, start=b * 10) for b in range(3)]
    with torch.no_grad():
        expected = [
            int(_fwd(model, [row])[0]["action_value"][-1].argmax().item())
            for row in rows
        ]
        cache = None
        actions = None
        for s in range(5):
            patched = [
                [{**row[s], "episode_index": row[s].get("episode_index", 0)}]
                for row in rows
            ]
            tb = batch_to_token_batch(
                _TOK,
                patched,
            )
            out = model(tb, cache=cache, use_cache=True)
            cache = out["cache"]
            actions = model.get_action(out=out, temperature=0.0)
        assert cache is not None
        cache.close()
        assert actions is not None
        assert actions.tolist() == expected

@pytest.mark.parametrize('backbone_cls', ["qwen3", "llama"])
@pytest.mark.parametrize('tokens', [1, 2])
def test_ragged_batched_chunks_match_unbatched(backbone_cls: Literal["llama", "qwen3"], tokens) -> None:
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
        preds: dict[str, torch.Tensor] | None = None
        cache = None
        consumed = [0] * len(rows)
        collected: list[list[torch.Tensor]] = [[] for _ in rows]
        for lengths in chunk_lengths:
            batch = []
            for b, n in enumerate(lengths):
                batch.append(rows[b][consumed[b]:consumed[b] + n])
                consumed[b] += n
            step_preds, cache = _fwd(model, batch, cache=cache, use_cache=True)
            preds = step_preds
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(step_preds['action_value'][b, padded_len - n:])
        assert preds is not None
        action = model.get_action(out={"predictions": preds}, temperature=0.0)
    for b, row_preds in enumerate(collected):
        batched = torch.cat(row_preds, dim=0)
        assert batched.shape == reference[b].shape
        assert torch.allclose(batched, reference[b], atol=1e-05), f'row {b}: ragged batched decode diverged from unbatched decode'
        assert action[b] == reference[b][-1].argmax()

@pytest.mark.parametrize('backbone_cls', ["qwen3", "llama"])
def test_empty_first_chunk_then_real_rows_match_unbatched(backbone_cls: Literal["llama", "qwen3"]) -> None:
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
            preds, cache = _fwd(model, batch, cache=cache, use_cache=True)
            assert torch.isfinite(preds['action_value']).all(), 'NaN/inf leaked from masked padding'
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(preds['action_value'][b, padded_len - n:])
    for b in range(2):
        batched = torch.cat(collected[b], dim=0)
        assert batched.shape == reference[b].shape
        assert torch.allclose(batched, reference[b], atol=1e-05), f'row {b} diverged'

def test_concat_fusion_ragged_chunks_match_unbatched() -> None:
    """Ragged decode: token counts vary by step and the mask must expand
    to exactly that many tokens per step."""
    torch.manual_seed(6)
    model = _tiny_model("qwen3")
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
            preds, cache = _fwd(model, batch, cache=cache, use_cache=True)
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(preds['action_value'][b, padded_len - n:])
    for b in range(3):
        batched = torch.cat(collected[b], dim=0)
        assert torch.allclose(batched, reference[b], atol=1e-05), f'row {b} diverged'

@pytest.mark.parametrize('backbone_cls', ["qwen3", "llama"])
@pytest.mark.parametrize('seed', range(5))
def test_ragged_decode_fuzz(backbone_cls: Literal["llama", "qwen3"], seed) -> None:
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
            preds, cache = _fwd(model, batch, cache=cache, use_cache=True)
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
    model = _tiny_model("qwen3")
    with torch.no_grad():
        _, cache = _fwd(model, [_steps(2)], use_cache=True)
        with pytest.raises(ValueError, match='use_cache'):
            _fwd(model, [_steps(1, start=2)], cache=cache)

def test_uniform_then_ragged_cached_decode() -> None:
    """A cache started with uniform rows can continue with ragged chunks."""
    torch.manual_seed(4)
    model = _tiny_model("qwen3")
    rows = [_steps(6, start=b * 10) for b in range(2)]
    with torch.no_grad():
        reference = [_fwd(model, [row])[0]['action_value'] for row in rows]
        cache = None
        preds, cache = _fwd(model, [row[:2] for row in rows], cache=cache, use_cache=True)
        collected = [[preds['action_value'][b]] for b in range(2)]
        consumed = [2, 2]
        for lengths in ([3, 1], [1, 3]):
            batch = [rows[b][consumed[b]:consumed[b] + n] for b, n in enumerate(lengths)]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, cache = _fwd(model, batch, cache=cache, use_cache=True)
            padded_len = max(lengths)
            for b, n in enumerate(lengths):
                collected[b].append(preds['action_value'][b, padded_len - n:])
    for b in range(2):
        batched = torch.cat(collected[b], dim=0)
        assert torch.allclose(batched, reference[b], atol=1e-05)

@pytest.mark.parametrize('backbone_cls', ["qwen3", "llama"])
def test_reset_rows_restarts_one_sequence_without_rebuild(backbone_cls: Literal["llama", "qwen3"]) -> None:
    """Clearing one row's cache length must restart that row; others keep decoding.

    Mimics a single env hitting a episode boundary: reset that stream and continue
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
        preds, cache = _fwd(model, [row0_a, row1[:4]], use_cache=True)
        assert cache is not None
        cache.reset_rows([0])
        collected0: list[torch.Tensor] = []
        collected1 = [preds['action_value'][1]]
        # Continue: row0 starts fresh from row0_b; row1 appends the rest.
        for lengths, batch in (
            ([2, 2], [row0_b[:2], row1[4:6]]),
            ([1, 1], [row0_b[2:], row1[6:]]),
        ):
            preds, cache = _fwd(model, batch, cache=cache, use_cache=True)
            padded = max(lengths)
            collected0.append(preds['action_value'][0, padded - lengths[0] :])
            collected1.append(preds['action_value'][1, padded - lengths[1] :])
        got0 = torch.cat(collected0, dim=0)
        got1 = torch.cat(collected1, dim=0)
    assert torch.allclose(got0, ref0, atol=1e-05)
    assert torch.allclose(got1, ref1, atol=1e-05)



def test_decode_cache_close_releases_hf_storage() -> None:
    from mouse_core.models import DecodeCache

    session, hidden = _raw_session(2)
    with torch.no_grad():
        _decode(session, [2, 1], hidden)
    cache = DecodeCache(sessions=(session,))
    cache.close()
    assert session._cache is None
    assert session._scheduler is None
    with pytest.raises(RuntimeError, match="close"):
        _decode(session, [1, 0], hidden)


def test_flex_decode_session_drops_without_cyclic_gc() -> None:
    """mask_mod must not close over the session (that kept KV alive until gc)."""
    import gc
    import weakref
    model = _tiny_model('qwen3')
    inner = cast(nn.Module, cast(Any, model.backbone).model)
    session = model.backbone.decode_session(2)
    wr = weakref.ref(session)
    gc.disable()
    try:
        del session
        assert wr() is None, 'ContinuousDecodeSession stayed alive without cyclic GC'
    finally:
        gc.enable()

def test_ragged_batched_decode_across_page_boundary_matches_full_forward() -> None:
    """A row that spans HF pages, batched with a short row, must
    match its unbatched full forward (page remap + RoPE across pages)."""
    torch.manual_seed(5)
    model = _tiny_model('qwen3')
    from transformers.generation import ContinuousBatchingConfig
    half_page_steps = ContinuousBatchingConfig().page_size // 4
    long_count = 2 * half_page_steps + 4
    long_row = _steps(long_count)
    short_row = _steps(4, start=200)
    with torch.no_grad():
        ref_long = _fwd(model, [long_row])[0]['action_value']
        ref_short = _fwd(model, [short_row])[0]['action_value']
        cache = None
        collected = [[], []]
        consumed = [0, 0]
        for lengths in ([half_page_steps, 1], [half_page_steps, 0], [4, 3]):
            batch = [long_row[consumed[0]:consumed[0] + lengths[0]], short_row[consumed[1]:consumed[1] + lengths[1]]]
            consumed = [c + n for c, n in zip(consumed, lengths)]
            preds, cache = _fwd(model, batch, cache=cache, use_cache=True)
            padded = max(lengths)
            for b, n in enumerate(lengths):
                if n:
                    collected[b].append(preds['action_value'][b, padded - n:])
        assert cache is not None
        session = cache.sessions[0]
    assert session.lengths == (2 * long_count, 8)
    assert torch.allclose(torch.cat(collected[0], dim=0), ref_long, atol=1e-05)
    assert torch.allclose(torch.cat(collected[1], dim=0), ref_short, atol=1e-05)


def _raw_session(batch_size: int):
    model = _tiny_model('qwen3')
    inner = cast(nn.Module, cast(Any, model.backbone).model)
    return (model.backbone.decode_session(batch_size), model.backbone.hidden_dim)


def _decode(session, counts: list[int], hidden: int) -> torch.Tensor:
    S = max(max(counts), 1)
    embeds = torch.randn(len(counts), S, hidden)
    gids = torch.zeros(len(counts), S, dtype=torch.long)
    return session.forward(embeds=embeds, lengths=counts)








@pytest.mark.parametrize('backbone_cls', ["qwen3", "llama"])
def test_decode_new_sample_matches_fresh_forward_after_reset(backbone_cls: Literal["llama", "qwen3"]) -> None:
    """A new sample starts after ``reset_rows``. Its predictions match a fresh forward."""
    torch.manual_seed(8)
    model = _tiny_model(backbone_cls)
    episode0 = [
        {'action': 0, 'reward': 0.0, 'repeat_episode_done': 0, 'episode_done': 0, 'episode_index': 0},
        {'action': 1, 'reward': 1.0, 'repeat_episode_done': 0, 'episode_done': 0, 'episode_index': 0},
        {'action': 2, 'reward': 2.0, 'repeat_episode_done': 1, 'episode_done': 2, 'episode_index': 0},
    ]
    episode1 = [
        {'action': 3, 'reward': 3.0, 'repeat_episode_done': 0, 'episode_done': 0, 'episode_index': 1},
        {'action': 1, 'reward': 4.0, 'repeat_episode_done': 0, 'episode_done': 0, 'episode_index': 1},
        {'action': 0, 'reward': 5.0, 'repeat_episode_done': 0, 'episode_done': 0, 'episode_index': 1},
    ]
    with torch.no_grad():
        ref = _fwd(model, [episode1])[0]['action_value']
        cache = None
        _, cache = _fwd(model, [episode0], use_cache=True)
        assert cache is not None
        cache.reset_rows(rows=[0])
        collected: list[torch.Tensor] = []
        for step in episode1:
            preds, cache = _fwd(model, [[step]], cache=cache, use_cache=True)
            collected.append(preds['action_value'][:, -1])
        got = torch.stack(collected, dim=1)
    assert torch.allclose(got, _as_rect(ref), atol=1e-05)
