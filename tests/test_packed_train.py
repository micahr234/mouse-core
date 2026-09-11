from __future__ import annotations

"""Packed variable-length training forward: metadata, kernel parity, isolation, gradients, compile."""
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest
import torch

from mouse_core.models.backbone import LlamaBackbone, Qwen3Backbone
from mouse_core.models.backbone import packed_train as packed_train_mod
from mouse_core.models.backbone.flex_decode import packed_rope_positions
from mouse_core.models.backbone.packed_train import (
    TrainKernel,
    _pad_packed,
    _packing_plan,
    _unpad_packed,
    install_compiled_decoder,
    packed_forward,
)
from mouse_core.models.base import Model, _flat_sequence_causal_mask, _flat_sequence_position_ids
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads.dqn import DiscreteActionValueHead
from mouse_core.models.lora import LoRAConfig
from tests._token_batch_helpers import batch_to_packed, batch_to_token_batch, tok_from_encoder

_tok = tok_from_encoder
_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_KERNELS: list[TrainKernel] = ["varlen", "flex", "padded"]


@pytest.fixture
def no_compiled_decoder() -> Iterator[None]:
    """Run eager and restore whatever compiled body other tests installed."""
    was = packed_train_mod._compiled_layer
    packed_train_mod._compiled_layer = None
    try:
        yield
    finally:
        packed_train_mod._compiled_layer = was


def _backbone(cls, *, hidden: int = 64, layers: int = 2, heads: int = 4, kv_heads: int = 4, head_dim: int | None = None, lora: LoRAConfig | None = None, kernel: TrainKernel = "varlen", dtype: torch.dtype = torch.float32):
    kwargs: dict[str, Any] = dict(train_kernel=kernel, decode_kernel="flex", dtype=dtype, hidden_dim=hidden, num_layers=layers, num_heads=heads, num_key_value_heads=kv_heads, lora=lora)
    if head_dim is not None:
        kwargs["head_dim"] = head_dim
    return cls(**kwargs)


def _scale_up(backbone: torch.nn.Module, factor: float = 4.0) -> None:
    """Random-init weights make attention tiny; scale q/k so masks change outputs measurably."""
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if name.endswith(("q_proj.weight", "k_proj.weight", "q_proj.base.weight", "k_proj.base.weight")):
                p.mul_(factor)


def _reference_with_layers(
    backbone, embeds: torch.Tensor, seq: torch.Tensor, grp: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Independent dense reference: HF SDPA with the original predicate and original-order RoPE."""
    mask = _flat_sequence_causal_mask(dtype=embeds.dtype, sequence_ids=seq, grouping_ids=grp)
    pos = _flat_sequence_position_ids(sequence_ids=seq, grouping_ids=grp)
    h, layers = backbone(embeds.unsqueeze(0), attention_mask=mask, position_ids=pos, output_hidden_states=True)
    return h.squeeze(0), tuple(x.squeeze(0) for x in layers)


def _reference(backbone, embeds: torch.Tensor, seq: torch.Tensor, grp: torch.Tensor) -> torch.Tensor:
    return _reference_with_layers(backbone, embeds, seq, grp)[0]


def _stream(L: int, *, n_seq: int, n_grp: int, seed: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    seq = torch.as_tensor(np.sort(rng.integers(0, n_seq, size=L)), device=device)
    grp = torch.as_tensor(rng.integers(-1, n_grp - 1, size=L), device=device)  # negative ids allowed, recurring
    return seq, grp


# ---- metadata -----------------------------------------------------------------


def test_packing_plan_matches_handoff_example() -> None:
    seq = torch.zeros(6, dtype=torch.long)
    grp = torch.tensor([0, 0, 1, 1, 0, 0])
    plan = _packing_plan(seq, grp)
    assert plan.order.tolist() == [0, 1, 4, 5, 2, 3]
    assert plan.inverse.tolist() == [0, 1, 4, 5, 2, 3]
    assert plan.cu_seqlens.tolist() == [0, 4, 6]
    assert plan.cu_seqlens.dtype == torch.int32
    assert plan.max_seqlen == 4
    assert plan.position_ids.tolist() == [0, 1, 2, 3, 0, 1]
    assert plan.position_ids[plan.inverse].tolist() == [0, 1, 0, 1, 2, 3]


def test_packing_plan_non_self_inverse_permutation() -> None:
    seq = torch.tensor([0, 1, 0, 1, 0])
    grp = torch.tensor([0, 0, 1, 0, 0])
    plan = _packing_plan(seq, grp)
    # classes: (0,0)->[0,4], (0,1)->[2], (1,0)->[1,3]
    assert plan.order.tolist() == [0, 4, 2, 1, 3]
    assert plan.inverse.tolist() == [0, 3, 2, 4, 1]
    assert plan.order.tolist() != plan.inverse.tolist()
    assert plan.cu_seqlens.tolist() == [0, 2, 3, 5]
    assert plan.max_seqlen == 2


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("L", [1, 3, 7, 50, 129, 300])
def test_packing_plan_invariants(device: str, L: int) -> None:
    seq, grp = _stream(L, n_seq=4, n_grp=4, seed=L, device=device)
    plan = _packing_plan(seq, grp)
    arange = torch.arange(L, device=device)
    assert torch.equal(plan.inverse[plan.order], arange)
    assert torch.equal(plan.order[plan.inverse], arange)
    packed_seq, packed_grp = seq[plan.order], grp[plan.order]
    lengths = (plan.cu_seqlens[1:] - plan.cu_seqlens[:-1]).tolist()
    assert all(n > 0 for n in lengths) and sum(lengths) == L
    assert plan.max_seqlen == max(lengths)
    pairs = {(int(s), int(g)) for s, g in zip(seq.tolist(), grp.tolist())}
    assert len(lengths) == len(pairs)
    for start, end in zip(plan.cu_seqlens[:-1].tolist(), plan.cu_seqlens[1:].tolist()):
        assert packed_seq[start:end].unique().numel() == 1
        assert packed_grp[start:end].unique().numel() == 1
        # original order preserved inside a class
        assert torch.equal(plan.order[start:end], plan.order[start:end].sort().values)
    assert torch.equal(
        plan.position_ids[plan.inverse], packed_rope_positions(sequence_ids=seq, grouping_ids=grp)
    )


def test_packing_plan_ids_shared_across_sequences_stay_separate() -> None:
    seq = torch.tensor([0, 0, 1, 1])
    grp = torch.tensor([5, 5, 5, 5])
    plan = _packing_plan(seq, grp)
    assert plan.cu_seqlens.tolist() == [0, 2, 4]


def test_pad_unpad_roundtrip_and_right_padding() -> None:
    x = torch.arange(10 * 2 * 3, dtype=torch.float32).view(10, 2, 3)
    cu = torch.tensor([0, 3, 10], dtype=torch.int32)
    padded = _pad_packed(x, cu, 7)
    assert padded.shape == (2, 2, 7, 3)
    assert torch.equal(padded[0, :, :3], x[:3].transpose(0, 1))
    assert torch.equal(padded[0, :, 3:], torch.zeros(2, 4, 3))
    assert torch.equal(padded[1, :, :7], x[3:].transpose(0, 1))
    assert torch.equal(_unpad_packed(padded, cu, 10), x)


def test_packing_plan_large_sparse_ids_do_not_collide() -> None:
    seq = torch.tensor([0, 2**40, 0, 2**40])
    grp = torch.tensor([-(2**50), 7, -(2**50), 7])
    plan = _packing_plan(seq, grp)
    assert plan.cu_seqlens.tolist() == [0, 2, 4]
    assert plan.order.tolist() == [0, 2, 1, 3]


# ---- RoPE position rule -------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_packed_rope_positions_match_brute_force_with_recurring_ids(device: str) -> None:
    """Shared rule == count of earlier same-(sequence, grouping) tokens."""
    for L in (1, 50, 200, 300):
        seq, grp = _stream(L, n_seq=4, n_grp=3, seed=L + 1, device=device)
        got = packed_rope_positions(sequence_ids=seq, grouping_ids=grp)
        same = (seq[:, None] == seq[None, :]) & (grp[:, None] == grp[None, :])
        earlier = torch.arange(L, device=device)[None, :] < torch.arange(L, device=device)[:, None]
        ref = (same & earlier).sum(-1)
        assert torch.equal(got, ref), f"mismatch at L={L} device={device}"
        assert torch.equal(_flat_sequence_position_ids(sequence_ids=seq, grouping_ids=grp).squeeze(0), ref)


# ---- forward parity against the dense reference --------------------------------


@pytest.mark.parametrize(
    "cls,kv_heads,head_dim",
    [(Qwen3Backbone, 4, None), (Qwen3Backbone, 2, 24), (LlamaBackbone, 4, None), (LlamaBackbone, 2, None)],
    ids=["qwen3-mha", "qwen3-gqa-hd24", "llama-mha", "llama-gqa"],
)
@pytest.mark.parametrize("L", [1, 3, 7, 61, 300])
@pytest.mark.parametrize("kernel", _KERNELS)
def test_cpu_fp32_forward_matches_dense_reference(no_compiled_decoder: None, cls, kv_heads: int, head_dim: int | None, L: int, kernel: TrainKernel) -> None:
    torch.manual_seed(0)
    bb = _backbone(cls, kv_heads=kv_heads, head_dim=head_dim)
    _scale_up(bb)
    seq, grp = _stream(L, n_seq=3, n_grp=3, seed=L)
    embeds = torch.randn(L, 64)
    with torch.no_grad():
        got, layers = packed_forward(
            model=bb.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp, output_hidden_states=True, train_kernel=kernel
        )
        ref, ref_layers = _reference_with_layers(bb, embeds, seq, grp)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
    # Per-layer states are every layer's output before the final norm (the
    # FlexDecodeSession contract); HF's tuple swaps the last one for the normed output.
    assert len(layers) == len(ref_layers) == 2
    torch.testing.assert_close(layers[0], ref_layers[0], atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        torch.testing.assert_close(bb.model.norm(layers[-1]), got, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(layers[-1], got)


@_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "cls,kv_heads,head_dim",
    [(Qwen3Backbone, 4, None), (Qwen3Backbone, 2, 32), (LlamaBackbone, 2, None)],
    ids=["qwen3-mha", "qwen3-gqa-hd32", "llama-gqa"],
)
@pytest.mark.parametrize("L", [12, 300], ids=["L<block", "L>block"])
@pytest.mark.parametrize("kernel", _KERNELS)
def test_cuda_fused_forward_matches_fp32_reference_within_half_precision_noise(
    no_compiled_decoder: None, dtype: torch.dtype, cls, kv_heads: int, head_dim: int | None, L: int, kernel: TrainKernel
) -> None:
    """Flash varlen / compiled Flex (bf16/fp16) vs fp32 dense reference, bounded by the same-dtype SDPA error."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    bb32 = _backbone(cls, kv_heads=kv_heads, head_dim=head_dim).to(device)
    _scale_up(bb32)
    bb16 = _backbone(cls, kv_heads=kv_heads, head_dim=head_dim, dtype=dtype).to(device)
    bb16.load_state_dict(bb32.state_dict())  # copies cast the fp32 weights into the bf16/fp16 base
    seq, grp = _stream(L, n_seq=3, n_grp=3, seed=L, device="cuda")
    embeds = torch.randn(L, 64, device=device)
    with torch.no_grad():
        got = packed_forward(
            model=bb16.model, embeds=embeds.to(dtype), sequence_ids=seq, grouping_ids=grp, train_kernel=kernel
        ).float()
        ref32 = _reference(bb32, embeds, seq, grp)
        ref16 = _reference(bb16, embeds.to(dtype), seq, grp).float()
    err = (got - ref32).abs()
    floor = (ref16 - ref32).abs().max().item()
    assert err.max().item() <= 2.0 * floor + 2e-3, (
        f"max {err.max().item():.4g} rms {err.pow(2).mean().sqrt().item():.4g} vs {dtype} SDPA floor {floor:.4g}"
    )


@_cuda
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("L", [40, 300])
def test_cuda_fp32_matches_dense_reference(no_compiled_decoder: None, kernel: TrainKernel, L: int) -> None:
    """fp32 on CUDA: varlen is packed-stream SDPA, padded is rectangular SDPA, flex is compiled block-sparse."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    bb = cast(Any, _backbone(Qwen3Backbone, kv_heads=2).to(device))
    _scale_up(bb)
    seq, grp = _stream(L, n_seq=2, n_grp=3, seed=9, device="cuda")
    embeds = torch.randn(L, 64, device=device)
    with torch.no_grad():
        got = packed_forward(model=bb.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp, train_kernel=kernel)
        ref = _reference(bb, embeds, seq, grp)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


# ---- isolation and causality -----------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kernel", _KERNELS)
def test_isolation_and_recurring_group_causality(no_compiled_decoder: None, device: str, kernel: TrainKernel) -> None:
    torch.manual_seed(1)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    bb = cast(Any, _backbone(Qwen3Backbone, kv_heads=2, dtype=dtype).to(device))
    _scale_up(bb)
    # seq 0: groups 0 0 1 1 0 0 ; seq 1: group 0 0 0
    seq = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], device=device)
    grp = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0], device=device)
    embeds = torch.randn(9, 64, device=device, dtype=dtype)

    def run(e: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return packed_forward(model=bb.model, embeds=e, sequence_ids=seq, grouping_ids=grp, train_kernel=kernel).float()

    base = run(embeds)
    changed = lambda a, b: (a - b).abs().amax(dim=1).gt(0)  # noqa: E731

    other_seq = embeds.clone()
    other_seq[7] += 3.0  # seq 1, middle token
    delta = changed(run(other_seq), base)
    assert not delta[:6].any() and not delta[6].item() and delta[7].item() and delta[8].item()

    other_group = embeds.clone()
    other_group[2] += 3.0  # seq 0, group 1 first occurrence
    delta = changed(run(other_group), base)
    assert delta.tolist() == [False, False, True, True, False, False, False, False, False]

    earlier_occurrence = embeds.clone()
    earlier_occurrence[1] += 3.0  # seq 0, group 0 first run -> its later run (4, 5) must see it
    delta = changed(run(earlier_occurrence), base)
    assert delta.tolist() == [False, True, False, False, True, True, False, False, False]

    future = embeds.clone()
    future[5] += 3.0  # last token of seq 0 group 0: nothing earlier moves
    delta = changed(run(future), base)
    assert delta.tolist() == [False] * 5 + [True] + [False] * 3


# ---- gradients -------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["varlen", "padded"])
def test_cpu_fp32_gradients_match_dense_reference(no_compiled_decoder: None, kernel: TrainKernel) -> None:
    torch.manual_seed(2)
    bb = _backbone(LlamaBackbone, kv_heads=2)
    _scale_up(bb)
    L = 23
    seq, grp = _stream(L, n_seq=2, n_grp=3, seed=5)
    embeds = torch.randn(L, 64, requires_grad=True)
    weight = torch.randn(L, 64)

    got = packed_forward(model=bb.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp, train_kernel=kernel)
    (got * weight).sum().backward()
    got_grads = {n: p.grad.clone() for n, p in bb.named_parameters() if p.grad is not None}
    assert embeds.grad is not None
    got_embed_grad = embeds.grad.clone()
    embeds.grad = None
    bb.zero_grad()

    ref = _reference(bb, embeds, seq, grp)
    (ref * weight).sum().backward()
    assert embeds.grad is not None
    torch.testing.assert_close(got_embed_grad, embeds.grad, atol=1e-4, rtol=1e-4)
    ref_grads = {n: p.grad for n, p in bb.named_parameters() if p.grad is not None}
    assert set(ref_grads) == set(got_grads)  # embed_tokens is unused on both paths
    assert any(".layers." in n for n in got_grads)
    for n, ref_grad in ref_grads.items():
        torch.testing.assert_close(got_grads[n], ref_grad, atol=1e-4, rtol=1e-4, msg=lambda m: f"{n}: {m}")


@pytest.mark.parametrize("kernel", ["varlen", "padded"])
def test_gradient_checkpointing_matches_plain_backward(no_compiled_decoder: None, kernel: TrainKernel) -> None:
    torch.manual_seed(3)
    bb = _backbone(Qwen3Backbone, layers=3, kv_heads=2)
    L = 31
    seq, grp = _stream(L, n_seq=2, n_grp=2, seed=8)
    embeds = torch.randn(L, 64, requires_grad=True)

    def grads(checkpoint: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out = packed_forward(model=bb.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp, train_kernel=kernel, checkpoint=checkpoint)
        out.square().sum().backward()
        assert embeds.grad is not None
        result = (out.detach(), [embeds.grad.clone()] + [p.grad.clone() for p in bb.parameters() if p.grad is not None])
        embeds.grad = None
        bb.zero_grad()
        return result

    out_plain, g_plain = grads(False)
    out_ckpt, g_ckpt = grads(True)
    torch.testing.assert_close(out_ckpt, out_plain)
    for a, b in zip(g_ckpt, g_plain):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


@_cuda
@pytest.mark.parametrize("kernel", _KERNELS)
def test_cuda_bf16_lora_gradients_on_frozen_base(no_compiled_decoder: None, kernel: TrainKernel) -> None:
    """fp32 LoRA and input-embedding grads flow through the fused kernel; frozen base gets none."""
    torch.manual_seed(4)
    device = torch.device("cuda")
    bb = cast(Any, _backbone(Qwen3Backbone, kv_heads=2, lora=LoRAConfig(rank=4, alpha=8.0), dtype=torch.bfloat16).to(device))
    for n, p in bb.named_parameters():
        if ".lora_" in n:
            p.data = p.data.float()
            if ".lora_B." in n:
                torch.nn.init.normal_(p, std=0.05)  # zero B would zero every A gradient
    L = 150
    seq, grp = _stream(L, n_seq=3, n_grp=3, seed=6, device="cuda")
    embeds = torch.randn(L, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(L, 64, device=device)

    got = packed_forward(model=bb.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp, train_kernel=kernel)
    (got.float() * weight).sum().backward()
    assert embeds.grad is not None and torch.isfinite(embeds.grad).all()
    got_embed_grad = embeds.grad.float().clone()
    got_lora = {n: p.grad.clone() for n, p in bb.named_parameters() if p.requires_grad and p.grad is not None}
    assert len(got_lora) == sum(p.requires_grad for p in bb.parameters()) > 0
    for n, p in bb.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n
    embeds.grad = None
    bb.zero_grad()

    ref = _reference(bb, embeds, seq, grp)
    (ref.float() * weight).sum().backward()
    assert embeds.grad is not None
    ref_embed_grad = embeds.grad.float()
    scale = ref_embed_grad.abs().max().item()
    assert (got_embed_grad - ref_embed_grad).abs().max().item() <= 0.05 * scale + 1e-3
    for n, p in bb.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert p.grad.dtype == torch.float32
            tol = 0.05 * p.grad.abs().max().item() + 1e-3
            assert (got_lora[n] - p.grad).abs().max().item() <= tol, n


# ---- compiled body --------------------------------------------------------------


def test_install_compiled_decoder_idempotent(no_compiled_decoder: None) -> None:
    assert install_compiled_decoder() is True
    compiled = packed_train_mod._compiled_layer
    assert compiled is not None
    assert install_compiled_decoder() is False
    assert packed_train_mod._compiled_layer is compiled


@pytest.mark.parametrize("cls", [Qwen3Backbone, LlamaBackbone])
@pytest.mark.parametrize("kernel", _KERNELS)
def test_cpu_compiled_body_matches_eager_across_layouts(no_compiled_decoder: None, cls, kernel: TrainKernel) -> None:
    torch.manual_seed(5)
    bb = _backbone(cls, hidden=32, heads=4, kv_heads=2)
    outs = []
    for L, ng in ((6, 1), (9, 3), (17, 4)):
        seq, grp = _stream(L, n_seq=2, n_grp=ng, seed=L)
        embeds = torch.randn(L, 32)
        with torch.no_grad():
            eager = packed_forward(
                model=bb.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp, output_hidden_states=True, train_kernel=kernel
            )
        outs.append((seq, grp, embeds, eager))
    install_compiled_decoder()
    for seq, grp, embeds, (eager_h, eager_layers) in outs:
        with torch.no_grad():
            h, layers = packed_forward(
                model=bb.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp, output_hidden_states=True, train_kernel=kernel
            )
        torch.testing.assert_close(h, eager_h, atol=1e-4, rtol=1e-4)
        for a, b in zip(layers, eager_layers):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@_cuda
@pytest.mark.parametrize("L", [5, 200], ids=["L<block", "L>block"])
@pytest.mark.parametrize("kernel", _KERNELS)
def test_cuda_bf16_lora_compiled_body_matches_eager_and_trains(no_compiled_decoder: None, L: int, kernel: TrainKernel) -> None:
    torch.manual_seed(6)
    device = torch.device("cuda")
    encoder = NumericEmbedder(hidden_dim=64, modalities=[{"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}])
    backbone = _backbone(Qwen3Backbone, kv_heads=2, lora=LoRAConfig(rank=4, alpha=8.0), dtype=torch.bfloat16)
    head = DiscreteActionValueHead(in_features=64, out_features=4, hidden_dim=64, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=head, action_head="action_value", reasoner=None, recurrence=None).to(device)
    bb = cast(Qwen3Backbone, model.backbone)
    for n, p in bb.named_parameters():
        if ".lora_B." in n:
            torch.nn.init.normal_(p, std=0.05)
    embeds = torch.randn(L, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    seq, grp = _stream(L, n_seq=3, n_grp=2, seed=L, device="cuda")

    def run() -> tuple[torch.Tensor, list[torch.Tensor]]:
        out = packed_forward(model=bb.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp, train_kernel=kernel)
        out.float().sum().backward()
        assert embeds.grad is not None
        grads = [embeds.grad.float().clone()] + [cast(torch.Tensor, p.grad).clone() for p in bb.parameters() if p.requires_grad]
        embeds.grad = None
        bb.zero_grad()
        return out.detach().float(), grads

    eager_out, eager_grads = run()
    install_compiled_decoder()
    compiled_out, compiled_grads = run()
    assert (compiled_out - eager_out).abs().max().item() < 0.1
    assert all(torch.isfinite(g).all() for g in compiled_grads)
    for a, b in zip(compiled_grads, eager_grads):
        assert (a - b).abs().max().item() <= 0.05 * b.abs().max().item() + 1e-2
    for n, p in bb.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n


# ---- contract edges --------------------------------------------------------------


def test_empty_stream(no_compiled_decoder: None) -> None:
    bb = _backbone(Qwen3Backbone)
    empty = torch.zeros(0, dtype=torch.long)
    out, layers = packed_forward(model=bb.model, embeds=torch.zeros(0, 64), sequence_ids=empty, grouping_ids=empty, train_kernel="varlen", output_hidden_states=True)
    assert out.shape == (0, 64)
    assert len(layers) == 2 and all(x.shape == (0, 64) for x in layers)


def test_shape_validation(no_compiled_decoder: None) -> None:
    bb = _backbone(Qwen3Backbone)
    ids = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match=r"\[L, D\]"):
        packed_forward(model=bb.model, embeds=torch.zeros(1, 4, 64), sequence_ids=ids, grouping_ids=ids, train_kernel="varlen")
    with pytest.raises(ValueError, match="grouping_ids"):
        packed_forward(model=bb.model, embeds=torch.zeros(4, 64), sequence_ids=ids, grouping_ids=ids[:3], train_kernel="varlen")


def test_sliding_window_config_is_rejected(no_compiled_decoder: None) -> None:
    bb = Qwen3Backbone(train_kernel="varlen", decode_kernel="flex", dtype=torch.float32, hidden_dim=64, num_layers=1, num_heads=4, use_sliding_window=True)
    ids = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="sliding-window"):
        packed_forward(model=bb.model, embeds=torch.zeros(4, 64), sequence_ids=ids, grouping_ids=ids, train_kernel="varlen")


def test_unknown_train_kernel_is_rejected(no_compiled_decoder: None) -> None:
    bb = _backbone(Qwen3Backbone)
    ids = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="train_kernel"):
        packed_forward(model=bb.model, embeds=torch.zeros(4, 64), sequence_ids=ids, grouping_ids=ids, train_kernel=cast(Any, "sdpa"))


def test_flex_kernel_is_forward_only_on_cpu(no_compiled_decoder: None) -> None:
    """FlexAttention has no CPU backward (torch limitation); the CPU kernel comparison is forward-only."""
    bb = _backbone(Qwen3Backbone)
    ids = torch.zeros(4, dtype=torch.long)
    embeds = torch.randn(4, 64, requires_grad=True)
    with pytest.raises(NotImplementedError, match="CPU"):
        packed_forward(model=bb.model, embeds=embeds, sequence_ids=ids, grouping_ids=ids, train_kernel="flex")
    with torch.no_grad():
        out = packed_forward(model=bb.model, embeds=embeds, sequence_ids=ids, grouping_ids=ids, train_kernel="flex")
    assert out.shape == (4, 64)


# ---- Model integration -------------------------------------------------------------


def test_prepare_sequence_id_col_matches_step_counts() -> None:
    encoder = NumericEmbedder(hidden_dim=8, modalities=[{"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}, {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}, {"type": "learnable", "tokens": 1, "std": 0.02, "positions": 1}])
    batch = [[{"action": s % 4, "reward": float(s)} for s in range(5)], [{"action": 1, "reward": 0.0}, {"action": 2, "reward": 1.0}, {"action": 3, "reward": 2.0}]]
    tb, objective_data = batch_to_packed(_tok(encoder), batch)
    assert list(tb.step_counts()) == [5, 3]
    assert objective_data["sequence_id"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert objective_data["grouping_id"].tolist() == [0] * 8
    assert tb.head_output_indices.shape == (8,)
    assert list(tb.sequence_ids[tb.head_output_indices]).count(0) == 5
    assert list(tb.sequence_ids[tb.head_output_indices]).count(1) == 3


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kernel", _KERNELS)
def test_model_forward_isolates_sequences(no_compiled_decoder: None, device: str, kernel: TrainKernel) -> None:
    torch.manual_seed(2)
    backbone = Qwen3Backbone(train_kernel=kernel, decode_kernel="flex", dtype=torch.float32, hidden_dim=64, num_layers=2, num_heads=4, num_key_value_heads=4)
    encoder = NumericEmbedder(hidden_dim=backbone.hidden_dim, modalities=[{"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}, {"type": "learnable", "tokens": 1, "std": 0.02, "positions": 1}])
    head = DiscreteActionValueHead(in_features=backbone.hidden_dim, out_features=4, hidden_dim=backbone.hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=head, action_head="action_value", reasoner=None, recurrence=None).to(device).eval()
    batch = [[{"action": i % 4} for i in range(3)], [{"action": i % 4} for i in range(3)]]
    tb = batch_to_token_batch(_tok(encoder), batch)
    with torch.no_grad():
        preds0 = model(tb).predictions
        batch_corrupt = [[{"action": 3} for _ in range(3)], [{"action": i % 4} for i in range(3)]]
        tb_c = batch_to_token_batch(_tok(encoder), batch_corrupt)
        preds1 = model(tb_c).predictions
    q0 = preds0["action_value"]
    q1 = preds1["action_value"]
    assert torch.allclose(q0[3:], q1[3:], atol=1e-05, rtol=1e-05)
    assert not torch.allclose(q0[:3], q1[:3], atol=1e-05, rtol=1e-05)
    assert tb.N == 6
    assert list(tb.sequence_ids[tb.head_output_indices]) == [0, 0, 0, 1, 1, 1]


def test_model_train_isolates_tasks_within_sequence(no_compiled_decoder: None) -> None:
    """Packed train forward on a two-task window matches a single-task suffix forward."""
    torch.manual_seed(11)
    backbone = Qwen3Backbone(train_kernel="varlen", decode_kernel="flex", dtype=torch.float32, hidden_dim=32, num_layers=2, num_heads=4, num_key_value_heads=4)
    encoder = NumericEmbedder(
        hidden_dim=backbone.hidden_dim,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
            {"type": "discrete", "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1},
            {"type": "learnable", "tokens": 1, "std": 0.02, "positions": 1},
        ],
    )
    head = DiscreteActionValueHead(in_features=backbone.hidden_dim, out_features=4, hidden_dim=backbone.hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=head, action_head="action_value", reasoner=None, recurrence=None).eval()
    task0 = [
        {"action": 0, "episode_done": 0, "task_done": 0, "task_index": 0},
        {"action": 1, "episode_done": 0, "task_done": 0, "task_index": 0},
        {"action": 2, "episode_done": 1, "task_done": 2, "task_index": 0},
    ]
    task1 = [
        {"action": 3, "episode_done": 0, "task_done": 0, "task_index": 1},
        {"action": 1, "episode_done": 0, "task_done": 0, "task_index": 1},
    ]
    with torch.no_grad():
        tb_both, od = batch_to_packed(_tok(encoder, grouping_field="task_index"), [task0 + task1], grouping_field="task_index")
        tb_t1 = batch_to_token_batch(_tok(encoder, grouping_field="task_index"), [task1], grouping_field="task_index")
        preds_both = model(tb_both).predictions
        preds_t1 = model(tb_t1).predictions
    assert od["task_index"].tolist() == [0, 0, 0, 1, 1]
    assert torch.allclose(preds_both["action_value"][3:], preds_t1["action_value"], atol=1e-05, rtol=1e-05)


def test_model_gradient_checkpointing_flag_reaches_backward(no_compiled_decoder: None) -> None:
    torch.manual_seed(12)
    backbone = Qwen3Backbone(train_kernel="varlen", decode_kernel="flex", dtype=torch.float32, hidden_dim=32, num_layers=2, num_heads=4)
    encoder = NumericEmbedder(hidden_dim=32, modalities=[{"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}])
    head = DiscreteActionValueHead(in_features=32, out_features=4, hidden_dim=32, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=head, action_head="action_value", reasoner=None, recurrence=None)
    tb = batch_to_token_batch(_tok(encoder), [[{"action": i % 4} for i in range(5)], [{"action": 1}]])

    def step() -> list[torch.Tensor]:
        model(tb).predictions["action_value"].square().sum().backward()
        grads = [cast(torch.Tensor, p.grad).clone() for p in model.parameters() if p.grad is not None]
        model.zero_grad()
        return grads

    plain = step()
    backbone.gradient_checkpointing = True
    ckpt = step()
    assert len(plain) == len(ckpt) > 0
    for a, b in zip(plain, ckpt):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
