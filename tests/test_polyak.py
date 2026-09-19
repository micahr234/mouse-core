from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from mouse_core.models import LatentReasoner, Model, ModelOutput
from mouse_core.models.backbone import IdentityBackbone, TransformerBackbone
from mouse_core.models.heads import (
    BaseHead,
    ClassificationHead,
    RegressionHead,
    LayerwiseRegressionHead,
)
from mouse_core.polyak import Polyak, _PolyakState
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

_TOK = token_tokenizer("action", "episode_done")
_BATCH = [
    [
        {"action": 0, "reward": 0.0, "episode_done": 0, "task_done": 0},
        {"action": 1, "reward": 1.0, "episode_done": 0, "task_done": 0},
        {"action": 2, "reward": 2.0, "episode_done": 1, "task_done": 0},
    ]
]


def _head(hidden_dim: int) -> RegressionHead:
    return RegressionHead(
        in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, use_norm=True
    )


def _tiny_model() -> Model:
    hidden_dim = 8
    backbone = IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32)
    head = _head(hidden_dim)
    return Model(backbone=backbone, heads=head, action_source=head, reasoner=None)


def _llama_model(*, layerwise: bool = False) -> Model:
    hidden_dim = 16
    backbone = TransformerBackbone(architecture="llama", 
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True,
        hidden_dim=hidden_dim, num_layers=2, num_heads=2, max_position_embeddings=64, vocab_size=32)
    head: BaseHead
    if layerwise:
        head = LayerwiseRegressionHead(
            num_backbone_layers=2,
            in_features=hidden_dim,
            out_features=4,
            hidden_dim=hidden_dim,
            num_layers=1, use_norm=True,
        )
    else:
        head = _head(hidden_dim)
    return Model(backbone=backbone, heads=head, action_source=head, reasoner=None)


def _token_batch(model: Model):
    return batch_to_token_batch(_TOK, _BATCH)


def _perturb(module: nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.add_(1.0)


def _q_close(a: ModelOutput, b: ModelOutput, key: str = "action_value") -> bool:
    return torch.allclose(a.predictions[key], b.predictions[key], atol=1e-5)


def _count_calls(module: nn.Module, name: str = "forward"):
    orig = getattr(module, name)
    calls = {"n": 0}

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    setattr(module, name, _wrapped)
    return calls


# ---- copy ---------------------------------------------------------


def test_copy_is_a_frozen_full_copy() -> None:
    model = _llama_model()
    delayed = model.copy(heads=(model._heads["action_value"],))
    assert delayed.backbone is not model.backbone
    assert delayed.reasoner is None
    assert delayed.action_source == model.action_source
    assert delayed.training
    assert all(not p.requires_grad for p in delayed.parameters())
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        assert p is not online[name]  # every trainable parameter is copied
        assert torch.equal(p, online[name])
    # Copying did not freeze the online model.
    assert all(p.requires_grad for p in model.parameters())


def test_copy_shares_frozen_parameters_by_reference() -> None:
    model = _llama_model()
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    delayed = model.copy(heads=(model._heads["action_value"],))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith("heads."):
            assert p is not online[name]
        else:
            assert p is online[name]  # frozen backbone: referenced


def test_copy_rejects_a_model_with_nothing_trainable() -> None:
    model = _tiny_model()
    delayed = model.copy(heads=(model._heads["action_value"],))
    with pytest.raises(ValueError, match="trainable online model"):
        delayed.copy(heads=(delayed._heads["action_value"],))
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable online model"):
        model.copy(heads=(model._heads["action_value"],))


def _two_head_model(hidden_dim: int = 8) -> Model:
    q_head = _head(hidden_dim)
    return Model(
        backbone=IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32),
        heads={
            "action_value": q_head,
            "behavior": ClassificationHead(
                in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, use_norm=True
            ),
        },
        action_source=q_head,
        reasoner=None,
    )


def test_copy_carries_only_the_named_heads() -> None:
    model = _two_head_model()
    delayed = model.copy(heads=(model._heads["action_value"],))
    assert tuple(delayed.heads) == ("action_value",)
    assert delayed.action_source == "action_value"
    assert not any(name.startswith("heads.behavior") for name, _ in delayed.named_parameters())
    # Leaving out the action source is allowed; the copy's action_source is the first name listed.
    behavior_only = model.copy(heads=(model._heads["behavior"],))
    assert tuple(behavior_only.heads) == ("behavior",)
    assert behavior_only.action_source == "behavior"


def test_copy_validates_heads() -> None:
    model = _two_head_model()
    with pytest.raises(TypeError, match="copy heads"):
        model.copy(heads="action_value")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model.copy()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="at least one head"):
        model.copy(heads=())
    with pytest.raises(ValueError, match="duplicate"):
        model.copy(heads=(model._heads["action_value"], model._heads["action_value"]))
    other = _head(8)
    with pytest.raises(ValueError, match="not one of the heads"):
        model.copy(heads=(other,))


def test_polyak_skips_online_heads_the_delayed_model_does_not_carry() -> None:
    model = _two_head_model()
    delayed = model.copy(heads=(model._heads["action_value"],))
    polyak = Polyak(online=model, delayed=delayed)
    _perturb(model.heads)
    polyak.update(tau_heads=1.0, tau_backbone=0.0)
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith("heads."):
            assert torch.equal(p, online[name])
    # The online behavior head trained on; nothing delayed mirrors it.
    assert not any(name.startswith("heads.behavior") for name, _ in delayed.named_parameters())


def test_polyak_rejects_delayed_heads_missing_online() -> None:
    model = _two_head_model()
    other = _tiny_model()
    with pytest.raises(ValueError, match=r"delayed heads \['behavior'\] do not exist"):
        Polyak(online=other, delayed=model.copy(heads=(model._heads["action_value"], model._heads["behavior"],)))


def test_copy_carries_reasoner() -> None:
    hidden_dim = 8
    reasoning = Model(
        backbone=IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32),
        heads=(head := _head(hidden_dim)),
        action_source=head,
        reasoner=LatentReasoner(hidden_dim=hidden_dim, num_thoughts=1),
    )
    dr = reasoning.copy(heads=(reasoning._heads["action_value"],))
    assert dr.reasoner is not None and dr.reasoner is not reasoning.reasoner


# ---- delayed forward --------------------------------------------------------


def test_delayed_model_matches_online_before_update_and_builds_no_graph() -> None:
    torch.manual_seed(0)
    model = _llama_model().eval()
    delayed = model.copy(heads=(model._heads["action_value"],)).eval()
    batch = _token_batch(model)
    out = model(batch)
    saved = {"n": 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved["n"] += 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.no_grad():
            delayed_out = delayed(batch)
    assert saved["n"] == 0
    assert delayed_out.predictions["action_value"].grad_fn is None
    assert _q_close(out, delayed_out)


def test_delayed_model_reruns_its_own_trunk() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.copy(heads=(model._heads["action_value"],)).eval()
    batch = _token_batch(model)
    emb_calls = _count_calls(delayed.backbone, name="embed")
    bb_calls = _count_calls(delayed.backbone)
    with torch.no_grad():
        delayed(batch)
    assert emb_calls["n"] == 1 and bb_calls["n"] == 1


def test_delayed_model_ignores_online_changes_until_update() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.copy(heads=(model._heads["action_value"],)).eval()
    polyak = Polyak(online=model, delayed=delayed)
    batch = _token_batch(model)
    with torch.no_grad():
        before = delayed(batch)
    _perturb(model.backbone)
    _perturb(model.heads)
    with torch.no_grad():
        online = model(batch)
        still_delayed = delayed(batch)
    assert _q_close(before, still_delayed)
    assert not _q_close(online, still_delayed)
    polyak.update(tau_heads=1.0, tau_backbone=1.0)
    with torch.no_grad():
        copied = delayed(batch)
    assert _q_close(online, copied)


def test_layerwise_delayed_model_matches_online_before_update() -> None:
    torch.manual_seed(0)
    model = _llama_model(layerwise=True).eval()
    delayed = model.copy(heads=(model._heads["action_value_layerwise"],)).eval()
    batch = _token_batch(model)
    with torch.no_grad():
        out = model(batch)
        delayed_out = delayed(batch)
    assert out.hidden_states is not None and len(out.hidden_states) == 2
    assert _q_close(out, delayed_out, key="action_value_layerwise")


# ---- Polyak ---------------------------------------------------------------


def test_all_zero_tau_does_not_write_delayed_params() -> None:
    model = _tiny_model()
    delayed = model.copy(heads=(model._heads["action_value"],))
    polyak = Polyak(online=model, delayed=delayed)
    versions = [param._version for param in delayed.parameters()]
    polyak.update(tau_heads=0.0, tau_backbone=0.0)
    assert [param._version for param in delayed.parameters()] == versions


def test_polyak_requires_a_tau_per_section() -> None:
    model = _tiny_model()
    polyak = Polyak(online=model, delayed=model.copy(heads=(model._heads["action_value"],)))
    with pytest.raises(TypeError):
        polyak.update(tau_heads=0.1)  # type: ignore[call-arg]
    polyak.update(tau_heads=0.1, tau_backbone=0.1)


def test_polyak_tau_is_convex_combination_and_can_change() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    delayed = model.copy(heads=(model._heads["action_value"],))
    polyak = Polyak(online=model, delayed=delayed)
    online = next(model.heads.parameters())
    delayed_p = next(delayed.heads.parameters())
    online.data.fill_(1.0)
    delayed_p.data.fill_(0.0)
    polyak.update(tau_heads=0.5, tau_backbone=0.0)
    assert torch.allclose(delayed_p, torch.full_like(delayed_p, 0.5))
    polyak.update(tau_heads=1.0, tau_backbone=0.0)
    assert torch.allclose(delayed_p, torch.ones_like(delayed_p))


def test_each_tau_interpolates_only_its_section() -> None:
    torch.manual_seed(0)
    model = _llama_model()
    delayed = model.copy(heads=(model._heads["action_value"],))
    polyak = Polyak(online=model, delayed=delayed)
    snapshot = {n: p.detach().clone() for n, p in delayed.named_parameters()}
    _perturb(model)
    polyak.update(tau_heads=0.0, tau_backbone=0.25)
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith("heads."):
            assert torch.equal(p, snapshot[name])
        else:
            assert name.startswith("backbone.")
            assert torch.allclose(p, 0.75 * snapshot[name] + 0.25 * online[name])


def test_tau_backbone_also_moves_reasoner() -> None:
    hidden_dim = 8
    reasoning = Model(
        backbone=IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32),
        heads=(head := _head(hidden_dim)),
        action_source=head,
        reasoner=LatentReasoner(hidden_dim=hidden_dim, num_thoughts=1),
    )
    dr = reasoning.copy(heads=(reasoning._heads["action_value"],))
    assert reasoning.reasoner is not None and dr.reasoner is not None
    _perturb(reasoning.reasoner)
    Polyak(online=reasoning, delayed=dr).update(tau_heads=0.0, tau_backbone=1.0)
    for a, b in zip(dr.reasoner.parameters(), reasoning.reasoner.parameters(), strict=True):
        assert torch.equal(a, b)


def test_polyak_rejects_tau_out_of_range() -> None:
    model = _tiny_model()
    polyak = Polyak(online=model, delayed=model.copy(heads=(model._heads["action_value"],)))
    with pytest.raises(ValueError, match=r"tau_heads must be in \[0, 1\]"):
        polyak.update(tau_heads=1.5, tau_backbone=0.1)
    with pytest.raises(ValueError, match=r"tau_backbone must be in \[0, 1\]"):
        polyak.update(tau_heads=0.1, tau_backbone=2.0)


def test_polyak_small_tau_accumulates_in_fp32() -> None:
    online = nn.Linear(8, 8, bias=False)
    delayed = nn.Linear(8, 8, bias=False)
    online.weight.data.fill_(1.0)
    delayed.weight.data.fill_(0.9)
    state = _PolyakState(online, delayed, section="heads")
    tau = 0.0005
    steps = 2000
    for _ in range(steps):
        state.update(tau)
    expected = 1.0 - 0.1 * (1.0 - tau) ** steps
    assert torch.allclose(delayed.weight, torch.full_like(delayed.weight, expected), atol=1e-4)


def test_polyak_rejects_non_fp32_interpolated_params() -> None:
    """A trainable bf16 copy would round a tiny tau away; Polyak refuses it."""
    online = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    delayed = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="fp32 parameters only"):
        _PolyakState(online, delayed, section="backbone")


def test_polyak_skips_shared_frozen_params_and_rejects_shared_trainable() -> None:
    online = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
    online[0].requires_grad_(False)
    delayed = nn.Sequential(online[0], nn.Linear(8, 8, bias=False))
    state = _PolyakState(online, delayed, section="backbone")
    assert len(state) == 1  # the shared frozen layer is not interpolated
    online_trainable = cast(nn.Linear, online[1])
    delayed_trainable = cast(nn.Linear, delayed[1])
    online_trainable.weight.data.fill_(1.0)
    delayed_trainable.weight.data.fill_(0.0)
    state.update(0.5)
    assert torch.allclose(delayed_trainable.weight, torch.full((8, 8), 0.5))

    shared_trainable = nn.Sequential(online[0], online[1])  # trainable layer shared too
    with pytest.raises(ValueError, match="same tensor online and delayed"):
        _PolyakState(online, shared_trainable, section="backbone")
    with pytest.raises(ValueError, match="copy"):
        _PolyakState(online, online, section="backbone")


def test_polyak_rejects_wrong_models() -> None:
    model = _tiny_model()
    delayed = model.copy(heads=(model._heads["action_value"],))
    with pytest.raises(TypeError):
        Polyak(online=model, delayed=nn.Linear(2, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="copy"):
        Polyak(online=model, delayed=model)
    with pytest.raises(ValueError, match="trainable model"):
        Polyak(online=delayed, delayed=model.copy(heads=(model._heads["action_value"],)))
    other = _tiny_model()
    with pytest.raises(ValueError, match="copy"):
        Polyak(online=model, delayed=other)  # trainable sections that are not copies
    mismatched = Model(
        backbone=IdentityBackbone(hidden_dim=8, vocab_size=32),
        heads=(head := RegressionHead(in_features=8, out_features=4, hidden_dim=8, num_layers=2, use_norm=True)),
        action_source=head,
        reasoner=None,
    ).requires_grad_(False)
    with pytest.raises(ValueError, match="parameter names"):
        Polyak(online=model, delayed=mismatched)


# ---- forward contract -------------------------------------------------------


def test_forward_returns_model_output() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch = _token_batch(model)
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, ModelOutput)
    assert out.last_hidden_state.shape == (batch.L, model.hidden_dim)
    assert out.head_output_indices.shape == (batch.P,)
    assert out.cache is None


def test_last_hidden_state_stays_on_the_tape() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    out = model(_token_batch(model))
    assert out.last_hidden_state.requires_grad
    assert out.predictions["action_value"].requires_grad


def _assert_no_autograd_graph(fn: Callable[[], Any]) -> Any:
    saved = {"n": 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved["n"] += 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = fn()
    assert saved["n"] == 0
    return result


def test_delayed_forward_under_no_grad_builds_no_graph_in_train_mode() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    delayed = model.copy(heads=(model._heads["action_value"],))
    batch = _token_batch(model)

    def run():
        with torch.no_grad():
            return delayed(batch)

    delayed_out = _assert_no_autograd_graph(run)
    assert delayed_out.predictions["action_value"].grad_fn is None


def test_forward_rejects_non_token_batch() -> None:
    model = _tiny_model().eval()
    with pytest.raises(TypeError, match="TokenBatch"):
        model(torch.zeros(3, 8))  # type: ignore[arg-type]
