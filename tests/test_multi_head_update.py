"""Tests for the multi-head-update loop in examples/16_train_offline_multi_head_update_dqn.ipynb."""

from __future__ import annotations

import math

import torch

from mouse_core import AdamW
from mouse_core.data import DataLoader, Datastore, Tokenizer
from mouse_core.models import Model, Polyak
from mouse_core.models.backbone import IdentityBackbone, LlamaBackbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import DiscreteActionValueHead
from mouse_core.objectives import DqnObjective

_MAX_ACTIONS = 4
_MAX_OBS = 16
_HIDDEN = 16


def _clone_params(module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def _changed(before: dict[str, torch.Tensor], module) -> bool:
    return any(not torch.equal(before[name], param.detach()) for name, param in module.named_parameters())


def _store(*, steps: int = 32) -> Datastore:
    store = Datastore()
    for i in range(steps):
        store.append(
            {
                "action": i % _MAX_ACTIONS,
                "observation": i % _MAX_OBS,
                "reward": float(i % 5),
                "episode_done": 1 if (i + 1) % 8 == 0 else 0,
                "task_done": 1 if (i + 1) % 16 == 0 else 0,
                "task_index": i // 16,
            }
        )
    return store


def _tokenizer() -> Tokenizer:
    return Tokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action"},
            {"type": "discrete", "input_field": "observation"},
            {"type": "fourier", "input_field": "reward"},
            {"type": "discrete", "input_field": "episode_done"},
            {
                "type": "learnable",
                "output_field": "value",
                "tokens": 1,
                "head_output": True,
            },
        ],
        objective_fields=[
            {"input_field": "action"},
            {"input_field": "reward"},
            {"input_field": "episode_done"},
            {"input_field": "task_done"},
        ],
        grouping_field="task_index",
    )


def _encoder(hidden_dim: int) -> NumericEmbedder:
    return NumericEmbedder(
        hidden_dim=hidden_dim,
        modalities=[
            {
                "type": "discrete",
                "field": "action",
                "vocab_size": _MAX_ACTIONS,
                "std": 0.02,
                "positions": 1,
            },
            {
                "type": "discrete",
                "field": "observation",
                "vocab_size": _MAX_OBS,
                "std": 0.02,
                "positions": 1,
            },
            {
                "type": "fourier",
                "field": "reward",
                "std": 0.02,
                "positions": 1,
                "fourier_min": 0.01,
                "fourier_max": 10.0,
            },
            {
                "type": "discrete",
                "field": "episode_done",
                "vocab_size": 3,
                "std": 0.02,
                "positions": 1,
            },
            {
                "type": "learnable",
                "field": "value",
                "tokens": 1,
                "std": 0.02,
                "positions": 1,
            },
        ],
    )


def _head(hidden_dim: int) -> DiscreteActionValueHead:
    return DiscreteActionValueHead(
        in_features=hidden_dim,
        out_features=_MAX_ACTIONS,
        hidden_dim=hidden_dim,
        num_layers=1,
        scale=0.1,
    )


def _identity_model(hidden_dim: int = _HIDDEN) -> Model:
    return Model(
        encoder=_encoder(hidden_dim),
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads=_head(hidden_dim),
        action_head="action_value",
        reasoner=None,
        recurrence=None,
    ).train()


def _llama_model(hidden_dim: int = _HIDDEN) -> Model:
    return Model(
        encoder=_encoder(hidden_dim),
        backbone=LlamaBackbone(
            train_kernel="reference",
            decode_kernel="flex",
            dtype=torch.float32,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=2,
            max_position_embeddings=64,
        ),
        heads=_head(hidden_dim),
        action_head="action_value",
        reasoner=None,
        recurrence=None,
    ).train()


def _optimizers(model: Model) -> tuple[AdamW, AdamW]:
    head_optimizer = AdamW(
        model.heads.parameters(),
        lr=1e-2,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    feature_optimizer = AdamW(
        list(model.encoder.parameters()) + list(model.backbone.parameters()),
        lr=1e-2,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return head_optimizer, feature_optimizer


def _objective() -> DqnObjective:
    return DqnObjective(
        gamma_step=1.0,
        gamma_episode_terminal=1.0,
        gamma_episode_truncated=1.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
        grouping_field="task_index",
    )


def run_train(
    *,
    model: Model,
    delayed_model: Model,
    polyak: Polyak,
    head_optimizer: AdamW,
    feature_optimizer: AdamW,
    objective: DqnObjective,
    loader: DataLoader,
    num_steps: int,
    head_updates: int,
    tau_heads: float,
    tau_encoder: float,
    tau_backbone: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Notebook ``run_train``: ``head_updates`` head steps per encoder/backbone step."""
    if head_updates < 1:
        raise ValueError(f"HEAD_UPDATES must be >= 1, got {head_updates}.")
    model.train()
    loss: torch.Tensor | None = None
    metrics: dict[str, float] = {}
    for _ in range(num_steps):
        inputs, objective_data = loader.next_batch()
        features = model.features(inputs)
        with torch.no_grad():
            delayed_features = delayed_model.features(inputs)
            delayed_preds = delayed_model.head(h=delayed_features)
        objective_data = objective_data.to(features.device)
        for i in range(head_updates):
            last = i == head_updates - 1
            h = features if last else features.detach()
            preds = model.head(h=h)
            loss, metrics = objective(objective_data, preds, delayed_preds)
            head_optimizer.zero_grad()
            if last:
                feature_optimizer.zero_grad()
            loss.backward()
            head_optimizer.step()
            if last:
                feature_optimizer.step()
            step_tau_heads = tau_heads
            step_tau_encoder = tau_encoder if last else 0.0
            step_tau_backbone = tau_backbone if last else 0.0
            if step_tau_heads > 0.0 or step_tau_encoder > 0.0 or step_tau_backbone > 0.0:
                polyak.update(
                    tau_heads=step_tau_heads,
                    tau_encoder=step_tau_encoder,
                    tau_backbone=step_tau_backbone,
                )
            if step_tau_heads > 0.0:
                with torch.no_grad():
                    delayed_preds = delayed_model.head(h=delayed_features)
    assert loss is not None
    return (loss, metrics)


def _loader() -> DataLoader:
    return DataLoader(
        stores=_store(),
        sequence_length=8,
        batch_size=2,
        transform=_tokenizer(),
        prefetch=1,
        num_workers=0,
        seed=0,
    )


def test_head_updates_must_be_at_least_one() -> None:
    model = _identity_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    head_opt, feat_opt = _optimizers(model)
    loader = _loader()
    try:
        try:
            run_train(
                model=model,
                delayed_model=delayed,
                polyak=polyak,
                head_optimizer=head_opt,
                feature_optimizer=feat_opt,
                objective=_objective(),
                loader=loader,
                num_steps=1,
                head_updates=0,
                tau_heads=0.5,
                tau_encoder=0.5,
                tau_backbone=0.5,
            )
        except ValueError as exc:
            assert "HEAD_UPDATES" in str(exc)
        else:
            raise AssertionError("expected ValueError for head_updates < 1")
    finally:
        loader.close()


def test_composite_step_updates_head_then_features() -> None:
    torch.manual_seed(0)
    model = _identity_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    head_opt, feat_opt = _optimizers(model)
    loader = _loader()
    objective = _objective()
    try:
        inputs, objective_data = loader.next_batch()
        head_before = _clone_params(model.heads)
        enc_before = _clone_params(model.encoder)
        dhead_before = _clone_params(delayed.heads)
        denc_before = _clone_params(delayed.encoder)

        features = model.features(inputs)
        assert features.ndim == 2
        assert features.requires_grad
        with torch.no_grad():
            delayed_features = delayed.features(inputs)
            delayed_preds = delayed.head(h=delayed_features)
        objective_data = objective_data.to(features.device)
        head_updates = 4
        for i in range(head_updates):
            last = i == head_updates - 1
            h = features if last else features.detach()
            preds = model.head(h=h)
            loss, metrics = objective(objective_data, preds, delayed_preds)
            assert torch.isfinite(loss)
            head_opt.zero_grad()
            if last:
                feat_opt.zero_grad()
            loss.backward()
            enc_has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.encoder.parameters()
            )
            head_opt.step()
            if last:
                feat_opt.step()
            tau_heads = 0.5
            tau_encoder = 0.5 if last else 0.0
            tau_backbone = 0.5 if last else 0.0
            if tau_heads > 0.0 or tau_encoder > 0.0 or tau_backbone > 0.0:
                polyak.update(
                    tau_heads=tau_heads,
                    tau_encoder=tau_encoder,
                    tau_backbone=tau_backbone,
                )
            if tau_heads > 0.0:
                with torch.no_grad():
                    delayed_preds = delayed.head(h=delayed_features)
            if i == 0:
                assert _changed(head_before, model.heads)
                assert not _changed(enc_before, model.encoder)
                assert _changed(dhead_before, delayed.heads)
                assert not _changed(denc_before, delayed.encoder)
                assert not enc_has_grad
            if last:
                assert _changed(enc_before, model.encoder)
                assert _changed(denc_before, delayed.encoder)
                assert enc_has_grad
        assert "q_values_mean" in metrics
    finally:
        loader.close()


def test_run_train_two_composite_steps() -> None:
    torch.manual_seed(0)
    model = _identity_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    head_opt, feat_opt = _optimizers(model)
    loader = _loader()
    try:
        loss, metrics = run_train(
            model=model,
            delayed_model=delayed,
            polyak=polyak,
            head_optimizer=head_opt,
            feature_optimizer=feat_opt,
            objective=_objective(),
            loader=loader,
            num_steps=2,
            head_updates=4,
            tau_heads=0.0001,
            tau_encoder=0.01,
            tau_backbone=0.01,
        )
        assert torch.isfinite(loss)
        assert math.isfinite(metrics["q_values_mean"])
    finally:
        loader.close()


def test_head_updates_one_updates_all_sections() -> None:
    torch.manual_seed(0)
    model = _identity_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    head_opt, feat_opt = _optimizers(model)
    loader = _loader()
    head_before = _clone_params(model.heads)
    enc_before = _clone_params(model.encoder)
    dhead_before = _clone_params(delayed.heads)
    denc_before = _clone_params(delayed.encoder)
    try:
        run_train(
            model=model,
            delayed_model=delayed,
            polyak=polyak,
            head_optimizer=head_opt,
            feature_optimizer=feat_opt,
            objective=_objective(),
            loader=loader,
            num_steps=1,
            head_updates=1,
            tau_heads=0.5,
            tau_encoder=0.5,
            tau_backbone=0.5,
        )
        assert _changed(head_before, model.heads)
        assert _changed(enc_before, model.encoder)
        assert _changed(dhead_before, delayed.heads)
        assert _changed(denc_before, delayed.encoder)
    finally:
        loader.close()


def test_llama_backbone_composite_step() -> None:
    torch.manual_seed(0)
    model = _llama_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    head_opt, feat_opt = _optimizers(model)
    loader = _loader()
    bb_before = _clone_params(model.backbone)
    dbb_before = _clone_params(delayed.backbone)
    try:
        loss, _metrics = run_train(
            model=model,
            delayed_model=delayed,
            polyak=polyak,
            head_optimizer=head_opt,
            feature_optimizer=feat_opt,
            objective=_objective(),
            loader=loader,
            num_steps=1,
            head_updates=3,
            tau_heads=0.5,
            tau_encoder=0.5,
            tau_backbone=0.5,
        )
        assert torch.isfinite(loss)
        assert _changed(bb_before, model.backbone)
        assert _changed(dbb_before, delayed.backbone)
    finally:
        loader.close()


def test_features_match_forward_pool_and_skip_heads() -> None:
    torch.manual_seed(0)
    model = _identity_model()
    loader = _loader()
    try:
        inputs, _ = loader.next_batch()
        out = model(inputs)
        features = model.features(inputs)
        pooled = out.last_hidden_state[out.head_output_indices]
        assert torch.allclose(features, pooled)
        preds = model.head(h=features)
        assert torch.allclose(preds["action_value"], out.predictions["action_value"])
        features.sum().backward()
        assert all(p.grad is None for p in model.heads.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    finally:
        loader.close()


def test_zero_tau_heads_scores_delayed_head_once() -> None:
    torch.manual_seed(0)
    model = _identity_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    head_opt, feat_opt = _optimizers(model)
    loader = _loader()
    calls = {"n": 0}
    original = delayed.head

    def counted(*, h, batch_size=None):
        calls["n"] += 1
        return original(h=h, batch_size=batch_size)

    delayed.head = counted  # type: ignore[method-assign]
    try:
        run_train(
            model=model,
            delayed_model=delayed,
            polyak=polyak,
            head_optimizer=head_opt,
            feature_optimizer=feat_opt,
            objective=_objective(),
            loader=loader,
            num_steps=1,
            head_updates=4,
            tau_heads=0.0,
            tau_encoder=0.5,
            tau_backbone=0.5,
        )
        assert calls["n"] == 1
    finally:
        loader.close()
