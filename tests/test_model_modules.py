"""Custom Python forwards, standard composition, and online-only persistence."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mouse_core import AdamW, model_polyak
from mouse_core.models import Model, TokenEncoder, load_model, save_model
from mouse_core.models.mlp import MLP
from mouse_core.models.backbone import Backbone

from tests._model_helpers import ExampleModel, fresh_model
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer


def _backbone() -> Backbone:
    return Backbone(architecture="qwen3", hidden_dim=8, num_layers=1, num_heads=2,
                               vocab_size=32, train_kernel="reference", decode_kernel="flex", dtype=torch.float32)


def _batch(start=0, stop=3):
    return batch_to_token_batch(token_tokenizer("action", "episode_done"),
                               [[{"action": i % 4, "episode_done": 0} for i in range(start, stop)]])


class BranchedModel(Model):
    def __init__(self, projection=True, stacked=True):
        super().__init__()
        self.encoder = TokenEncoder([_backbone() for _ in range(2 if stacked else 1)])
        self.projection = (nn.Sequential(
            MLP(in_features=8, out_features=8, hidden_dim=12),
            MLP(in_features=8, hidden_dim=12, out_features=4),
        ) if projection else nn.Identity())
        width = 4 if projection else 8
        self.skip = nn.Linear(8, width)
        self.value = MLP(in_features=width, out_features=1, hidden_dim=8,
                                    num_layers=2, propagate_gradient=1.)
        self.policy = MLP(in_features=width, out_features=4, hidden_dim=8,
                                         num_layers=1, propagate_gradient=1.)

    def get_action(self, *, out):
        # This implementation chooses continuous actions from its scalar output.
        return out.predictions["baseline"].tanh()

    def forward(self, batch, *, cache=None, use_cache=False, residual=True):
        encoded = self.encoder(batch, cache=cache, use_cache=use_cache)
        h = encoded.pool().float()
        features = self.projection(h)
        if residual:
            features = features + self.skip(h)
        return encoded.with_predictions({"baseline": self.value(features), "action": self.policy(features)})


@pytest.mark.parametrize("projection", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_custom_forward_gradients_head_tokens_and_cache(projection, stacked):
    model = BranchedModel(projection, stacked)
    seen = []
    handle = model.projection.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape))
    full = model(_batch())
    handle.remove()
    assert seen == [torch.Size([3, 8])] and _batch().L > 3
    torch.stack([value.square().mean() for value in full.predictions.values()]).sum().backward()
    for module in (model.encoder, model.skip, model.value, model.policy):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert not torch.equal(full.predictions["baseline"], model(_batch(), residual=False).predictions["baseline"])
    model.eval()
    with torch.no_grad():
        full = model(_batch())
        first = model(_batch(0, 2), use_cache=True)
        assert first.cache is not None
        try:
            assert len(first.cache.sessions) == (2 if stacked else 1)
            last = model(_batch(2, 3), cache=first.cache, use_cache=True)
            for name in full.predictions:
                actual = torch.cat([first.predictions[name], last.predictions[name]], dim=1).squeeze(0)
                torch.testing.assert_close(actual, full.predictions[name], atol=2e-5, rtol=2e-5)
            first.cache.reset_rows([0])
            reset = model(_batch(0, 1), use_cache=True, cache=first.cache)
            reference = model(_batch(0, 1))
            torch.testing.assert_close(reset.predictions["action"].squeeze(0), reference.predictions["action"], atol=2e-5, rtol=2e-5)
        finally:
            first.cache.close()


def test_custom_copy_preserves_forward_and_module_rates():
    model = BranchedModel()
    rates = {"encoder": 0., "projection.0": 0., "projection.1": 0.2,
             "skip": 0., "value": 0., "policy": 0.}
    optimizer = AdamW(params=model.parameter_groups(lr=rates), lr=0., weight_decay=0., betas=(0., 0.), eps=0.)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name] - (0.2 if name.startswith("projection.1.") else 0.))
    delayed = model.copy(modules=["projection", "skip", "value", "policy"])
    assert type(delayed) is BranchedModel and delayed.encoder is model.encoder
    assert all(not p.requires_grad for p in delayed.projection.parameters())
    assert all(p.requires_grad for p in model.projection.parameters())
    assert isinstance(model.projection, nn.Sequential) and isinstance(delayed.projection, nn.Sequential)
    update = model_polyak(online=model, delayed=delayed, modules=tuple(rates))
    with torch.no_grad():
        old = {name: p.clone() for name, p in delayed.projection[1].named_parameters()}
        for p in model.projection[1].parameters():
            p.add_(1.)
        update(tau={name: None if name == "encoder" else (0.25 if name == "projection.1" else 0.) for name in rates})
        for name, p in delayed.projection[1].named_parameters():
            torch.testing.assert_close(p, old[name] + 0.25)
    output = delayed(_batch(), residual=False)
    assert output.predictions["action"].shape == (3, 4)
    torch.testing.assert_close(delayed.get_action(out=output), output.predictions["baseline"].tanh())


def test_groups_reject_overlap_missing_and_unknown_modules():
    model = BranchedModel()
    for rates, match in [({"projection": .1, "projection.0": .1}, "overlap"),
                         ({"missing": .1}, "unknown"), ({"projection": .1}, "every trainable")]:
        with pytest.raises(ValueError, match=match):
            model.parameter_groups(lr=rates)


@pytest.mark.parametrize('projection', [False, True])
def test_explicit_composition_checkpoint_roundtrip(tmp_path: Path, projection):
    model = ExampleModel(encoder=TokenEncoder([_backbone(), _backbone()]), projection=nn.Sequential(MLP(in_features=8, out_features=8, hidden_dim=12), MLP(in_features=8, hidden_dim=12, out_features=4)) if projection else None, heads={'q': MLP(in_features=4 if projection else 8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0)}).eval()
    save_model(model=model, path=tmp_path)
    loaded = load_model(model=fresh_model(model), repo_id_or_path=str(tmp_path))
    torch.testing.assert_close(loaded(_batch()).predictions['q'], model(_batch()).predictions['q'])
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_custom_checkpoint_saves_only_online_state(tmp_path: Path):
    model = BranchedModel()
    optimizer = AdamW(params=model.parameters(), lr=0.01)
    delayed = model.copy()
    model(_batch()).predictions['baseline'].square().mean().backward()
    optimizer.step()
    assert optimizer.state_dict()['inner']['state']
    assert any((not torch.equal(p, dict(delayed.named_parameters())[name]) for name, p in model.named_parameters()))
    save_model(model=model, path=tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {'config.json', 'pytorch_model.bin'}
    state = torch.load(tmp_path / 'pytorch_model.bin', weights_only=True)
    assert set(state) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(state[name], tensor)
    config = json.loads((tmp_path / 'config.json').read_text())
    assert 'optimizer' not in config and 'delayed' not in config
    with pytest.raises(TypeError, match='model'):
        load_model(repo_id_or_path=str(tmp_path))  # type: ignore[call-arg]
    restored = BranchedModel()
    result = load_model(repo_id_or_path=str(tmp_path), model=restored)
    assert result is restored
    output = restored(_batch())
    torch.testing.assert_close(output.predictions['action'], model(_batch()).predictions['action'])
    torch.testing.assert_close(restored.get_action(out=output), output.predictions['baseline'].tanh())


def test_mlp_instances_compose_as_projection_and_scalar_prediction():
    projection = MLP(in_features=8, out_features=4, hidden_dim=12)
    regression = MLP(in_features=4, out_features=1, hidden_dim=8, scale=0.1)
    h = torch.randn(5, 8, requires_grad=True)
    values = regression(projection(h))
    assert values.shape == (5, 1)
    values.square().mean().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert not {id(p) for p in projection.parameters()} & {id(p) for p in regression.parameters()}
    assert all(p.grad is not None for p in (*projection.parameters(), *regression.parameters()))
