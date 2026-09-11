from __future__ import annotations

"""Tests for LayerwiseDiscreteActionValueHead and Model integration."""
import torch
from tensordict import TensorDict
from mouse_core.models.backbone import Qwen3Backbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.data import NumericTokenizer
from mouse_core.models.heads import LayerwiseDiscreteActionValueHead
from mouse_core.models.base import Model
from mouse_core.objectives import LayerwiseDqnObjective
from mouse_core.polyak import Polyak
from tests._token_batch_helpers import batch_to_packed, tok_from_encoder

_tok = tok_from_encoder

def _tiny_batch() -> list[list[dict]]:
    return [[{'action': 0, 'observation': 1, 'reward': 0.0, 'episode_done': 0, 'task_done': 0}, {'action': 1, 'observation': 2, 'reward': 1.0, 'episode_done': 0, 'task_done': 0}, {'action': 0, 'observation': 3, 'reward': 0.5, 'episode_done': 0, 'task_done': 0}]]

def test_layerwise_head_forward_shape() -> None:
    head = LayerwiseDiscreteActionValueHead(num_backbone_layers=2, in_features=8, out_features=4, hidden_dim=8, num_layers=1, scale=0.1)
    h = torch.randn(1, 2, 3, 8)
    q = head.forward(h)
    assert q.shape == (1, 3, 2, 4)

def test_model_layerwise_forward_and_objective() -> None:
    backbone = Qwen3Backbone(train_kernel="varlen", decode_kernel="flex", dtype=torch.float32, hidden_dim=16, num_layers=2, num_heads=2)
    encoder = NumericEmbedder(hidden_dim=backbone.hidden_dim, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}, {"type": 'discrete', "field": "observation", "vocab_size": 8, "std": 0.02, "positions": 1}, {"type": 'fourier', "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0}, {"type": 'discrete', "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1}])
    head = LayerwiseDiscreteActionValueHead(num_backbone_layers=2, in_features=backbone.hidden_dim, out_features=4, hidden_dim=backbone.hidden_dim, num_layers=1, scale=0.1)
    model = Model(encoder=encoder, backbone=backbone, heads=head, action_head="action_value_layerwise", reasoner=None, recurrence=None)
    batch = _tiny_batch()
    token_batch, objective_data = batch_to_packed(
        _tok(
            model.encoder,
            objective_fields=["action", "observation", "reward", "episode_done", "task_done"],
        ),
        batch,
    )
    delayed = model.delayed_copy()
    out = model(token_batch)
    predictions = out.predictions
    with torch.no_grad():
        delayed_predictions = delayed(token_batch).predictions
    assert 'action_value_layerwise' in predictions.keys()
    assert predictions['action_value_layerwise'].shape[-2:] == (2, 4)
    objective = LayerwiseDqnObjective(num_backbone_layers=2, gamma_step_start=0.0, gamma_step=0.99)
    loss, metrics = objective(objective_data, predictions, delayed_predictions)
    assert loss.ndim == 0
    assert metrics['action_value_layerwise'] >= 0.0
    action = model.get_action(predictions, temperature=0.0, num_actions=4)
    assert action.shape == (1,)
    Polyak(model, delayed).update(tau_heads=0.1, tau_encoder=0.1, tau_backbone=0.1)

def test_layerwise_objective_q_metrics_use_curr_max_q() -> None:
    """q_values_mean and layer_q_mean report max online Q at the current state."""
    step_stream = TensorDict({'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 0, 0]), 'task_done': torch.tensor([0, 0, 0])}, batch_size=[3])
    predictions = TensorDict({'action_value_layerwise': torch.tensor([[[0.0, 2.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])}, batch_size=[3])
    delayed = TensorDict({'action_value_layerwise': torch.zeros(3, 2, 2)}, batch_size=[3])
    _, metrics = LayerwiseDqnObjective(num_backbone_layers=2, gamma_step_start=0.0, gamma_step=0.0)(step_stream, predictions, delayed)
    assert abs(metrics['q_values_mean'] - 1.5) < 1e-05
    assert abs(metrics['layer_0_q_mean'] - 1.0) < 1e-05
    assert abs(metrics['layer_1_q_mean'] - 1.5) < 1e-05


def _layerwise_lambda_fixture() -> tuple[TensorDict, TensorDict, TensorDict]:
    """The DQN λ fixture on two layers (layer 0 gamma 0.5, layer 1 gamma 0.9).

    Action from s0 is 0, from s1 is 1; rewards out of s0 / s1 are 1 and 10;
    delayed max-Q is 3 at s1 and 100 at s2; online Q(s0, 0) = 5, Q(s1, 1) = 0.
    """
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0, 1]),
            "reward": torch.tensor([0.0, 1.0, 10.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
        },
        batch_size=[3],
    )
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 100.0]])
    predictions = TensorDict(
        {"action_value_layerwise": torch.stack([online, online], dim=1)}, batch_size=[3]
    )
    delayed_td = TensorDict(
        {"action_value_layerwise": torch.stack([delayed, delayed], dim=1)}, batch_size=[3]
    )
    return step_stream, predictions, delayed_td


# Layer 1 (gamma 0.9): one-step 3.7 / 100 → ((5-3.7)^2 + 100^2) / 2 = 5000.845;
#   λ=1: G_0 = 1 + 0.9 * 100 = 91 → (7396 + 10000) / 2 = 8698.
# Layer 0 (gamma 0.5): one-step 2.5 / 60 → ((5-2.5)^2 + 60^2) / 2 = 1803.125;
#   λ=1: G_0 = 1 + 0.5 * 60 = 31 → (676 + 3600) / 2 = 2138.
def _layerwise(td_lambda: float = 0.0, watkins: bool = False) -> LayerwiseDqnObjective:
    return LayerwiseDqnObjective(
        num_backbone_layers=2, gamma_step_start=0.5, gamma_step=0.9,
        td_lambda=td_lambda, watkins=watkins,
    )


def test_layerwise_td_lambda_zero_is_one_step() -> None:
    step_stream, predictions, delayed = _layerwise_lambda_fixture()
    loss, metrics = _layerwise()(step_stream, predictions, delayed)
    assert abs(loss.item() - (1803.125 + 5000.845) / 2) < 1e-02
    assert "watkins_greedy_frac" not in metrics


def test_layerwise_td_lambda_uses_each_layers_discount() -> None:
    step_stream, predictions, delayed = _layerwise_lambda_fixture()
    loss, metrics = _layerwise(td_lambda=1.0)(step_stream, predictions, delayed)
    assert abs(metrics["layer_0_loss"] - 2138.0) < 1e-02
    assert abs(metrics["layer_1_loss"] - 8698.0) < 1e-02
    assert abs(loss.item() - (2138.0 + 8698.0) / 2) < 1e-02


def test_layerwise_watkins_cuts_per_layer() -> None:
    """Layer 1 prefers a=0 at s1 (taken a=1 → cut); layer 0 ties (→ continue)."""
    step_stream, predictions, delayed = _layerwise_lambda_fixture()
    q = predictions["action_value_layerwise"].clone()
    q[1, 1] = torch.tensor([10.0, 0.0])
    predictions["action_value_layerwise"] = q
    loss, metrics = _layerwise(td_lambda=1.0, watkins=True)(step_stream, predictions, delayed)
    assert abs(metrics["layer_0_loss"] - 2138.0) < 1e-02
    assert abs(metrics["layer_1_loss"] - 5000.845) < 1e-02
    assert abs(metrics["watkins_greedy_frac"] - 0.5) < 1e-06
