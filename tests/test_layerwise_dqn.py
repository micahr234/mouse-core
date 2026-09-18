from __future__ import annotations

"""Tests for LayerwiseRegressionHead and Model integration."""
import torch
from mouse_core.models.backbone import TransformerBackbone
from mouse_core.data import Tokenizer
from mouse_core.models.heads import LayerwiseRegressionHead
from mouse_core.models.base import Model
from mouse_core.objectives import LayerwiseDqnObjective
from tests._bound_head import BoundHead
from mouse_core.polyak import Polyak
from tests._token_batch_helpers import batch_to_packed, batch_to_token_batch, token_tokenizer

def _tiny_batch() -> list[list[dict]]:
    return [[{'action': 0, 'observation': 1, 'reward': 0.0, 'episode_done': 0, 'task_done': 0}, {'action': 1, 'observation': 2, 'reward': 1.0, 'episode_done': 0, 'task_done': 0}, {'action': 0, 'observation': 3, 'reward': 0.5, 'episode_done': 0, 'task_done': 0}]]

def test_layerwise_head_forward_shape() -> None:
    head = LayerwiseRegressionHead(num_backbone_layers=2, in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True, scale=0.1)
    h = torch.randn(1, 2, 3, 8)
    q = head.forward(h)
    assert q.shape == (1, 3, 2, 4)

def test_model_layerwise_forward_and_objective() -> None:
    backbone = TransformerBackbone(architecture="qwen3", train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, hidden_dim=16, num_layers=2, num_heads=2, vocab_size=32)
    head = LayerwiseRegressionHead(num_backbone_layers=2, in_features=backbone.hidden_dim, out_features=4, hidden_dim=backbone.hidden_dim, num_layers=1, use_norm=True, scale=0.1)
    model = Model(backbone=backbone, heads=head, action_source=head, reasoner=None)
    batch = _tiny_batch()
    token_batch, objective_data = batch_to_packed(
        token_tokenizer(
            "action",
            "observation",
            "episode_done",
            objective_fields=["action", "observation", "reward", "episode_done", "task_done"],
        ),
        batch,
    )
    delayed = model.delayed_copy(heads=(model._heads["action_value_layerwise"],))
    out = model(token_batch)
    predictions = out.predictions
    with torch.no_grad():
        delayed_predictions = delayed(token_batch).predictions
    assert 'action_value_layerwise' in predictions.keys()
    assert predictions['action_value_layerwise'].shape[-2:] == (2, 4)
    objective = LayerwiseDqnObjective(head=model._heads["action_value_layerwise"], num_backbone_layers=2, gamma_step_start=0.0, gamma_step=0.99, gamma_episode_terminal_start=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated_start=0.0, gamma_episode_truncated=0.0, gamma_task_terminal_start=0.0, gamma_task_terminal=0.0, gamma_task_truncated_start=0.0, gamma_task_truncated=0.0, grouping_field=None, temperature=0.0)
    loss, metrics = objective(objective_data=objective_data, predictions=predictions, delayed_predictions=delayed_predictions)
    assert loss.ndim == 0
    assert metrics['action_value_layerwise'] >= 0.0
    Polyak(online=model, delayed=delayed).update(tau_heads=0.1, tau_backbone=0.1)

def test_layerwise_objective_q_metrics_use_curr_max_q() -> None:
    """q_values_mean and layer_q_mean report max online Q at the current state."""
    step_stream = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 0, 0]), 'task_done': torch.tensor([0, 0, 0])}
    predictions = {'action_value_layerwise': torch.tensor([[[0.0, 2.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])}
    delayed = {'action_value_layerwise': torch.zeros(3, 2, 2)}
    _, metrics = LayerwiseDqnObjective(head=BoundHead("action_value_layerwise"), num_backbone_layers=2, gamma_step_start=0.0, gamma_step=0.0, gamma_episode_terminal_start=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated_start=0.0, gamma_episode_truncated=0.0, gamma_task_terminal_start=0.0, gamma_task_terminal=0.0, gamma_task_truncated_start=0.0, gamma_task_truncated=0.0, grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics['q_values_mean'] - 1.5) < 1e-05
    assert abs(metrics['layer_0_q_mean'] - 1.0) < 1e-05
    assert abs(metrics['layer_1_q_mean'] - 1.5) < 1e-05


def _layerwise_lambda_fixture() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The DQN λ fixture on two layers (layer 0 gamma 0.5, layer 1 gamma 0.9).

    Action from s0 is 0, from s1 is 1; rewards out of s0 / s1 are 1 and 10;
    delayed max-Q is 3 at s1 and 100 at s2; online Q(s0, 0) = 5, Q(s1, 1) = 0.
    """
    step_stream = {
            "action": torch.tensor([0, 0, 1]),
            "reward": torch.tensor([0.0, 1.0, 10.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
        }
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 100.0]])
    predictions = {"action_value_layerwise": torch.stack([online, online], dim=1)}
    delayed_td = {"action_value_layerwise": torch.stack([delayed, delayed], dim=1)}
    return step_stream, predictions, delayed_td


# Layer 1 (gamma 0.9): one-step 3.7 / 100 → ((5-3.7)^2 + 100^2) / 2 = 5000.845;
#   λ=1: G_0 = 1 + 0.9 * 100 = 91 → (7396 + 10000) / 2 = 8698.
# Layer 0 (gamma 0.5): one-step 2.5 / 60 → ((5-2.5)^2 + 60^2) / 2 = 1803.125;
#   λ=1: G_0 = 1 + 0.5 * 60 = 31 → (676 + 3600) / 2 = 2138.
def _layerwise(td_lambda: float = 0.0, watkins: bool = False) -> LayerwiseDqnObjective:
    return LayerwiseDqnObjective(head=BoundHead("action_value_layerwise"), 
        num_backbone_layers=2, gamma_step_start=0.5, gamma_step=0.9,
        td_lambda=td_lambda, watkins=watkins,
        gamma_episode_terminal_start=0.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated_start=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal_start=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated_start=0.0,
        gamma_task_truncated=0.0,
        grouping_field=None, temperature=0.0)


def test_layerwise_td_lambda_zero_is_one_step() -> None:
    step_stream, predictions, delayed = _layerwise_lambda_fixture()
    loss, metrics = _layerwise()(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - (1803.125 + 5000.845) / 2) < 1e-02
    assert "watkins_greedy_frac" not in metrics


def test_layerwise_td_lambda_uses_each_layers_discount() -> None:
    step_stream, predictions, delayed = _layerwise_lambda_fixture()
    loss, metrics = _layerwise(td_lambda=1.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics["layer_0_loss"] - 2138.0) < 1e-02
    assert abs(metrics["layer_1_loss"] - 8698.0) < 1e-02
    assert abs(loss.item() - (2138.0 + 8698.0) / 2) < 1e-02


def test_layerwise_watkins_cuts_per_layer() -> None:
    """Layer 1 prefers a=0 at s1 (taken a=1 → cut); layer 0 ties (→ continue)."""
    step_stream, predictions, delayed = _layerwise_lambda_fixture()
    q = predictions["action_value_layerwise"].clone()
    q[1, 1] = torch.tensor([10.0, 0.0])
    predictions["action_value_layerwise"] = q
    loss, metrics = _layerwise(td_lambda=1.0, watkins=True)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(metrics["layer_0_loss"] - 2138.0) < 1e-02
    assert abs(metrics["layer_1_loss"] - 5000.845) < 1e-02
    assert abs(metrics["watkins_greedy_frac"] - 0.5) < 1e-06


def test_cached_decode_matches_full_forward_with_layerwise() -> None:
    torch.manual_seed(0)
    backbone = TransformerBackbone(
        architecture="qwen3",
        train_kernel="reference",
        decode_kernel="flex",
        dtype=torch.float32,
        use_norm=True,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        vocab_size=32,
    )
    head = LayerwiseRegressionHead(
        num_backbone_layers=2,
        in_features=backbone.hidden_dim,
        out_features=4,
        hidden_dim=backbone.hidden_dim,
        num_layers=1, use_norm=True,
        scale=0.1,
    )
    model = Model(
        backbone=backbone,
        heads=head,
        action_source=head,
        reasoner=None,
    ).eval()
    steps = _tiny_batch()[0]
    tok = token_tokenizer("action", "observation", "episode_done")
    with torch.no_grad():
        full = model(batch_to_token_batch(tok, [steps]))
        cache = None
        chunks = []
        for lo, hi in ((0, 1), (1, 3)):
            out = model(
                batch_to_token_batch(tok, [steps[lo:hi]]),
                cache=cache,
                use_cache=True,
            )
            cache = out.cache
            chunks.append(out.predictions["action_value_layerwise"])
        incremental = torch.cat(chunks, dim=1)
    assert torch.allclose(
        incremental,
        full.predictions["action_value_layerwise"].unsqueeze(0),
        atol=1e-5,
    )
