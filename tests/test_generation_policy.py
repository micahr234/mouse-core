"""Shared generation preserves behavior probabilities used by policy training."""
import numpy as np
import pytest
import torch
from torch import nn

from mouse_core.models import Backbone, generate
from mouse_core.objectives import GrpoObjective
from mouse_core.sampling import sample_tempered_actions


@pytest.mark.parametrize('backend', ['continuous', 'uncached'])
def test_policy_generation_recycles_requests_and_matches_grpo_replay(backend):
    torch.manual_seed(23)
    backbone = Backbone(architecture='qwen3', hidden_dim=8, num_layers=1,
                        num_heads=2, head_dim=4, max_position_embeddings=128,
                        dtype=torch.float32).eval()
    head = nn.Linear(8, 4).eval()
    prompts = [torch.randn(size, 8) for size in (2, 3, 1, 4)]
    durations = [3, 1, 2, 1]
    # Environment observations can be multi-token chunks, just like text or
    # residual continuation inputs. The scheduler must preserve their context.
    observations = [[torch.randn(2, 8) for _ in range(n - 1)] for n in durations]
    histories = [prompt.clone() for prompt in prompts]
    actions, logs, features = ([[] for _ in prompts] for _ in range(3))
    rng = np.random.default_rng(37)
    admissions = []
    temperature = .7
    def advance(requests, hidden):
        assert not torch.is_grad_enabled()
        admissions.append(list(requests))
        chosen, behavior = sample_tempered_actions(head(hidden), [temperature] * len(requests), rng)
        chunks = []
        for i, request in enumerate(requests):
            reference = backbone(histories[request], sequence_ids=torch.zeros(
                len(histories[request]), dtype=torch.long))[-1]
            torch.testing.assert_close(hidden[i], reference, atol=3e-5, rtol=3e-5)
            expected_log = (head(reference) / temperature).log_softmax(-1)[chosen[i]]
            torch.testing.assert_close(behavior[i], expected_log, atol=3e-5, rtol=3e-5)
            features[request].append(reference)
            actions[request].append(chosen[i])
            logs[request].append(behavior[i])
            count = len(actions[request])
            if count == durations[request]:
                chunks.append(None)
            else:
                chunk = observations[request][count - 1]
                histories[request] = torch.cat((histories[request], chunk))
                chunks.append(chunk)
        return chunks
    generate(backbone=backbone, prompts=prompts, advance=advance,
             backend=backend, max_active_requests=2, max_batch_tokens=4,
             cache_capacity_tokens=1024)
    assert admissions == [[0, 1], [0, 2], [0, 2], [3]]
    assert all(parameter.grad is None for parameter in head.parameters())
    # Training recomputes policy logits outside generation's no-grad scope.
    # Every branch includes an initial state and a final destination row.
    data_actions, data_logs, advantages, groups, replay_features = [], [], [], [], []
    for request, duration in enumerate(durations):
        data_actions.extend([torch.tensor(float('nan')), *actions[request]])
        data_logs.extend([torch.tensor(0.), *logs[request]])
        advantages.extend([0.] + [(-1.) ** request] * duration)
        groups.extend([request] * (duration + 1))
        replay_features.extend([*features[request], features[request][-1]])
    predictions = head(torch.stack(replay_features)) / temperature
    data = dict(action=torch.stack(data_actions), old_log_prob=torch.stack(data_logs),
                advantage=torch.tensor(advantages))
    losses, _ = GrpoObjective(ent_coef=0.)(
        predictions=predictions, objective_data=data, sequence_ids=torch.tensor(groups),
        step_ids=torch.arange(len(groups)))
    losses['grpo'].backward()
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
    assert not data['old_log_prob'].requires_grad
