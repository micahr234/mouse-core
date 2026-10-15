from __future__ import annotations

import json

import pytest
import torch

from mouse_core.models import DiscretizedActionHead, Model, load_model, model_polyak, save_model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.heads import RegressionHead


def make_head(table: torch.Tensor) -> DiscretizedActionHead:
    return DiscretizedActionHead(
        in_features=8, action_vectors=table, hidden_dim=16, num_layers=2,
        propagate_gradient=1.0,
    )


def make_model(head: DiscretizedActionHead) -> Model:
    return Model(
        backbone=IdentityBackbone(hidden_dim=8, vocab_size=16),
        heads=head, action_source="action_value", reasoner=None,
    )


def test_vector_lookup_preserves_shape_and_does_not_alias_table() -> None:
    table = torch.tensor([[2.0, -1.0], [-2.0, 1.0], [0.0, 0.25]])
    head = make_head(table)
    ids = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    actions = head.discrete_to_continuous(ids)
    assert actions.shape == (2, 2, 2)
    assert torch.equal(actions, table[ids])
    assert torch.equal(head.continuous_to_discrete(actions), ids.long())
    assert head.continuous_to_discrete(actions).dtype == torch.int64
    scalar_action = head.discrete_to_continuous(0)
    assert scalar_action.shape == (2,)
    assert head.continuous_to_discrete(scalar_action).shape == ()
    scalar_action.zero_()
    actions.zero_()
    assert torch.equal(head.action_vectors, table)
    assert head.discrete_to_continuous(torch.empty(0, dtype=torch.long)).shape == (0, 2)
    assert head.continuous_to_discrete(torch.empty(0, 2)).shape == (0,)


def test_quantization_uses_whole_vector_and_first_row_on_ties() -> None:
    head = make_head(torch.tensor([[-1.0, 2.0], [1.0, 2.0], [0.0, -3.0]]))
    actions = torch.tensor([
        [0.0, 2.0], [0.8, 1.9], [-100.0, 2.0], [0.0, -20.0],
    ], requires_grad=True)
    assert head.continuous_to_discrete(actions).tolist() == [0, 1, 0, 2]
    decoded = head.discrete_to_continuous(head.continuous_to_discrete(actions))
    assert not decoded.requires_grad
    assert not torch.equal(decoded, actions)  # Off-table inputs lose precision.


def test_fine_grid_roundtrips_without_distance_cancellation() -> None:
    # More than 25 rows normally enables cdist's less accurate matmul identity.
    table = torch.linspace(1.0, 1.0001, 101)[:, None]
    head = make_head(table)
    assert torch.equal(head.continuous_to_discrete(table), torch.arange(len(table)))


def test_large_table_accepts_ids_stored_in_narrow_integer_dtype() -> None:
    head = make_head(torch.arange(300, dtype=torch.float32)[:, None])
    assert head.discrete_to_continuous(torch.tensor([255], dtype=torch.uint8)).item() == 255.0
    assert head.discrete_to_continuous(torch.tensor([127], dtype=torch.int8)).item() == 127.0


@pytest.mark.parametrize("ids", [-1, 3, torch.tensor([0, -1]), torch.tensor([1, 3])])
def test_invalid_ids_do_not_wrap(ids) -> None:
    head = make_head(torch.tensor([[-2.0], [0.0], [2.0]]))
    with pytest.raises(ValueError, match="action_ids must be in"):
        head.discrete_to_continuous(ids)


@pytest.mark.parametrize("ids", [True, 1.5, torch.tensor([1.0]), torch.tensor([False])])
def test_noninteger_ids_are_rejected(ids) -> None:
    head = make_head(torch.tensor([[-2.0], [0.0], [2.0]]))
    with pytest.raises(TypeError, match="integers"):
        head.discrete_to_continuous(ids)


@pytest.mark.parametrize("table", [
    torch.tensor(1.0), torch.ones(3), torch.ones(2, 2, 2),
    torch.empty(0, 2), torch.empty(2, 0),
    torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]]),
    torch.tensor([[0.0], [0.0]]),
    torch.tensor([[1.0], [1.0 + 1e-10]], dtype=torch.float64),
])
def test_invalid_tables_are_rejected(table) -> None:
    with pytest.raises(ValueError, match="action_vectors"):
        make_head(table)


@pytest.mark.parametrize("table", [torch.tensor([[1]]), torch.tensor([[True]])])
def test_table_requires_floats(table) -> None:
    with pytest.raises(TypeError, match="floating-point"):
        make_head(table)


@pytest.mark.parametrize("actions", [
    torch.tensor(1.0), torch.ones(3), torch.empty(2, 0),
    torch.tensor([[float("nan"), 0.0]]), torch.tensor([[0.0, float("inf")]]),
])
def test_invalid_continuous_actions_are_rejected(actions) -> None:
    head = make_head(torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="actions must"):
        head.continuous_to_discrete(actions)


def test_head_trains_like_regression_but_table_is_fixed_and_copied() -> None:
    table = torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float64, requires_grad=True)
    head = make_head(table)
    expected_table = table.detach().float().clone()
    with torch.no_grad():
        table.fill_(99.0)
    assert torch.equal(head.action_vectors, expected_table)
    assert head.action_vectors.dtype == torch.float32
    assert not head.action_vectors.requires_grad
    assert "action_vectors" not in dict(head.named_parameters())
    assert "action_vectors" in head.state_dict()

    regression = RegressionHead(
        in_features=8, out_features=3, hidden_dim=16, num_layers=2,
        propagate_gradient=1.0,
    )
    regression.load_state_dict({k: v for k, v in head.state_dict().items() if k != "action_vectors"})
    h = torch.randn(2, 4, 8, requires_grad=True)
    output = head(h)
    assert output.shape == (2, 4, 3)
    assert torch.equal(output, regression(h))
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    output[..., 1].square().mean().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(head(h), output)
    assert torch.equal(head.action_vectors, expected_table)

    model = make_model(head)
    delayed = model.copy(heads=True, backbone=False, reasoner=False)
    delayed_head = delayed.heads["action_value"]
    assert isinstance(delayed_head, DiscretizedActionHead)
    assert delayed_head.action_vectors.data_ptr() != head.action_vectors.data_ptr()
    update = model_polyak(online=model, delayed=delayed)
    update(tau_heads=0.5, tau_backbone=None)
    assert torch.equal(delayed_head.action_vectors, expected_table)
    assert torch.equal(delayed_head.continuous_to_discrete(expected_table), torch.arange(3))


def test_model_checkpoint_preserves_both_conversions_and_q_values(tmp_path) -> None:
    head = make_head(torch.tensor([[2.0, -1.0], [-2.0, 1.0], [0.0, 0.25]]))
    model = Model(
        backbone=IdentityBackbone(hidden_dim=8, vocab_size=16),
        heads={"motor": head}, action_source="motor", reasoner=None,
    )
    h = torch.randn(2, 8)
    expected = model.head(h=h)["motor"]
    save_model(model=model, path=tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {"config.json", "pytorch_model.bin"}
    config = json.loads((tmp_path / "config.json").read_text())
    spec = config["heads"]["heads"][0]
    assert spec["type"] == "discretized_action"
    assert spec["out_features"] == 3 and spec["action_dim"] == 2
    assert "action_vectors" not in spec  # Saved once, in the state dict.
    loaded = load_model(
        repo_id_or_path=str(tmp_path), train_kernel="reference", decode_kernel="flex",
        dtype=torch.float32,
    )
    restored = loaded.heads["motor"]
    assert isinstance(restored, DiscretizedActionHead)
    assert torch.equal(loaded.head(h=h)["motor"], expected)
    assert torch.equal(restored.discrete_to_continuous(torch.arange(3)), head.action_vectors)
    assert torch.equal(restored.continuous_to_discrete(head.action_vectors), torch.arange(3))
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])

    spec["action_dim"] = 99
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Action table shape"):
        load_model(
            repo_id_or_path=str(tmp_path), train_kernel="reference", decode_kernel="flex",
            dtype=torch.float32,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_table_moves_with_model_and_conversions_accept_cpu_inputs() -> None:
    head = make_head(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    make_model(head).to("cuda")
    actions = head.discrete_to_continuous(torch.tensor([0, 1]))
    ids = head.continuous_to_discrete(torch.tensor([[-0.5, 0.0], [0.5, 0.0]]))
    assert head.action_vectors.device.type == actions.device.type == ids.device.type == "cuda"
    assert ids.tolist() == [0, 1]
