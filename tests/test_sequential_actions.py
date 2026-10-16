"""Sequential decisions keep action time separate from environment time."""
import pytest
import torch

from mouse_core.objectives import DqnObjective, boundary_discount, nstep_gate


def _fixture():
    # Two choices per environment step. Incoming rewards occur at rows 2, 4.
    data = dict(
        action=torch.tensor([0, 1, 0, 1, 0]),
        reward=torch.tensor([0., 0., 3., 0., 7.]),
        terminated=torch.zeros(5, dtype=torch.bool),
        truncated=torch.zeros(5, dtype=torch.bool),
        environment_step=torch.tensor([False, False, True, False, True]),
    )
    values = torch.tensor([[0., 0.], [10., 20.], [30., 40.], [50., 60.], [70., 80.]])
    return data, values


def _objective(gate):
    return DqnObjective(
        discount=boundary_discount(gamma_step=.5, gamma_terminated=0.,
                                   gamma_truncated=1., step_key="environment_step"),
        reward=None, value=None, gate=gate, cross_group_backups="bootstrap", temperature=0.0,
    )


@pytest.mark.parametrize("gate, expected", [
    (None, [20., 23., 60., 47., 0.]),
    (nstep_gate(n=2), [23., 33., 47., 47., 0.]),
    (nstep_gate(n=1, step_key="environment_step"), [23., 23., 47., 47., 0.]),
    (nstep_gate(n=2, step_key="environment_step"), [26.5, 26.5, 47., 47., 0.]),
])
def test_component_and_environment_horizons(gate, expected):
    data, values = _fixture()
    online = torch.zeros_like(values, requires_grad=True)
    losses, metrics = _objective(gate)(objective_data=data,
        group_id=torch.zeros(5, dtype=torch.int64), predictions=online,
        backup_value_predictions=values, backup_action_predictions=values)
    assert isinstance(metrics["backup"], torch.Tensor)
    torch.testing.assert_close(metrics["backup"][:4], torch.tensor(expected[:4]))
    losses["action_value"].backward()
    # Each decision regresses its OWN selected ID, stored on the following row.
    assert online.grad is not None
    assert online.grad[0, 1] != 0 and online.grad[0, 0] == 0
    assert online.grad[1, 0] != 0 and online.grad[1, 1] == 0
    assert (online.grad[-1] == 0).all()


def test_variable_component_counts_and_group_isolation():
    data, values = _fixture()
    # First step takes three choices; the second takes one.
    data["environment_step"] = torch.tensor([False, False, False, True, True])
    data["reward"] = torch.tensor([0., 0., 0., 3., 7.])
    objective = _objective(nstep_gate(n=1, step_key="environment_step"))
    def targets(data, values, groups):
        result = objective(objective_data=data, group_id=groups,
            predictions=torch.zeros_like(values), backup_value_predictions=values,
            backup_action_predictions=values)[1]["backup"]
        assert isinstance(result, torch.Tensor)
        return result
    single = targets(data, values, torch.zeros(5, dtype=torch.int64))
    torch.testing.assert_close(single[:4], torch.tensor([33., 33., 33., 47.]))
    combined = targets({k: torch.cat([v, v]) for k, v in data.items()},
                       torch.cat([values, values + 1000]),
                       torch.tensor([0] * 5 + [1] * 5))
    torch.testing.assert_close(combined[:4], single[:4])


@pytest.mark.parametrize("flag, expected", [("terminated", 3.), ("truncated", 23.)])
def test_boundary_flags_propagate_through_component_return(flag, expected):
    data, values = _fixture()
    data[flag][2] = True
    _, metrics = _objective(nstep_gate(n=1, step_key="environment_step"))(
        objective_data=data, group_id=torch.zeros(5, dtype=torch.int64),
        predictions=torch.zeros_like(values), backup_value_predictions=values,
        backup_action_predictions=values)
    assert isinstance(metrics["backup"], torch.Tensor)
    torch.testing.assert_close(metrics["backup"][:2], torch.full((2,), expected))


@pytest.mark.parametrize("bad, error", [
    (torch.tensor([0, 2, 0, 1, 0]), ValueError),
    (torch.zeros(5), TypeError),
    (torch.zeros(5, 1, dtype=torch.bool), ValueError),
    (torch.zeros(4, dtype=torch.bool), ValueError),
])
def test_environment_marker_validation(bad, error):
    data, values = _fixture()
    data["environment_step"] = bad
    with pytest.raises(error):
        nstep_gate(n=1, step_key="environment_step")(q=values, **data)
    discount = boundary_discount(gamma_step=.5, gamma_terminated=0.,
                                 gamma_truncated=1., step_key="environment_step")
    with pytest.raises(error):
        discount(**data)


def test_missing_environment_marker_is_explicit():
    data, values = _fixture()
    del data["environment_step"]
    with pytest.raises(ValueError, match="Missing environment-step"):
        nstep_gate(n=1, step_key="environment_step")(q=values, **data)


@pytest.mark.parametrize("components", [1, 2, 16])
def test_all_components_receive_reward_without_extra_discount(components):
    count = components + 1
    data = dict(action=torch.zeros(count, dtype=torch.int64),
                reward=torch.tensor([0.] * components + [6.]),
                terminated=torch.zeros(count, dtype=torch.bool),
                truncated=torch.zeros(count, dtype=torch.bool),
                environment_step=torch.tensor([False] * components + [True]))
    values = torch.zeros(count, 2)
    values[-1, 1] = 10.
    _, metrics = _objective(nstep_gate(n=1, step_key="environment_step"))(
        objective_data=data, group_id=torch.zeros(count, dtype=torch.int64),
        predictions=values, backup_value_predictions=values, backup_action_predictions=values)
    backup = metrics["backup"]
    assert isinstance(backup, torch.Tensor)
    torch.testing.assert_close(backup[:-1], torch.full((components,), 11.))


@pytest.mark.parametrize("mode", ["next_action", "n_actions", "next_step"])
def test_sequential_notebook_collects_and_trains(mode, monkeypatch):
    import json
    from pathlib import Path

    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    make = gym.make
    applied_actions = []

    def make_recorded(*args, **kwargs):
        assert "max_episode_steps" not in kwargs
        env = make(*args, **kwargs)
        step = env.step

        def record_step(action):
            assert env.action_space.contains(action)
            applied_actions.append(action.copy())
            return step(action)

        monkeypatch.setattr(env, "step", record_step)
        return env

    monkeypatch.setattr(gym, "make", make_recorded)
    path = Path(__file__).resolve().parents[1] / "examples/20_train_sequential_actions_dqn.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    threads = torch.get_num_threads()
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if index == 1:
                source = source.replace('BACKUP_MODE = "next_step"', f'BACKUP_MODE = "{mode}"')
                source = source.replace('NUM_CYCLES = 2', 'NUM_CYCLES = 1')
                source = source.replace('TRAIN_STEPS = 2', 'TRAIN_STEPS = 1')
            exec(compile(source, f"{path}:cell {index}", "exec"), namespace)
        assert namespace["env"].spec.id == "Reacher-v5"
        assert namespace["OBSERVATION_DIM"] == 10
        episode_steps = gym.spec("Reacher-v5").max_episode_steps
        assert episode_steps is not None
        assert namespace["MAX_STEPS_PER_EPISODE"] == episode_steps
        assert len(applied_actions) == 2 * episode_steps
        history = namespace["last_history"]
        for index, applied in enumerate(applied_actions[-episode_steps:]):
            # One vector per physical step, assembled in the recorded component order.
            assert applied.tolist() == [history[2 * index + 1]["torque"],
                                        history[2 * index + 2]["torque"]]
    finally:
        if "env" in namespace:
            namespace["env"].close()
        torch.set_num_threads(threads)
