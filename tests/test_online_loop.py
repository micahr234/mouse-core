"""Tests for online training loop semantics (examples/03_train_online.ipynb)."""
from __future__ import annotations

def simulate_online_loop(*, gradient_steps: int, env_steps_per_cycle: int, steps_per_env: int, gradient_steps_per_cycle: int, learning_starts: int, num_envs: int=10000) -> dict[str, int]:
    """Pure-Python model of the notebook master loop + rollout budget."""
    env_steps = 0
    grad_steps = 0
    cycles = 0
    train_bursts = 0
    envs_visited = 0
    while grad_steps < gradient_steps:
        cycles += 1
        collected = 0
        for _env_idx in range(num_envs):
            if collected >= env_steps_per_cycle:
                break
            chunk = min(steps_per_env, env_steps_per_cycle - collected)
            env_steps += chunk
            collected += chunk
            envs_visited += 1
        if env_steps >= learning_starts:
            burst = min(gradient_steps_per_cycle, gradient_steps - grad_steps)
            grad_steps += burst
            train_bursts += 1
    return {'cycles': cycles, 'train_bursts': train_bursts, 'grad_steps': grad_steps, 'env_steps': env_steps, 'envs_visited': envs_visited}

def test_notebook_defaults_reach_full_gradient_budget() -> None:
    stats = simulate_online_loop(gradient_steps=20000, env_steps_per_cycle=1000, steps_per_env=100, gradient_steps_per_cycle=1000, learning_starts=2000)
    assert stats['grad_steps'] == 20000
    assert stats['train_bursts'] == 20
    assert stats['env_steps'] == 21000
    assert stats['envs_visited'] == 210

def test_final_train_burst_is_partial() -> None:
    stats = simulate_online_loop(gradient_steps=2500, env_steps_per_cycle=200, steps_per_env=20, gradient_steps_per_cycle=1000, learning_starts=0)
    assert stats['grad_steps'] == 2500
    assert stats['train_bursts'] == 3

def test_learning_starts_delays_first_train_burst() -> None:
    stats = simulate_online_loop(gradient_steps=1000, env_steps_per_cycle=500, steps_per_env=100, gradient_steps_per_cycle=500, learning_starts=1000, num_envs=100)
    assert stats['grad_steps'] == 1000
    assert stats['train_bursts'] == 2
    assert stats['env_steps'] == 1500
    assert stats['cycles'] == 3

def test_epsilon_decay() -> None:

    def epsilon(*, env_step: int, exploration_ends: int) -> float:
        if exploration_ends <= 0:
            raise ValueError('exploration_ends must be positive.')
        frac = min(env_step / exploration_ends, 1.0)
        return 1.0 - frac
    assert epsilon(exploration_ends=10000, env_step=0) == 1.0
    assert epsilon(exploration_ends=10000, env_step=5000) == 0.5
    assert epsilon(exploration_ends=10000, env_step=10000) == 0.0
    assert epsilon(exploration_ends=10000, env_step=20000) == 0.0
