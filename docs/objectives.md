# Objectives

All four objective functions share the same call signature pattern:

```python
loss, metrics = xxx_objective(step_stream, model_output_tensor, cfg)
```

- `step_stream` — `TensorDict[B, S]` batch from `DataLoader`.
- model output tensor — sliced from `model(step_stream)`.
- `cfg` — frozen dataclass with hyperparameters.
- Returns `(scalar_loss, dict[str, float])` — the dict is ready for direct logging to W&B / TensorBoard.

---

## Transition alignment

A row at step `t` in your data normally contains the observation that resulted from the action/reward/done at that position (this is a convention of how the data was collected). The "next" quantities therefore live at `t+1`. The objectives consume the flat `step_stream` produced by the `DataLoader` and use `[:, :-1]` / `[:, 1:]` slicing to line up targets. The store itself does not enforce any particular alignment.

---

## DQN objective (`mouse_core.objectives.dqn`)

One-step TD objective with twin (online / target) Q-heads.

```python
from mouse_core.objectives.dqn import DqnObjectiveConfig, dqn_objective

cfg = DqnObjectiveConfig(
    weight=1.0,
    gamma=0.99,
    gamma_terminal=0.0,      # discount on max Q(s') at terminal steps
    gamma_truncated=0.0,     # discount on max Q(s') at truncated steps
    tau=0.005,               # Polyak rate (applied externally via model.polyak_update)
    normalize_reward_mean=False,
    normalize_reward_std=False,
    cql_weight=0.0,          # > 0 enables CQL penalty
    reward_scale=1.0,
    reward_shift=0.0,
    use_xformed_reward=False,
)

loss, metrics = dqn_objective(step_stream, out, cfg)
```

### TD target

```
td_target = r * reward_scale + reward_shift + discount * max_a Q_target(s', a)
```

where `discount` is:

```
discount = gamma * (1 − terminal − truncated)
         + gamma_terminal * terminal
         + gamma_truncated * truncated
```

Setting `gamma_terminal=0` and `gamma_truncated=0` zeroes the bootstrap term at all episode ends.

### CQL penalty

When `cql_weight > 0`, a conservative penalty is added:

```
cql_penalty = log Σ_a exp Q(s, a) − Q(s, a_executed)
```

The penalty is scaled by `|td_target| + cql_scale_q_eps` to keep its magnitude in proportion to the squared TD error as Q values grow.

### Metrics returned

`q_values_mean`, `q_values_std`, `q_values_min`, `q_values_max`, `q_values_target`, `dqn`, `cql_penalty` (if enabled).

---

## Vector DQN objective (`mouse_core.objectives.vec_dqn`)

Geometric objective for the `VecDQNHead`. Instead of scalar Q-values, each action is represented as a unit vector in `ℝ^D`. The objective trains the online action vector to point in the direction of a **reward-rotated** bootstrap target vector.

```python
from mouse_core.objectives.vec_dqn import VecDqnObjectiveConfig, vec_dqn_objective

cfg = VecDqnObjectiveConfig(
    weight=1.0,
    tau=0.005,
    reward_scale=1.0,   # rotation angle = reward * reward_scale + reward_shift
    reward_shift=0.0,
    normalize_reward_mean=False,
    normalize_reward_std=False,
    use_xformed_reward=False,
)

loss, metrics = vec_dqn_objective(
    step_stream,
    out["vec_dqn"],
    out["vec_dqn_target"],
    cfg,
)
```

### Algorithm

1. For the executed action at step `t`, take the online vector `v(s_t, a_t)`.
2. Find the greedy action at `s_{t+1}` using `vec_dqn_scores` on the target vectors.
3. Rotate the greedy target vector by `θ = reward * reward_scale + reward_shift` using RoPE: `v_rotated = rope_rotate(v_greedy, θ)`.
4. Minimise `1 − cosine_similarity(v(s_t, a_t), v_rotated.detach())`.

The rotation encodes the reward directly into the geometry of the representation — a higher-reward transition produces a larger angular displacement toward "better" actions.

### Metrics returned

`vec_dqn`, `vec_dqn_score_abs_min`, `vec_dqn_score_abs_max`, `vec_dqn_score_abs_mean`.

---

## Supervised policy objective (`mouse_core.objectives.sp`)

Distils `q_star` annotations into the `sp` head logits. Six variants are available.

```python
from mouse_core.objectives.sp import SpObjectiveConfig, sp_objective

cfg = SpObjectiveConfig(
    weight=1.0,
    loss_type="ce",          # see table below
    temperature=1.0,         # used for all soft variants
    label_smoothing=0.0,     # applied to teacher distribution only
)

loss, metrics = sp_objective(step_stream, out["sp"], cfg)
```

### Loss types

| `loss_type` | Description |
|---|---|
| `"ce"` | Hard cross-entropy — argmax of `q_star` as the label |
| `"ce-soft-fwd"` | `H(P_teacher, Q_student)` = −Σ P log Q |
| `"ce-soft-bwd"` | `H(Q_student, P_teacher)` = −Σ Q log P |
| `"js"` | Jensen–Shannon divergence: `0.5 KL(P‖M) + 0.5 KL(Q‖M)`, M = (P+Q)/2 |
| `"kl-fwd"` | `KL(P_teacher ‖ Q_student)` |
| `"kl-bwd"` | `KL(Q_student ‖ P_teacher)` |

All soft variants use `softmax(q_star / temperature)` as the teacher distribution. `q_star` values of `-inf` (invalid/padding actions) are treated as zero probability via `nan_to_num`.

### Metrics returned

`sp`.

---

## Supervised value objective (`mouse_core.objectives.sv`)

Directly regresses the `sv` head onto `q_star` values. Only finite entries in `q_star` participate; `-inf` padding never contributes gradients.

```python
from mouse_core.objectives.sv import SvObjectiveConfig, sv_objective

cfg = SvObjectiveConfig(
    weight=1.0,
    loss_type="mse",   # "mse" or "mae"
)

loss, metrics = sv_objective(step_stream, out["sv"], cfg)
```

### Metrics returned

`sv`.

---

## Combining objectives

Objective functions are designed to be composed freely (weight them and sum). A complete runnable training loop using multiple heads is in [`examples/02_train_offline.ipynb`](../examples/02_train_offline.ipynb).
