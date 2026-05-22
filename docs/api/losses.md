# Losses

All loss functions return `(scalar_loss, dict[str, float])` for logging. Implementations live under [`src/losses/`](../../src/losses/).

## DQN

| Name | Source |
|------|--------|
| [`DqnLossConfig`](../../src/losses/dqn.py) | TD loss hyperparameters |
| [`dqn_loss`](../../src/losses/dqn.py) | One-step two-head DQN TD loss |

## Vector DQN

| Name | Source |
|------|--------|
| [`VecDqnLossConfig`](../../src/losses/vec_dqn.py) | Vector DQN config |
| [`vec_dqn_loss`](../../src/losses/vec_dqn.py) | Vector DQN TD loss |

## Supervised policy

| Name | Source |
|------|--------|
| [`SpLossConfig`](../../src/losses/sp.py) | SP loss config |
| [`sp_loss`](../../src/losses/sp.py) | Dispatching SP loss |
| [`sp_js_loss`](../../src/losses/sp.py) | Jensen–Shannon variant |
| [`sp_kl_loss`](../../src/losses/sp.py) | KL variant |
| [`sp_soft_ce_loss`](../../src/losses/sp.py) | Soft cross-entropy variant |

## Supervised value

| Name | Source |
|------|--------|
| [`SvLossConfig`](../../src/losses/sv.py) | SV loss config |
| [`sv_loss`](../../src/losses/sv.py) | Q-star regression loss |

Public API: [`mouse.losses`](../../src/losses/__init__.py).
