# Losses

All loss functions return `(scalar_loss, dict[str, float])`. The dict is ready to log directly to W&B or TensorBoard.

## DQN loss

::: mouse.losses.dqn.DqnLossConfig

::: mouse.losses.dqn.dqn_loss

---

## Vector DQN loss

::: mouse.losses.vec_dqn.VecDqnLossConfig

::: mouse.losses.vec_dqn.vec_dqn_loss

---

## Supervised policy loss

::: mouse.losses.sp.SpLossConfig

::: mouse.losses.sp.sp_loss

::: mouse.losses.sp.sp_js_loss

::: mouse.losses.sp.sp_kl_loss

::: mouse.losses.sp.sp_soft_ce_loss

---

## Supervised value loss

::: mouse.losses.sv.SvLossConfig

::: mouse.losses.sv.sv_loss
