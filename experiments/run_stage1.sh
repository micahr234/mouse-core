#!/usr/bin/env bash
# Stage 1 sweep: diagnose the held-out generalization failure in online training.
#
# All runs share a reduced budget (seq 256, 5k grad steps, ~10k env steps) so a
# single run takes ~30 min on the RTX 3090 Ti. Axes:
#   asis    — notebook 03 behavior: no env rotation (only maps 0-9 ever trained on)
#   div50   — rotation fix, 50 distinct maps
#   div1000 — rotation fix, 1000 maps (each visited ~once: max diversity, min depth)
#   explore — rotation + per-env epsilon decay with 0.05 floor
#   gamma   — rotation + gamma_step=0.99 (incentive for short paths)
#   penalty — rotation + step_penalty=-0.02 (env-side step cost)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

COMMON=(
  --sequence-length 256
  --batch-size 4
  --gradient-steps 5000
  --gradient-steps-per-cycle 500
  --env-steps-per-cycle 1000
  --steps-per-env 100
  --learning-starts 1000
  --exploration-ends 6000
  --eval-maps 10
  --eval-steps 150
  --eval-fracs 0.5 1.0
)

run() {
  echo "===== STARTING $* ====="
  "$PY" experiments/online_experiment.py "${COMMON[@]}" "$@" \
    || echo "===== FAILED $* ====="
  echo "===== FINISHED $* ====="
}

run --name asis    --rotate-envs false --num-envs 1000
run --name div50   --rotate-envs true  --num-envs 50
run --name div1000 --rotate-envs true  --num-envs 1000
run --name explore --rotate-envs true  --num-envs 50 \
    --exploration-mode per_env --per-env-exploration-steps 150 --epsilon-floor 0.05
run --name gamma   --rotate-envs true  --num-envs 50 --gamma-step 0.99
run --name penalty --rotate-envs true  --num-envs 50 --step-penalty -0.02
echo "===== STAGE1 COMPLETE ====="
