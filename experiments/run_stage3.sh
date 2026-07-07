#!/usr/bin/env bash
# Stage 3: step_penalty unlocks a real Q* expert (without it Q* ties at 1.0 and
# argmax degenerates to "always left"). Test AD-style online data with the
# fixed expert, with and without augmentation, and a no-oracle counterpart.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

COMMON=(
  --sequence-length 256
  --batch-size 4
  --gradient-steps 5000
  --gradient-steps-per-cycle 500
  --learning-starts 1000
  --eval-maps 10
  --eval-steps 150
  --eval-fracs 0.5 1.0
  --rotate-envs true
  --num-envs 100
  --env-steps-per-cycle 2000
  --steps-per-env 200
  --per-env-exploration-steps 150
  --epsilon-floor 0.05
  --step-penalty -0.01
)

run() {
  echo "===== STARTING $* ====="
  "$PY" experiments/online_experiment.py "${COMMON[@]}" "$@" \
    || echo "===== FAILED $* ====="
  echo "===== FINISHED $* ====="
}

run --name oracle_pen       --exploration-mode oracle  --augment true
run --name oracle_pen_noaug --exploration-mode oracle  --augment false
run --name model_pen        --exploration-mode per_env --augment true
echo "===== STAGE3 COMPLETE ====="
