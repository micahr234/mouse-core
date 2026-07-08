#!/usr/bin/env bash
# Stage 4: full-budget validation of the fixed online recipe.
#   full_online       — 20k grad steps @ seq 512 (notebook budget), 500 maps x 200
#                       steps with per-env exploration progression, unshaped
#                       rewards (no step penalty; 50-step truncation provides
#                       the progress pressure), id-permutation augmentation.
#                       Honest online run: behavior comes from the model +
#                       epsilon only.
#   full_online_noaug — control: same but without augmentation.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

COMMON=(
  --sequence-length 512
  --batch-size 4
  --gradient-steps 20000
  --gradient-steps-per-cycle 1000
  --learning-starts 2000
  --eval-maps 10
  --eval-steps 512
  --eval-fracs 0.25 0.5 0.75 1.0
  --rotate-envs true
  --num-envs 500
  --env-steps-per-cycle 5000
  --steps-per-env 200
  --exploration-mode per_env
  --per-env-exploration-steps 150
  --epsilon-floor 0.05
)

run() {
  echo "===== STARTING $* ====="
  "$PY" experiments/online_experiment.py "${COMMON[@]}" "$@" \
    || echo "===== FAILED $* ====="
  echo "===== FINISHED $* ====="
}

run --name full_online --augment true
run --name full_online_noaug --augment false
echo "===== STAGE4 COMPLETE ====="
