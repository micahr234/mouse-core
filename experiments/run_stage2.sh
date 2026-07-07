#!/usr/bin/env bash
# Stage 2 sweep: what the offline recipe has that online lacks.
#   aug50       — stage-1 div50 + id-permutation augmentation (isolates augmentation)
#   aug_rich    — augmentation + per-env epsilon progression + 100 maps x 200 steps
#   oracle_rich — augmentation + random->Q* oracle progression (online analogue of
#                 notebook 01 data; upper bound on behavior-data quality)
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
  --augment true
)

run() {
  echo "===== STARTING $* ====="
  "$PY" experiments/online_experiment.py "${COMMON[@]}" "$@" \
    || echo "===== FAILED $* ====="
  echo "===== FINISHED $* ====="
}

run --name aug50 --num-envs 50 --env-steps-per-cycle 1000 --steps-per-env 100 \
    --exploration-ends 6000
run --name aug_rich --num-envs 100 --env-steps-per-cycle 2000 --steps-per-env 200 \
    --exploration-mode per_env --per-env-exploration-steps 150 --epsilon-floor 0.05
run --name oracle_rich --num-envs 100 --env-steps-per-cycle 2000 --steps-per-env 200 \
    --exploration-mode oracle --per-env-exploration-steps 150 --epsilon-floor 0.05
echo "===== STAGE2 COMPLETE ====="
