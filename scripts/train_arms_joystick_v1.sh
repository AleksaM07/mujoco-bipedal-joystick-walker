#!/usr/bin/env bash
# 30M curriculum: verified walking policy + active BVH arms + easy full joystick.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/mujoco-bipedal-jax}"

mkdir -p artifacts
exec 9>artifacts/arms_joystick_v1.lock
if ! flock -n 9; then
  printf 'The arms/joystick launcher is already running.\n' >&2
  exit 1
fi

warmstart_root="artifacts/arms_joystick_v1_warmstart"
warmstart_checkpoint="$warmstart_root/checkpoints/000000000000"
if [[ ! -d "$warmstart_checkpoint" ]]; then
  .venv/bin/python tools/migrate_checkpoint_to_arms.py \
    --output-root "$warmstart_root"
fi

printf 'Arms + joystick v1 launch; UTC: '
date -u --iso-8601=seconds
printf 'Git HEAD: '
git rev-parse HEAD
git status --short
sha256sum \
  train.py \
  biomechanics_env.py \
  config.py \
  tools/migrate_checkpoint_to_arms.py \
  generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml \
  BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh

# We deliberately do not use --one-clip-overfit: that forensic preset disables
# arms and joystick. This run retains the proven clip and physics, but changes
# reward to mixed and samples gentle commands in every direction.
exec .venv/bin/python -u train.py \
  --device gpu \
  --playground-impl jax \
  --physics-backend mjx_jax \
  --timesteps "${ARMS_JOYSTICK_TIMESTEPS:-30000000}" \
  --num-envs "${ARMS_JOYSTICK_NUM_ENVS:-64}" \
  --num-eval-envs 4 \
  --batch-size 128 \
  --unroll-length 10 \
  --num-minibatches 4 \
  --updates-per-batch 2 \
  --num-evals "${ARMS_JOYSTICK_NUM_EVALS:-31}" \
  --training-metrics-steps "${ARMS_JOYSTICK_METRICS_STEPS:-50000}" \
  --episode-length 500 \
  --learning-rate 0.00005 \
  --seed 7 \
  --command-profile standard_easy \
  --arms-on \
  --reference-gait bvh \
  --reference-action-mode residual \
  --reference-residual-scale 1.0 \
  --reference-gait-file BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh \
  --reference-clip-ids 1 \
  --reference-force-clip-id 0 \
  --reference-loop-mode wrap \
  --reference-root-xy-scale 0.35 \
  --reference-lock-stance-feet \
  --bvh-target-observation-steps 0,1,2,3 \
  --recovery-physics-version consistent_v6 \
  --reference-loop-closure bounded \
  --deepmimic-reward-mode mixed \
  --deepmimic-key-bodies metatarsal_midpoint_right,metatarsal_midpoint_left \
  --action-smoothing 0.2 \
  --bare \
  --xml-path generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml \
  --resume-from "$warmstart_checkpoint" \
  --run-tag "${ARMS_JOYSTICK_RUN_TAG:-arms_joystick_v1_30m}"
