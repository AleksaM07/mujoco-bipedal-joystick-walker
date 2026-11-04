#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${HOME}/.cache/mujoco-bipedal-jax}"
mkdir -p artifacts
exec 9>artifacts/arm_ik_transfer.lock
flock -n 9 || { printf 'Arm IK transfer is already running.\n' >&2; exit 1; }
printf 'Arm IK source commit: '
git rev-parse HEAD
exec .venv/bin/python -u scripts/train_arm_ik_transfer.py \
  --num-envs "${ARM_IK_NUM_ENVS:-64}" \
  --timesteps "${ARM_IK_TIMESTEPS:-10000000}" \
  --arm-reward-weight "${ARM_IK_REWARD_WEIGHT:-1.0}" \
  --run-tag "${ARM_IK_RUN_TAG:-arms_ik_v2_single_clip_10m}"
