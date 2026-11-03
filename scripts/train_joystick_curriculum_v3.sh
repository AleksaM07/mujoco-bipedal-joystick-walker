#!/usr/bin/env bash
# Three automatically gated 10M blocks from the frozen second-success policy.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${HOME}/.cache/mujoco-bipedal-jax}"
mkdir -p artifacts
exec 9>artifacts/joystick_curriculum_v3.lock
flock -n 9 || { printf 'Joystick curriculum v3 is already running.\n' >&2; exit 1; }
printf 'Curriculum source commit: '
git rev-parse HEAD
exec .venv/bin/python -u scripts/train_joystick_curriculum.py \
  --num-envs "${JOYSTICK_NUM_ENVS:-64}" \
  --timesteps "${JOYSTICK_TIMESTEPS:-30000000}" \
  --speed-center "${JOYSTICK_SPEED_CENTER:-1.1}" \
  --campaign "artifacts/${JOYSTICK_CAMPAIGN:-joystick_v3_$(date +%Y%m%d_%H%M%S)}"
