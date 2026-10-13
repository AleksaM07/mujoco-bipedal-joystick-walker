#!/usr/bin/env bash
# Fresh 80M learned baseline. No experimental reward/reference monkeypatches.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p artifacts
exec 9>artifacts/recovery_v6_tracker.lock
if ! flock -n 9; then
  printf 'This tracker launcher is already running.\n' >&2
  exit 1
fi

printf 'Recovery v6 tracker launch; UTC: '
date -u --iso-8601=seconds
printf 'Git HEAD: '
git rev-parse HEAD
git status --short
sha256sum train.py biomechanics_env.py config.py bvh_reference.py \
  reference_interpolation.py reference_ik.py reference_root.py \
  generated_models/human_male_180cm_75kg_standard_trainfast_v21_noarms.xml \
  BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh

# 64 worlds / 128 batch fit the local WSL 8 GB RAM limit. The first attempt
# with 256/512 was killed by the kernel OOM killer before its first PPO update.
# 41 evaluations = initial plus about every 2M steps, with saved checkpoints.
# The preset otherwise retains residual1/current-frame/pure/noDR/noERFI.
exec .venv/bin/python -u train.py \
  --device gpu \
  --one-clip-overfit \
  --timesteps 80000000 \
  --num-envs 64 \
  --num-eval-envs 4 \
  --batch-size 128 \
  --num-evals 41 \
  --episode-length 500 \
  --seed 7 \
  --reference-gait-file BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh \
  --reference-clip-ids 1 \
  --reference-force-clip-id 0 \
  --recovery-physics-version consistent_v6 \
  --reference-loop-closure bounded \
  --xml-path generated_models/human_male_180cm_75kg_standard_trainfast_v21_noarms.xml \
  --run-tag recovery_v6_pure_tracker_mean_80m
