#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if pgrep -af "python.*train_arm_ik_transfer.py" >/dev/null; then
  echo "An arm IK transfer process is already running." >&2
  pgrep -af "python.*train_arm_ik_transfer.py" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
launch_log="logs/arms_ik_v3_${timestamp}.log"
run_tag="arms_ik_v3_elbow_wrist_${timestamp}"
mkdir -p logs

nohup env \
  PYTHONUNBUFFERED=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -u scripts/train_arm_ik_transfer.py \
  --num-envs 64 \
  --timesteps 10000000 \
  --arm-reward-weight 1.0 \
  --run-tag "$run_tag" \
  >"$launch_log" 2>&1 </dev/null &

pid=$!
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Training exited during startup. See: $launch_log" >&2
  tail -n 50 "$launch_log" >&2
  exit 1
fi

printf 'PID=%s\nLOG=%s\nRUN_TAG=%s\n' "$pid" "$launch_log" "$run_tag"
