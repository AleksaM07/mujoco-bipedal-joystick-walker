#!/usr/bin/env bash
set -Eeuo pipefail

# Train only the learned bridge:
# FINAL_WALKING -> BRAKE_TRANSITION -> FINAL_STANDING.
# The two final policy archives are read-only inputs and are never overwritten.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-32}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
BATCH_SIZE="${BATCH_SIZE:-$((NUM_ENVS / NUM_MINIBATCHES))}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="brake_transition_v2_40m_${RUN_SUFFIX}"

FINAL_WALKING="prezentacija/5th_success_final_natural_walk/model_000001024000"
FINAL_STANDING="runs/stand_horizon60_20261106_1m_lr3e-4/resume/20261106_1414_stand_balance_v1_resume_1000000_lr0_0003_V1_steps-1m_reward-8554p6201_s/checkpoints/000001024000"
FINAL_XML="prezentacija/5th_success_final_natural_walk/generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml"
FINAL_WALK_BVH="BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh"
FINAL_WALK_STATE_BANK="skills/assets/final_walking_state_bank.npz"

for required in \
    "$PYTHON_BIN" \
    "$FINAL_XML" \
    "$FINAL_WALK_BVH" \
    "$FINAL_WALK_STATE_BANK"; do
    if [[ ! -e "$required" ]]; then
        echo "Required path is missing: $required" >&2
        exit 2
    fi
done
for checkpoint in "$FINAL_WALKING" "$FINAL_STANDING"; do
    if [[ ! -d "$checkpoint" ]]; then
        echo "Final policy checkpoint is missing: $checkpoint" >&2
        exit 2
    fi
done
if pgrep -af '[p]ython.*train.py' >/dev/null; then
    echo "Another train.py process is active; refusing a duplicate." >&2
    pgrep -af '[p]ython.*train.py' >&2
    exit 3
fi
if compgen -G "runs/*_${RUN_TAG}_V1_steps-*" >/dev/null; then
    echo "A run with tag $RUN_TAG already exists; refusing a duplicate." >&2
    exit 4
fi
if (( BATCH_SIZE * NUM_MINIBATCHES % NUM_ENVS != 0 )); then
    echo "Invalid PPO geometry: batch * minibatches must divide envs." >&2
    exit 2
fi

"$PYTHON_BIN" -c \
    'from skills.policy_registry import validate_final_policy_pair; validate_final_policy_pair()'

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
    echo "BRAKE_TRANSITION 40M precheck passed."
    exit 0
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "Training BRAKE_TRANSITION v2 for 40M steps from FINAL_WALKING states"
echo "PPO geometry: envs=$NUM_ENVS eval_envs=$NUM_EVAL_ENVS batch=$BATCH_SIZE minibatches=$NUM_MINIBATCHES"
"$PYTHON_BIN" -u train.py \
    --device gpu \
    --env-version standard \
    --playground-impl jax \
    --physics-backend mjx_jax \
    --xml-path "$FINAL_XML" \
    --locomotion-skill brake_transition \
    --transition-blend-seconds 0.8 \
    --transition-min-walk-seconds 0.0 \
    --transition-stop-phase 0.5263157894736842 \
    --brake-lead-seconds-min 0.2 \
    --brake-lead-seconds-max 0.8 \
    --brake-state-bank-file "$FINAL_WALK_STATE_BANK" \
    --command-profile walk \
    --action-smoothing 0.2 \
    --reference-gait bvh \
    --reference-gait-file "$FINAL_WALK_BVH" \
    --reference-clip-ids 1 \
    --reference-force-clip-id 0 \
    --reference-segment-mode locomotion \
    --reference-loop-mode wrap \
    --reference-action-mode residual \
    --reference-action-center default \
    --reference-action-range reference_targets \
    --reference-action-range-scale 1.1 \
    --reference-residual-scale 1.0 \
    --recovery-physics-version consistent_v6 \
    --reference-motion-speed-scale 1.0 \
    --reference-loop-closure bounded \
    --reference-root-xy-scale 0.35 \
    --reference-lock-stance-feet \
    --reference-foot-lock-height 0.035 \
    --reference-replay-target-step 0 \
    --bvh-target-observation-steps 0,1,2,3 \
    --arm-reference-mode cmu_elbow_wrist_ik \
    --reference-elbow-flexion-alpha 1.0 \
    --reference-stability-arm-alpha 1.0 \
    --arms-on \
    --deepmimic-root-velocity-weight-scale 1.0 \
    --deepmimic-arm-reward-weight 1.0 \
    --deepmimic-reward-mode pure \
    --bare \
    --seed 11 \
    --timesteps 40000000 \
    --num-evals 21 \
    --learning-rate 0.00002 \
    --num-envs "$NUM_ENVS" \
    --num-eval-envs "$NUM_EVAL_ENVS" \
    --training-metrics-steps 250000 \
    --episode-length 200 \
    --unroll-length 10 \
    --batch-size "$BATCH_SIZE" \
    --num-minibatches "$NUM_MINIBATCHES" \
    --updates-per-batch 2 \
    --resume-from "$FINAL_WALKING" \
    --no-restore-value-fn \
    --run-tag "$RUN_TAG"

shopt -s nullglob
matches=(runs/*_"$RUN_TAG"_V1_steps-*)
shopt -u nullglob
if [[ "${#matches[@]}" -ne 1 || "${matches[0]}" == *_running ]]; then
    echo "Could not identify one completed BRAKE_TRANSITION run." >&2
    exit 5
fi

echo "Training completed: ${matches[0]}"
"$PYTHON_BIN" tools/check_brake_training_gate.py "${matches[0]}"
SELECTED_CHECKPOINT="$(
    "$PYTHON_BIN" tools/check_brake_training_gate.py \
        "${matches[0]}" --print-checkpoint
)"
echo "Selected checkpoint: $SELECTED_CHECKPOINT"

PROBE_OUT="runs/brake_handoff_gate_${RUN_SUFFIX}"
"$PYTHON_BIN" tools/evaluate_skill_transitions.py \
    --protocol brake_handoff \
    --device gpu \
    --seed 11 \
    --brake-checkpoint "$SELECTED_CHECKPOINT" \
    --brake-phase-candidates 64 \
    --output-dir "$PROBE_OUT" \
    --video

echo "Connected three-policy gate passed: $PROBE_OUT"
