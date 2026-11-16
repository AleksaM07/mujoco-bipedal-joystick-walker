#!/usr/bin/env bash
set -Eeuo pipefail

# Three gated phases, 40M + 15M + 20M = 75M environment steps.
# The first gate therefore always receives the requested minimum 40M budget.
# Run from any directory with: bash scripts/run_transition_curriculum_40m.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-16}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
if [[ -z "${BATCH_SIZE:-}" ]]; then
    if (( NUM_ENVS >= 512 && NUM_ENVS % NUM_MINIBATCHES == 0 )); then
        BATCH_SIZE=$((NUM_ENVS / NUM_MINIBATCHES))
    else
        BATCH_SIZE=128
    fi
fi
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
NATURAL_CHECKPOINT="${NATURAL_CHECKPOINT:-prezentacija/5th_success_final_natural_walk/model_000001024000}"
NATURAL_XML="${NATURAL_XML:-prezentacija/5th_success_final_natural_walk/generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml}"
TRANSITION_BVH="${TRANSITION_BVH:-BVH_walking_animation/cmuconvert-max-131-135/133/133_21.bvh}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found or not executable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -d "$NATURAL_CHECKPOINT" ]]; then
    echo "Natural-walk checkpoint not found: $NATURAL_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$NATURAL_XML" ]]; then
    echo "Archived natural-walk XML not found: $NATURAL_XML" >&2
    exit 2
fi
if [[ ! -f "$TRANSITION_BVH" ]]; then
    echo "Transition BVH not found: $TRANSITION_BVH" >&2
    exit 2
fi
if pgrep -af '[p]ython.*train.py' >/dev/null; then
    echo "Another train.py process is already active; refusing a duplicate." >&2
    pgrep -af '[p]ython.*train.py' >&2
    exit 3
fi
if (( BATCH_SIZE * NUM_MINIBATCHES % NUM_ENVS != 0 )); then
    echo "Invalid PPO geometry: BATCH_SIZE=$BATCH_SIZE, " \
         "NUM_MINIBATCHES=$NUM_MINIBATCHES, NUM_ENVS=$NUM_ENVS." >&2
    echo "BATCH_SIZE * NUM_MINIBATCHES must be divisible by NUM_ENVS." >&2
    exit 2
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

COMMON_ARGS=(
    --device gpu
    --env-version standard
    --playground-impl jax
    --physics-backend mjx_jax
    --xml-path "$NATURAL_XML"
    --command-profile walk
    --action-smoothing 0.2
    --reference-gait bvh
    --reference-gait-file "$TRANSITION_BVH"
    --reference-loop-mode clamp
    --reference-action-mode residual
    --reference-action-center default
    --reference-action-range reference_targets
    --reference-action-range-scale 1.1
    --reference-residual-scale 1.0
    --recovery-physics-version consistent_v6
    --reference-motion-speed-scale 1.0
    --reference-loop-closure bounded
    --reference-root-xy-scale 0.35
    --reference-lock-stance-feet
    --reference-foot-lock-height 0.035
    --reference-replay-target-step 0
    --bvh-target-observation-steps 0,1,2,3
    --arm-reference-mode cmu_elbow_wrist_ik
    --reference-elbow-flexion-alpha 1.0
    --reference-stability-arm-alpha 1.0
    --arms-on
    --deepmimic-root-velocity-weight-scale 1.0
    --deepmimic-arm-reward-weight 1.0
    --deepmimic-reward-mode mixed
    --bare
    --num-envs "$NUM_ENVS"
    --num-eval-envs "$NUM_EVAL_ENVS"
    --training-metrics-steps 250000
    --episode-length 500
    --unroll-length 10
    --batch-size "$BATCH_SIZE"
    --num-minibatches "$NUM_MINIBATCHES"
    --updates-per-batch 2
)

find_completed_run() {
    local tag="$1"
    local matches=()
    shopt -s nullglob
    matches=(runs/*_"$tag"_V1_steps-*)
    shopt -u nullglob
    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "Expected exactly one run for tag $tag, found ${#matches[@]}." >&2
        return 1
    fi
    if [[ "${matches[0]}" == *_running ]]; then
        echo "Run did not reach a terminal status: ${matches[0]}" >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

gate_checkpoint() {
    local run_dir="$1"
    "$PYTHON_BIN" tools/check_transition_training_gate.py "$run_dir" >&2
    "$PYTHON_BIN" tools/check_transition_training_gate.py \
        "$run_dir" --print-checkpoint
}

run_phase() {
    local tag="$1"
    local timesteps="$2"
    local evals="$3"
    local learning_rate="$4"
    local resume_from="$5"
    shift 5

    if compgen -G "runs/*_${tag}_V1_steps-*" >/dev/null; then
        echo "A run with tag $tag already exists; refusing a duplicate." >&2
        exit 4
    fi
    "$PYTHON_BIN" -u train.py \
        "${COMMON_ARGS[@]}" \
        --seed 11 \
        --timesteps "$timesteps" \
        --num-evals "$evals" \
        --learning-rate "$learning_rate" \
        --resume-from "$resume_from" \
        --no-restore-value-fn \
        --run-tag "$tag" \
        "$@"
}

PHASE_A_TAG="transition40m_a_stride_${RUN_SUFFIX}"
PHASE_B_TAG="transition40m_b_endpoints_${RUN_SUFFIX}"
PHASE_C_TAG="transition40m_c_full_${RUN_SUFFIX}"

echo "Phase A/3: 40M finite middle-stride balance adaptation"
echo "PPO geometry: envs=$NUM_ENVS eval_envs=$NUM_EVAL_ENVS batch=$BATCH_SIZE minibatches=$NUM_MINIBATCHES"
run_phase \
    "$PHASE_A_TAG" 40000000 21 0.00005 "$NATURAL_CHECKPOINT" \
    --reference-segment-mode locomotion \
    --reference-force-clip-id 0
PHASE_A_RUN="$(find_completed_run "$PHASE_A_TAG")"
PHASE_A_CHECKPOINT="$(gate_checkpoint "$PHASE_A_RUN" | tail -n 1)"

echo "Phase B/3: 15M start/stop endpoint curriculum"
run_phase \
    "$PHASE_B_TAG" 15000000 16 0.00005 "$PHASE_A_CHECKPOINT" \
    --reference-segment-mode transition_endpoints
PHASE_B_RUN="$(find_completed_run "$PHASE_B_TAG")"
PHASE_B_CHECKPOINT="$(gate_checkpoint "$PHASE_B_RUN" | tail -n 1)"

echo "Phase C/3: 20M complete start/walk/stop take"
run_phase \
    "$PHASE_C_TAG" 20000000 21 0.00002 "$PHASE_B_CHECKPOINT" \
    --reference-segment-mode full \
    --reference-force-clip-id 0
PHASE_C_RUN="$(find_completed_run "$PHASE_C_TAG")"
PHASE_C_CHECKPOINT="$(gate_checkpoint "$PHASE_C_RUN" | tail -n 1)"

echo "75M transition curriculum passed all gates."
echo "Final run: $PHASE_C_RUN"
echo "Selected checkpoint: $PHASE_C_CHECKPOINT"
