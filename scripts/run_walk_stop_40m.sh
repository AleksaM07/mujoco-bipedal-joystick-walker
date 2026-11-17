#!/usr/bin/env bash
set -Eeuo pipefail

# First physically gated walk -> brake -> stand experiment.  This intentionally
# runs one 40M phase only; a failed stop must not trigger another long run.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-32}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
BATCH_SIZE="${BATCH_SIZE:-$((NUM_ENVS / NUM_MINIBATCHES))}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
NATURAL_CHECKPOINT="${NATURAL_CHECKPOINT:-prezentacija/5th_success_final_natural_walk/model_000001024000}"
NATURAL_XML="${NATURAL_XML:-prezentacija/5th_success_final_natural_walk/generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml}"
NATURAL_BVH="${NATURAL_BVH:-BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh}"

for required in "$PYTHON_BIN" "$NATURAL_CHECKPOINT" "$NATURAL_XML" "$NATURAL_BVH"; do
    if [[ ! -e "$required" ]]; then
        echo "Required input missing: $required" >&2
        exit 2
    fi
done
if pgrep -af '[p]ython.*train.py' >/dev/null; then
    echo "Another train.py process is active; refusing a duplicate." >&2
    pgrep -af '[p]ython.*train.py' >&2
    exit 3
fi
if (( BATCH_SIZE * NUM_MINIBATCHES % NUM_ENVS != 0 )); then
    echo "Invalid PPO geometry: batch * minibatches must divide num envs." >&2
    exit 2
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "40M walk -> brake -> stand curriculum on proven 02_01 gait"
echo "PPO geometry: envs=$NUM_ENVS eval_envs=$NUM_EVAL_ENVS batch=$BATCH_SIZE minibatches=$NUM_MINIBATCHES"
"$PYTHON_BIN" -u train.py \
    --device gpu \
    --env-version standard \
    --playground-impl jax \
    --physics-backend mjx_jax \
    --xml-path "$NATURAL_XML" \
    --locomotion-skill walk_stop \
    --transition-min-walk-seconds 2.0 \
    --transition-blend-seconds 0.8 \
    --transition-stop-phase 0.5263157894736842 \
    --command-profile walk \
    --action-smoothing 0.2 \
    --reference-gait bvh \
    --reference-gait-file "$NATURAL_BVH" \
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
    --num-envs "$NUM_ENVS" \
    --num-eval-envs "$NUM_EVAL_ENVS" \
    --training-metrics-steps 250000 \
    --episode-length 500 \
    --unroll-length 10 \
    --batch-size "$BATCH_SIZE" \
    --num-minibatches "$NUM_MINIBATCHES" \
    --updates-per-batch 2 \
    --seed 11 \
    --timesteps 40000000 \
    --num-evals 21 \
    --learning-rate 0.00005 \
    --resume-from "$NATURAL_CHECKPOINT" \
    --no-restore-value-fn \
    --run-tag "walk_stop_40m_${RUN_SUFFIX}"

echo "Training finished. Do not start another phase before tools/evaluate_walk_stop.py passes."
