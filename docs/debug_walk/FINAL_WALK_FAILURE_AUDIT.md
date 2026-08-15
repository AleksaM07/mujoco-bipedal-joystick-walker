# Final Walk Failure Audit

Date: 2026-08-15

## Executive Verdict

The humanoid is not failing to walk because PPO needs tuning. The strongest evidence points to pre-PPO invariants:

1. Recent SMPL reference playback already fails at the oracle layer.
2. The active motion libraries are dominated by short clamp clips, not stable cyclic walking references.
3. The DeepMimic reward is not normalized after root-velocity weight scaling, so exact imitation tops out around `0.915` instead of `1.0`.

## Critical Finding 1 - Reference Tracking Fails Before Learning

Severity: critical

Evidence:

- Existing `runs/20260815_165330_reference_playback_mjx_jax/playback_summary.json` reported `valid=0/4`.
- Failures were split between physical low-height failure and natural `motion_over`.
- Recent training logs showed short episodes around 38 to 45 steps, `done=1.0`, and frequent low-height, motion-over, and pose-termination components.

Effect:

- A policy cannot reliably learn a walking imitation if the reference target itself cannot be tracked by the simulator/controller stack.

Minimal fix:

- Stop PPO experiments until one clip passes PD reference tracking.
- Use:

```powershell
python tools\debug_reference_pd_tracking.py --reference-gait smpl --clip-id 0 --steps 180
```

Pass condition:

- The exact-reference PD target survives the intended clip without low-height or tipping failure.

## Critical Finding 2 - The Default Motion Set Is Not A Clean Walking Task

Severity: high

Evidence:

- Default BVH list: `config.py:133`.
- Loop/clamp classification: `bvh_reference.py:478`.
- Motion-over termination: `biomechanics_env.py:2729`.
- Default BVH inspection found 18 clips: 16 clamp, 2 wrap.
- Most clips are around 0.5 to 0.6 seconds, and the two wrap clips are only around 0.167 and 0.100 seconds.

Effect:

- Random reset plus short clamp clips creates short terminal fragments instead of a sustained gait objective.
- This matches observed short episode lengths and frequent motion-over termination.

Minimal fix:

- Build a one-clip walking debug preset with a visually verified cyclic reference.
- Do not train on the full mixed reference library until the one-clip oracle passes.

## Finding 3 - Reward Normalization Is Broken

Severity: medium-high

Evidence:

- `config.py:644` sets `deepmimic_root_velocity_weight_scale=0.15`.
- `biomechanics_env.py:2970` applies that scale to the root velocity weight without renormalizing.
- `tools/debug_reward_landscape.py` showed exact BVH reference reward near `0.914938`.

Effect:

- Exact imitation is not rewarded as a normalized perfect match.
- Debugging and thresholds become misleading.

Minimal fix:

- Normalize active weights after scaling, or set the root velocity weight scale back to `1.0` for invariant testing.

## Finding 4 - Phase Timing Is Probably Not The Main Bug

Severity: low for suspected phase-speed issue

Evidence:

- `biomechanics_env.py:1935` increments `motion_step` once per environment step.
- `biomechanics_env.py:3335` multiplies by `dt=0.02`.
- `n_substeps=4` only affects physics integration at `0.005s`.

Effect:

- The reference does not appear to advance four times too fast.

## Finding 5 - Coordinate Frames Mostly Pass BVH Exact-State Checks

Severity: medium

Evidence:

- Exact BVH reset gives near-zero pose and key errors.
- Reward ordering degrades monotonically under 1 degree, 5 degree, 20 degree, and random perturbations.

Remaining issue:

- Root angular velocity can mismatch exact reset because reward compares anchor-body velocity while reset stores free-root reference velocity.

## Recommended Debug Order

1. Pick one reference clip.
2. Run `debug_reference_playback.py` in kinematic mode.
3. Run `debug_reference_pd_tracking.py` in PD mode.
4. Fix reference, retargeting, root scaling, morphology, or PD gains until PD tracking survives.
5. Fix reward normalization so exact reference is `1.0`.
6. Only then run a tiny PPO smoke experiment.

## New Diagnostic Tools

- `tools/debug_reference_playback.py`
- `tools/debug_reference_pd_tracking.py`
- `tools/debug_reward_landscape.py`

## Verification Completed

```powershell
python -m py_compile tools\debug_reward_landscape.py tools\debug_reference_playback.py tools\debug_reference_pd_tracking.py
python tools\debug_reference_playback.py --reference-gait bvh --mode kinematic --clip-id 0 --steps 5
```

Playback smoke output confirmed zero pose RMSE for kinematic BVH clip 0 over the sampled steps and no immediate low-height or tipping failure.

## Repair Pass Applied

Date: 2026-08-15

Implemented fixes:

- Normalized the DeepMimic weighted reward after `deepmimic_root_velocity_weight_scale`, restoring exact-reference reward to approximately `1.0`.
- Fixed queried root angular velocity to use precomputed anchor-body FK velocity targets instead of free-root quaternion interval velocity.
- Changed phase-1 default action mode to `residual`, so zero action tracks the reference target instead of commanding a neutral standing pose.
- Restored MimicKit-style future target observations `(1, 2, 3)`.
- Filtered default reference clips below `0.8s`, reducing the default BVH debug task from 18 clips to 5 longer clips.
- Added a clamp reset margin of 25 control steps so random resets do not start at the terminal edge of finite clips.
- Marked clamp `motion_over` as truncation/natural end instead of physical termination.
- Applied configurable root-XY damping and stability projection to BVH references, not only SMPL references.
- Bumped generated XML version to `trainfast_v18` and strengthened position actuators so reference tracking has enough authority.
- Updated PD/playback diagnostics to command next-frame targets for PD mode.

Verification after repair:

```powershell
python -m py_compile biomechanics_env.py biomechanics_model.py config.py tools\debug_reward_landscape.py tools\debug_reference_playback.py tools\debug_reference_pd_tracking.py
python tools\debug_reward_landscape.py --reference-gait bvh --clip-id 0 --phase 0.0 --assert-invariants
python tools\debug_reference_pd_tracking.py --reference-gait bvh --clip-id 0 --steps 100 --print-every 10
```

Observed:

- Exact-reference reward: `0.999999`.
- Reward ordering: exact > 1 degree > 5 degree > 20 degree > random.
- Default BVH filtered clip count: 5.
- BVH clip 0 PD tracking reached natural `motion_over` at `1.66s` without low-height or tipped failure.
