# Walk Failure Audit - Architecture Map

Date: 2026-08-15

Scope: static audit of the repository plus deterministic CPU diagnostics. No PPO tuning was used as evidence.

## Main Training Flow

- `train.py:513` runs training and delegates to Brax PPO at `train.py:694`.
- `train.py:1154` builds `BiomechanicsJoystickEnv` from `EnvConfig`.
- `config.py:611` defines the default environment and PPO settings.
- Current default reference path:
  - `config.py:622` sets `reference_gait="bvh"`.
  - `config.py:623` uses `BVH_walking_animation/tier1_debug_10.txt`.
  - `config.py:632` sets `reference_action_mode="mimickit"`.
  - `config.py:644` sets `deepmimic_root_velocity_weight_scale=0.15`.

## Environment Flow

- `biomechanics_env.py:676` configures BVH/SMPL reference clips.
- `biomechanics_env.py:1560` resets the environment and samples the reference clip/time.
- `biomechanics_env.py:1901` steps physics and advances reference time.
- `biomechanics_env.py:2096` maps policy actions to actuator position targets.
- `biomechanics_env.py:2292` builds observations.
- `biomechanics_env.py:2870` computes the DeepMimic imitation reward.
- `biomechanics_env.py:2729` terminates non-looping reference clips when motion time reaches clip duration.
- `biomechanics_env.py:3343` queries interpolated reference frames.
- `biomechanics_env.py:3536` creates exact or projected reset states from the reference.

## Reference Loaders

- `bvh_reference.py:290` retargets BVH clips to MuJoCo controlled DOFs.
- `bvh_reference.py:478` classifies clips as wrap or clamp.
- `bvh_reference.py:521` selects clips for each BVH source.
- `bvh_reference.py:1023` converts BVH root motion into MuJoCo root targets.
- `bvh_reference.py:1119` maps BVH Y-up centimeter positions into MuJoCo Z-up meters.
- `bvh_reference.py:1125` maps BVH rotations into MuJoCo coordinates.
- `smpl_reference.py` provides the SMPL path used by recent failed runs.

## Baseline Comparison Repositories

- `.tmp_mimickit/data/envs/deepmimic_humanoid_env.yaml` uses target observations, random reset, pose termination, and normalized reward weights.
- `.tmp_mimickit/mimickit/envs/deepmimic_env.py:181` resets the simulated body to the sampled reference root/joint state.
- `.tmp_mimickit/mimickit/envs/deepmimic_env.py:787` computes the DeepMimic reward.
- `.tmp_deepmimic/DeepMimicCore/anim/Motion.cpp:32` and `:486` implement reference phase and frame interpolation.
- `.tmp_deepmimic/DeepMimicCore/scenes/SceneImitate.cpp:7` computes the original DeepMimic imitation reward.

## Audit Verdict

The primary learning failure is not explained by PPO hyperparameters. The highest-impact failures are before PPO:

1. Recent SMPL reference playback/oracle evidence already fails to produce a valid survivable imitation episode.
2. The active reference libraries are dominated by short clamp clips, which create short episodes instead of sustained walking cycles.
3. The reward has a non-normalized weight modification, so an exact reference match reaches about `0.915` DeepMimic reward instead of `1.0`.

