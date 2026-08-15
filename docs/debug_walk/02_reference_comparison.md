# Reference Comparison

## DeepMimic Timing

Reference:

- `.tmp_deepmimic/DeepMimicCore/anim/Motion.cpp:32` computes phase as time over duration, with wrap or clamp semantics.
- `.tmp_deepmimic/DeepMimicCore/anim/Motion.cpp:486` maps time to frame index and blend.
- `.tmp_deepmimic/DeepMimicCore/anim/Motion.cpp:529` declares non-looping motions over at duration.

Local code:

- `biomechanics_env.py:3343` uses `motion_time = offset + motion_step * dt`.
- `biomechanics_env.py:1901` advances physics once per control step.
- `biomechanics_env.py:1935` increments `motion_step` by exactly one per control step.

Verdict:

- I did not find evidence for a 4x phase-speed bug. `dt=0.02` and `n_substeps=4` mean four `0.005s` physics substeps per one `0.02s` policy/reference step, which is correct.

## DeepMimic/MimicKit Reward

Reference:

- `.tmp_deepmimic/DeepMimicCore/scenes/SceneImitate.cpp:7` uses normalized reward weights.
- `.tmp_mimickit/data/envs/deepmimic_humanoid_env.yaml:33` and `:34` set root pose and root velocity weights.
- `.tmp_mimickit/mimickit/envs/deepmimic_env.py:787` computes pose, velocity, root pose, root velocity, and key-position reward.

Local code:

- `biomechanics_env.py:2870` computes the same general reward components.
- `biomechanics_env.py:2970` multiplies root velocity weight by `deepmimic_root_velocity_weight_scale`.
- `config.py:644` sets that scale to `0.15`.

Verdict:

- The local reward is structurally DeepMimic-like, but the exact-match reward is no longer normalized to `1.0`. This is a real invariant break.

## MimicKit Action Control

Reference:

- `.tmp_mimickit/mimickit/envs/char_env.py` uses position target bounds and sends clipped actions as joint targets.

Local code:

- `biomechanics_env.py:2096` maps `mimickit` actions to actuator center plus range.
- The action oracle found BVH reference qpos values are representable by this mapping, with max absolute normalized action around `1.0`.

Verdict:

- The action mapping is not the primary suspected failure for the default BVH path. Some targets sit exactly at bounds, so clipping margins should still be monitored.

