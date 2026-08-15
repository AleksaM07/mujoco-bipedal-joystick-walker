# Reference Motion Audit

## Finding 1 - Default BVH Library Is Mostly Clamp Clips

Severity: high

Evidence:

- Default list: `config.py:133` points to `BVH_walking_animation/tier1_debug_10.txt`.
- Clip loop classification is decided in `bvh_reference.py:478`.
- Clamp clips terminate in `biomechanics_env.py:2729`.
- Deterministic library inspection found 18 default BVH clips. Loop modes were 16 clamp clips and 2 wrap clips.
- Clip durations were approximately:
  - `1.65, 0.558, 1.442, 0.517, 0.600, 0.550, 0.167, 0.500, 0.517, 1.600, 0.508, 0.100, 1.725, 0.542, 0.583, 1.758, 0.583, 0.633` seconds.
- The only wrap clips were very short: about `0.167s` and `0.100s`.

Semantic difference from DeepMimic/MimicKit:

- DeepMimic can train finite motions, but locomotion normally needs a vetted cyclic walk motion or a long enough clip with appropriate end handling.
- MimicKit samples random motion times, but its reference library is expected to be a valid motion set for the task. A set of mostly sub-second clamp fragments is not a sustained walking curriculum.

Training effect:

- Random reset can start near the end of a clamp clip.
- Many episodes terminate from `motion_over` within a few dozen control steps.
- PPO receives short rollouts where "episode ended" often means "clip ended", not "walking succeeded".
- The policy can learn to survive resets or exploit terminal structure without learning a durable gait.

Minimal fix:

- Create a debug preset with one visually verified cyclic walking clip in `WRAP` mode.
- For clamp clips, either sample reset times away from the end or treat motion end as a success/time-limit condition with correct bootstrapping.
- Do not mix many short clamp snippets into the first walking debug run.

Test:

```powershell
python tools\debug_reference_playback.py --reference-gait bvh --mode kinematic --clip-id 0 --steps 5
python tools\debug_reference_pd_tracking.py --reference-gait bvh --clip-id 0 --steps 120
```

Pass condition: the chosen debug clip is wrap or long enough, and PD tracking survives a complete intended gait segment before PPO is launched.

## Finding 2 - Recent SMPL Runs Used a Reference Set That Fails the Oracle Layer

Severity: critical

Evidence from existing run artifacts:

- `runs/20260815_165330_reference_playback_mjx_jax/playback_summary.json` reported `valid=0`, `failed=4`, `physical_fail=2`, `motion_over=2`, and average failure step around `44`.
- The same playback reported low key tracking quality during simulated tracking: mean `dm_key` around `0.572`, mean key error around `0.222`, and mean root velocity reward around `0.686`.
- Recent training logs showed evaluation episode lengths around 38 to 45 steps with `done=1.0`, frequent low-height failure, motion-over termination, and pose termination.

Training effect:

- If the reference-target oracle cannot track the motion through the simulator, PPO starts below the required invariants.
- Tuning PPO can hide this for a while, but it cannot reliably recover a walking policy from inconsistent or physically unsustainable targets.

Minimal fix:

- Make `debug_reference_pd_tracking.py` pass for a single SMPL walking clip before using SMPL for RL.
- If it fails, fix retargeting, root scaling, height alignment, PD gains, or morphology mismatch first.

