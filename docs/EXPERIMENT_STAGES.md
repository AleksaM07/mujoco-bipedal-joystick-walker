# Walking experiment stages

The experiments are intentionally separated so a task objective cannot fight a
fixed mocap trajectory.

## Current traffic lights (2026-10-09)

| Gate | Status | Current evidence |
| --- | --- | --- |
| Windows/JAX/MuJoCo execution | **GREEN** | Native Windows CPU JAX loads v21 checkpoints and completes inspect rollouts. |
| Stage 1A wide-stance balance | **GREEN** | The v21 policy survives `500/500`, but its neutral feet are about `52.9 cm` apart and it does not locomote. |
| Stage 1A2 narrow-stance balance | **RUNNING** | `neutral_stance_hip_x=0` reduces foot spacing to about `21.3 cm`. The 1M probe reached about `110/500`; a fair continuation to the original 10M budget is in progress. |
| Stage 1B reference-free forward motion | **RED** | At fixed `0.14 m/s`, checkpoint `000006144000` averages about `-0.001 m/s`; checkpoint `000002560000` averages about `0.005 m/s`. Both are standing policies. |
| Stage 1B support transfer | **RED** | Procedural swing targets can lift a foot, but all forced-lift runs fall because the controller has not first learned lateral COM transfer to the stance foot. |
| Stage 2 reference kinematics | **GREEN** | Velocity, seam, joint-limit, foot-penetration, and stance-slip invariants pass for `02_01`, clip 1. |
| Stage 2 dynamic feasibility | **RED** | Default free-root PD tips at `0.92 s`; inverse dynamics reaches a `14.51x` actuator ratio. Static holds pass `3/8` phases with gravity and `8/8` with gravity disabled. |
| Stage 2 multi-cycle tracking | **RED** | The best in-place diagnostic (`sagittal_alpha=0.10`, root XY scale `0`, stance lock off) passes three cycles but collapses at `4.30 s` and has no forward travel. |
| Stage 2 training backend | **GREEN** | Warp completed the 1k smoke run and the 200k isolated tracker run. The current MJX-JAX compile path is not practical on this machine. |
| Stage 2 support transfer | **RED** | The least-bad in-place reference keeps both feet contact-ready for every sampled frame; COM is outside support for `22.3%` and capture point for `29.5%` of the cycle. |
| Stage 3 joystick plus style | **BLOCKED** | Stages 1B and 2 must pass independently first. |

The first red light in the reference branch is therefore dynamic retargeting,
not BVH parsing and not PPO. The kinematic reference looks coherent, but its
root/COM/contact trajectory is not a self-consistent free-root motion for the
v21 humanoid.

The 2026-10-09 sweep also exposed a configuration interaction:
`reference_root_xy_scale` produced identical PD results from `0.0` through
`1.0` while `reference_lock_stance_feet=True`. The hard stance-foot correction
overrides the earlier root-XY scaling. With stance locking disabled, the same
scale sweep changes survival from `1.78 s` at scale `0.0` to `0.68 s` at scale
`1.0`. Root travel and stance locking must therefore be redesigned as one
retargeting operation instead of tuned independently.

Fresh visual evidence is in `videos/current_v21_gate_ab`: free root fails at
`0.94 s`, Z-only pinning fails at `3.04 s`, XY-only pinning fails at `0.90 s`,
and full root-position pinning fails at `2.36 s`. These are diagnostic pins,
not candidate training physics.

## Stage 1A: v21 standing-stability bootstrap

Purpose: prove that the current generated model, contacts, action map, reward,
and PPO stack can learn stable control without any BVH dependency.

The preset uses a standing reset, default-pose-centered actions, forward-slow
commands, task reward, 18 actuators, and no ERFI or domain randomization.

```bash
python train.py --task-only-baseline --low-vram --timesteps 10000000
```

`--low-vram` is the recommended schedule for the 6 GiB RTX 2060. It uses 512
training environments, 8 evaluation environments, a 10-step rollout, batch
size 512, 4 minibatches, and 2 updates per batch. Explicit CLI values still
override individual preset values.

Pass criteria:

- evaluation episode length reaches `500/500`;
- physical termination rate trends toward zero;
- fixed in-distribution commands are measured explicitly so standing is not
  mistaken for walking.

The completed 10M checkpoint reached `500/500`, but a fixed `0.10 m/s` audit
measured `-0.003 m/s`. Stage 1A therefore learned standing stability, not gait.

The old neutral pose is also unusually wide for a single-support curriculum:
the right and left soles start near `y=-0.264 m` and `y=+0.264 m`. A swing asks
the COM to cross roughly `26 cm` toward the stance foot. Setting the symmetric
hip-abduction center from `0.18 rad` to `0` narrows the feet to about `21.3 cm`.
Because the original wide policy needed almost the full 10M steps to reach
`500/500` (`131/500` at roughly 1.1M), the narrow pose cannot be rejected from
its 1M result alone. Its 9M continuation is the current controlled A/B test.

## Stage 1B: required forward locomotion

Before forward commands, isolate lateral support transfer on the narrow model.
The environment now has an opt-in phase-aware COM target. It moves smoothly
from the foot midpoint toward the stance foot, then returns to the midpoint at
the phase transition. The default reward scale is zero, so existing experiments
are unchanged. Once the narrow standing policy passes, run a transfer-only
curriculum with gait active at zero command, foot-lift shaping disabled, and
the centered COM-target shaping enabled. Require decreasing `gait_com_error`
without reduced survival before adding swing assistance:

```bash
python train.py \
  --task-weight-transfer \
  --weight-transfer-prior-scale 1.0 \
  --low-vram \
  --timesteps 2000000 --num-evals 9 \
  --resume-from <stable-narrow-stance-checkpoint>
```

This order is based on the procedural probes:

- the old sinusoidal target at small scale barely cleared the floor;
- a calibrated sagittal hip/knee target produced up to about `13.5 cm` lift at
  full scale, but free-root survival collapsed to roughly 100 steps;
- PPO learned to cancel weak assistance, while enforcing the lift also failed;
- removing the zero-command overspeed contradiction improved reward and
  survival, but still did not create a swing because support transfer was absent.

The first reward-only 2M transfer run failed: its COM error grew from about
`4.6 cm` to `7.3 cm`. A constrained static solve then proved that both target
poses are physically reachable, shifting COM by about `5.3 cm` with only
`0.12 mm` foot-marker drift. The calibrated action prior improved the untrained
dynamic COM error, but the first 1M guided PPO run again selected its step-0
checkpoint and ended at `8.7 cm` error. That audit exposed another direct
conflict: zero-velocity tracking penalized the lateral velocity required to
move between stance feet. During zero-command gait only, lateral velocity is
now supervised by the phase-aware COM objective instead; sagittal and yaw drift
remain penalized.

Those assisted probes also started from a randomized gait phase, so some resets
immediately received a peak hip/knee lift target. The new curricula ramp COM and
procedural action assistance in over their first 25 control steps. This removes
the reset impulse without changing old checkpoints (`gait_warmup_steps=0` by
default).

Resume the stable Stage 1A checkpoint with the `forward` command distribution
(`0.15-0.35 m/s`, no zero-command episodes). Those commands are above the
environment's stuck threshold, so standing is no longer a good solution.

First run a short velocity bootstrap on the bridge command range
(`0.10-0.18 m/s`) with phase-clock foot shaping disabled and PPO discounting
raised to `0.995`. This prevents the binary swing penalties from dominating
before the policy can balance while moving, while making delayed falls matter
to the optimizer:

```bash
python train.py \
  --task-forward-bootstrap \
  --low-vram \
  --timesteps 2000000 --num-evals 5 \
  --resume-from runs/20261008_1823_v21_task_only_baseline_V1_steps-10m_reward-245p5164_s/checkpoints/000010137600
```

Only after this passes the fixed-command survival and velocity gates, use
`--task-forward-curriculum` to restore full gait-clock shaping. Intermediate
values can be tested with `--gait-shaping-scale`.

Pass criteria:

- `500/500` survival at fixed `0.15`, `0.25`, and `0.35 m/s` commands;
- mean forward velocity is close to each requested command;
- positive forward displacement without repeated resets;
- visual inspection confirms stepping rather than sliding or leaning.

## Stage 2: isolated reference tracker

Do not run PPO until every pre-training gate below passes on the selected clip.

```bash
python tools/audit_reference_invariants.py
python tools/audit_reference_dynamics.py \
  --reference-gait-file marina_bvh_parse/walking_forward_bvh/02_01.bvh \
  --clip-id 1 --reference-loop-mode wrap \
  --out-dir runs/reference_preflight/dynamics
python tools/debug_reference_pd_tracking.py \
  --reference-gait-file marina_bvh_parse/walking_forward_bvh/02_01.bvh \
  --clip-id 1 --steps 180
```

Required gates:

- stored joint/root velocities match finite differences of positions;
- joint pose, root orientation, and velocity are continuous at the loop seam;
- stance foot slip and floor penetration stay below configured tolerances;
- reference positions stay within joint limits;
- inverse-dynamics actuator ratios remain at or below `1.0`;
- a free-root PD tracker survives at least three complete motion cycles.

Only after those gates pass:

```bash
python train.py --reference-tracker
```

Current `02_01`, clip 1 status after fixing the old `qvel *= 0.25` mismatch:

- kinematic invariant audit: **PASS**;
- free-root PD tracking: **FAIL**, tips after about `0.92 s`;
- inverse dynamics: **FAIL**, max actuator ratio about `14.51`.

This means the clip is internally time-consistent now, but it is not yet a
physically executable target for this humanoid.

The first isolated learning probe uses the least-bad physically tested target:
zero root XY travel, no hard stance-foot lock, sagittal projection `0.10`, and
other-joint projection `0.35`. It deliberately asks for marching in place
before forward travel:

```bash
python train.py \
  --reference-tracker-inplace \
  --reference-tracker-backend warp \
  --low-vram \
  --timesteps 1000000 \
  --num-evals 11
```

Tracker episodes are `500` control steps (`10 s`). This is intentionally longer
than the open-loop PD failure at `4.30 s`; the former `180`-step horizon ended at
`3.6 s` and could therefore report a misleading full-length episode before the
known failure. Passing requires free-root survival across the full horizon plus
alternating foot contact and bounded pose/root/key-marker errors. Reward alone
is not a pass criterion.

For this `mjx_jax` tracker, `--low-vram` uses 128 training environments, four
evaluation environments, and batch size 128. The Warp task presets retain their
512-environment low-VRAM profile. A 512-environment MJX-JAX compile reached more
than 7.4 GiB RSS and failed to produce an executable within 20 minutes on the
current WSL setup.

The first Warp learning probe used the old PPO discount `0.97` and was stopped
at 512k steps after a monotonic failure trend: mean episode length moved from
`116.1` at step zero to `113.6`, `104.3`, `83.4`, `88.0`, and `82.3`. Foot slip
also grew sharply. At 100 steps, `0.97^100` leaves only about 4.8% of a delayed
fall cost, so the tracker preset now uses `0.995` (about 60.6% at 100 steps).
This is the next isolated comparison; physics, reference, reward, and action
mapping remain unchanged.

That `0.995` A/B run still degraded from `116.1` to `81.6` steps by 512k; its
best checkpoint was the untrained step-zero policy. The pure imitation branch
does not apply the generic action/action-rate costs, while full residual scale
`1.0` allowed normalized action magnitude to grow from about `2.18` to `2.86`.
The next curriculum therefore limits residual authority to `0.10`, preserving
the feed-forward reference target while allowing small learned balance
corrections. Discounting remains `0.995`; no other physics or reward term is
changed.

The `0.10` residual probe is the first run that improved relative to its own
initial policy: mean episode length rose from `103.8` to `119.9` steps at
checkpoint `000000102400`, then ended at `117.0` by 204800 steps. It still
falls after only about 2.4 seconds, so this is a useful optimizer signal, not a
passing gait.

Frame-by-frame inspection exposed why the damped target cannot teach walking.
With the old post-IK stability blend, the right sole stays at about `-5 mm`
throughout the cycle and the left sole rises only about `20 mm`. At a `35 mm`
contact-ready threshold both feet are therefore down in `112/112` samples.
The stability blend erased the foot targets fitted by IK.

Applying the stability prior before IK restores alternating swing: at full IK
correction the left and right soles reach roughly `114 mm` and `86 mm`, with
one support foot in `56/112` samples. That target is much less feasible,
however: mean/max actuator ratios increase to `15.12x/69.15x`, and capture
point is outside support in `68.8%` of samples. Partial IK corrections trace
the same trade-off; a `0.50` correction produces one-foot support in `41/112`
samples at a `10 mm` contact threshold but still reaches a `31.5x` actuator
ratio. Preserving only lateral root translation does not improve the relative
COM-to-foot margin because a free-root translation moves the whole character.

The next reference change must therefore be support-aware retargeting: jointly
optimize joint pose, stance-foot position, COM/capture-point margin, temporal
smoothness, and actuator effort. Neither the old stability blend nor another
long PPO run can resolve this trade-off on its own.

The evaluator also previously reconstructed only part of `EnvConfig`, so old
videos silently used residual scale `1.0` and full sagittal motion even for the
`0.10`/`0.10` checkpoint. It now restores the saved reference action, reset,
lookahead, stability, reward, and termination fields. Videos named
`*_evaluator_config_bug_*` are retained only as invalid forensic artifacts;
the corrected checkpoint-102400 render is
`videos/reference_tracker_residual010_best_102400_warp_corrected_quad.mp4`.

## Stage 3: joystick plus motion style

Start only after Stages 1 and 2 pass independently. A fixed forward clip must
not be rewarded against arbitrary lateral/backward/yaw commands. Use either a
command-conditioned reference transformation or a hierarchy with a low-level
gait controller and a high-level joystick controller.
