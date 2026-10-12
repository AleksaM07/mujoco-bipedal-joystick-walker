# Recovery phase 2: consistent reference curves

Branch: `recovery/project-v2-imitation`. Follows phase-1 commit `6f5d697`.
This is a numerical-reference repair and diagnostic checkpoint, **not a walking
success**. No PPO training, push, checkpoint migration, or reward retuning was done.

## What changed and why

Phase 1 removed the velocity-only x0.25 bug and corrected measurements, but
positions and velocities were still interpolated independently. Consequently,
the velocity rewarded by imitation did not equal the derivative of the pose the
robot was supposed to imitate. WRAP clips also had discontinuous endpoints.

The new behavior requires `recovery_physics_version="consistent_v2"`:

1. Joint and root-position coordinates use cubic Hermite curves with analytic
   derivatives. Joint tangents are monotone PCHIP-style slopes: between-frame
   interpolation does not overshoot the corrected joint nodes.
2. Root rotation uses a sign-continuous normalized quaternion cubic. Its
   differentiated normalization supplies the MuJoCo **local** free-joint angular
   velocity. Normalizing a derivative quaternion itself would be incorrect.
3. Pelvis anchor position/orientation/velocity and key markers come from forward
   kinematics of that same runtime pose and velocity. They are no longer separate
   interpolated trajectories that can disagree between frames.
4. WRAP uses one shared pose and tangent at both ends. Root XY translation is
   separated from the periodic residual and accumulated over positive and negative
   cycles. Nonzero Z drift and unsupported whole-spin quaternion loops are rejected.
5. CLAMP endpoints have zero derivatives; single-frame clips remain stationary.
   A test caught and fixed a single-frame zero-duration division issue.
6. An `exact=True` reset now uses the actual queried pose/velocity without a second
   root-height projection. The existing projected reset/fallback path still exists
   and must be audited separately; this does not abolish reset acceptance checks.

The curves are C1 (continuous pose and velocity), not necessarily C2. Acceleration
can change at knots, and no feasible-torque claim follows from this interpolation.

## Explicit loop repair, not a silent reinterpretation

`reference_loop_closure="strict"` is the default. It rejects mismatched wrapping
endpoints. Our actual processed 02_01 data fails that check, as expected.

`reference_loop_closure="bounded"` is an explicit v2-only geometric edit. It
distributes endpoint corrections over the clip using a smooth phase function.
Maximum accepted source gaps are 0.2 rad per joint, 0.02 m root height, and 0.2 rad
root orientation. These are engineering guardrails, not physical tolerances or
proof that all accepted references are good.

The first real-BVH test exposed another problem: closing the loop can move already
legal joint nodes outside actuator limits. The fix projects corrected **nodes**
into the intersection of joint-position and actuator-control ranges, then rebuilds
their monotone tangents. Runtime output is not clipped, which would break analytic
derivative consistency. Invalid original nodes are rejected (apart from 1e-6 rad
numerical/XML rounding tolerance), and every projection is recorded.

The loader prepares all three cycles even when clip 1 is forced. Maximum changes:

| Clip | Max joint edit | Max root-position edit | Max root-orientation edit | Projected coordinates |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.061022 rad | 0.2041 mm | 0.017529 rad | 40 |
| 1 | 0.024609 rad | 0.1454 mm | 0.016303 rad | 17 |
| 2 | 0.068237 rad | 0.5935 mm | 0.012249 rad | 0 |

For clip 1 this is at most about 1.41 degrees per joint and 0.934 degrees root
orientation. Its largest limit projection is 0.007921 rad. The JSON records both
projection distance and total change from the original, so these are not hidden.

## Measurements on the actual model and BVH

Inputs: v21 no-arms trainfast model, CMU `02_01.bvh`, clip 1, source frames 79:213,
134 frames, duration 1.108329 seconds, full playback speed. Evidence with input,
processed-reference and source hashes:
`analysis_outputs/recovery_phase2_20261012.json`.

| Numerical measurement | consistent_v1 | consistent_v2 |
| --- | ---: | ---: |
| Joint dq/dt vs target RMSE | 0.106101 rad/s | 6.51e-8 rad/s |
| Root dpos/dt vs target RMSE | 0.033605 m/s | 1.08e-8 m/s |
| Endpoint joint pose gap | 0.049218 rad | 0 |
| Endpoint joint velocity gap | 4.039754 rad/s | 0 |

The JAX derivative comparison samples 32 between-frame times. Quaternion angular
velocity error is about 1.36e-8 rad/s. An independent native MuJoCo
`mj_differentiatePos` finite-difference comparison also passes; the JSON gives its
step size and errors. Finite differences have truncation and float32 roundoff,
so they are not expected to match the analytic comparison's near-zero error.

At 533 dense samples, v2 native-FK key-position disagreement is below 0.8 micrometers;
anchor linear/angular velocity disagreement is below 3e-7 in the respective units.
Sampled joint/control-limit excess is about 4.8e-10 rad, i.e. floating-point noise.
The numerical gate permits at most 1e-6 rad here.

Exact nodal seam equality does not imply identical values at two different times:
at one microsecond before/after the runtime seam, the maximum joint velocity
difference is about 0.00212 rad/s. The tests check shared endpoint tangents and
runtime continuity; they do not hide a finite seam gap by lowering target speeds.

Quarter-speed preparation produces identical positions/orientations, four times
the duration and one quarter of all curve tangents. The v1 processed-reference
digest still matches the saved phase-1 digest (`b7567506...fab14309`).

## The next red light: contact geometry

Both versions were sampled at 533 times, four subdivisions per interval, with the
native MuJoCo foot shapes against the horizontal floor:

| Contact diagnostic | consistent_v1 | consistent_v2 |
| --- | ---: | ---: |
| Maximum geometric penetration | 5.20 mm | 6.03 mm |
| Excess beyond existing 5 mm preload | 0.20 mm | 1.03 mm |
| Foot/time samples penetrating >1 mm | 62.66% | 62.29% |
| Near-floor point horizontal speed, median | 0.0908 m/s | 0.1014 m/s |
| Near-floor point horizontal speed, 95th percentile | 0.4202 m/s | 0.4319 m/s |

Important baseline: `FOOT_CONTACT_PRELOAD=0.005` in `biomechanics_env.py` and
`_align_reference_roots_to_sim_floor` deliberately place the lowest foot surface
5 mm below the floor. Most of the reported penetration is therefore inherited,
intentional preload, not an accidental 6 mm introduced by phase 2. Its dynamic
appropriateness remains unvalidated; do not interpret the raw negative-height
fraction as a proven failure rate. Phase 2 adds about 0.83 mm to the deepest
penetration in this clip relative to v1.

The velocity proxy is the material-point velocity at a lowest foot support point,
using the queried pose/qvel, when surface height is <=1 cm (including penetrating
poses). It is **not force-labeled stance**, and v1's qvel is itself inconsistent
with its position derivative. Thus the speed columns are not a controlled claim
that v2 increased actual stance slip. The geometry shows that closing the seam
does not establish foot-floor compatibility and slightly deepens the most
penetrating pose. Small soft-contact penetration can occur in simulation;
these kinematic measurements alone neither prove nor disprove dynamic acceptability.

Do not start a long PPO run from these numbers. Next isolate stance classification,
root-height/foot-lock retargeting, and acceleration/actuator-force requirements.
PD/free-root replay is useful evidence, not an absolute theoretical prerequisite
for RL success. Preserve this version as a control when changing contact geometry.

### Code-backed leads for the next contact audit (not yet proven causes)

`_configure_bvh_reference` currently applies floor alignment/foot locking, shifts
and normalizes IK targets, solves IK, applies the stability prior, scales root XY,
then aligns/locks again. The v2 loop correction comes **after** that final pass;
it does not repeat contact projection, which could break the newly closed loop.

`_lock_clip_root_xy_to_stance_feet` locks one active foot's **geom center** in XY
using a 3.5 cm height threshold. It does not constrain a rotating sole's material
contact point or both feet during double support. Our velocity diagnostic instead
uses a lowest surface point and a 1 cm threshold. The distinction can explain
nonzero reported speed even with an apparently successful center lock; measure
both, per foot, before declaring the lock broken.

The final lock follows `reference_root_xy_scale=0.35`, so it can undo some scaling
during uninterrupted stance. It is premature to blame a simple 65% translation
mismatch. A small controlled comparison should vary XY scale (0.35/1.0) and locking
(on/off), holding source, model, timing and other settings fixed; report total
travel, phase labels, center/support-point speeds and switch events separately.
Also compare v1/v2 per-node versus between-node heights and the root-Z speed-limit
pass. None of these four-way experiments has been run in this phase.

## Tests, compatibility and costs

- 63 unittest tests pass, including analytic/JVP/native comparisons, negative and
  multiple loops, clamp/single-frame handling, sign-invariant quaternion inputs,
  bounded edits, invalid inputs, exact reset, support geometry, config forwarding,
  saved evaluation semantics and resume rejection. Tests are pytest-compatible;
  pytest is not installed in the existing native environment.
- One actual CPU MJX reset/step is finite with `done=0`; no policy is trained.
  Seed 7 accepts one exact reset with zero rejections and no standing fallback.
  The evidence includes reset diagnostics. This one sample cannot establish that
  all reset phases are accepted or that the robot survives a whole cycle.
- New modules/tests pass Ruff E/F/I and formatting checks; changed production
  modules compile and `git diff --check` passes. Existing unrelated lint findings
  in the large legacy modules are not reformatted away.
- The default remains `legacy`. Existing `consistent_v1` interpolation stays
  unchanged. Saved evaluation restores v2/closure settings, and training refuses
  silent v1-to-v2 checkpoint migration. This is project-v2 compatibility, not a
  claim that arbitrary checkpoints from main/other experiments match this branch.
- Runtime FK adds cost: the initial native CPU measurement of 533 warmed queries
  was about 12 ms for v2 versus 0.5 ms for v1. Timings vary; this is an isolated
  diagnostic batch, not PPO throughput. First reset/step compilation took minutes.
  Warp physics and actual training throughput remain untested in this phase.

## Reproduce on Windows Git Bash

From the repository root (no WSL, no uv dependency synchronization):

```bash
./.venv-win/Scripts/python.exe -m unittest discover -s tests -v
./.venv-win/Scripts/python.exe -m tools.check_reference_curves --output analysis_outputs/recovery_phase2_20261012.json --mjx-smoke
```

Omit `--mjx-smoke` for the faster numerical/contact audit. These commands do not
load or alter checkpoints. Re-running the audit deliberately replaces its JSON
report. Its `git_head` identifies the parent of the uncommitted patch being tested;
the recorded source hashes identify the actual measured source files.

After contact/dynamics checks: contrast reward on reference, standing, wrong phase,
swapped legs and sliding; freeze one pure-tracker preset and training provenance;
run a short integration pilot. Only then choose a serious training budget. Do not
add joystick, domain randomization and a new curriculum at the same time.
