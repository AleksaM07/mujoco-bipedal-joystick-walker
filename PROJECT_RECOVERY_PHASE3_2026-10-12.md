# Recovery phase 3: contact, retargeting and dynamics diagnosis

Branch: `recovery/project-v2-imitation`, based on phase-2 commit `142d93c`.
No PPO run, policy evaluation, push or checkpoint migration. The native model's
production collision masks, gains, damping, friction and force limits are unchanged.

## Outcome

The reference passes numerical curve tests but **fails the physical self-collision
check**. Left and right sole capsules intersect throughout the sampled cycle.
This is a stronger red light than the previously measured few millimeters of
floor penetration. It is present before attempting to train a policy.

A separate confirmed NumPy indexing bug in foot-target height normalization is
fixed behind `consistent_v3`. Controlled replays improve, but remain unstable and
the foot-foot intersection remains. Do not equate this bugfix with a valid gait.

## 1. Confirmed height-normalization bug and versioned fix

In `_normalize_ik_foot_target_heights`, mixed NumPy indexing
`array[clip, :frames, foot_indices, 2]` yields `(feet, frames)`, not `(frames, feet)`.
Taking `min(axis=1)` therefore subtracts each foot's minimum over the whole clip,
instead of applying one common offset to both feet in each frame.

For real `02_01`, clip 1, 133/134 frames miss the documented lowest-foot target
height. The maximum discrepancy is **0.189345 m**, RMS **0.125954 m**. These are
upstream IK-marker target discrepancies, not actual sole-floor penetration.
Later IK and floor alignment can conceal them visually.

V3 slices the clip first, explicitly keeping `(frames, feet)`, and applies one
shared vertical offset per frame. It preserves XY, non-foot targets and the input
array. Padding repeats the final valid corrected frame. The measured target-height
error becomes about 3e-8 m, and all three prepared clips pass the unchanged bounded
loop-closure limits. Tests include nonsquare frame/foot dimensions and old behavior.

`legacy`, `consistent_v1` and `consistent_v2` keep their previous behavior. The
default stays `legacy`; v3 must be chosen explicitly. Config restoration/forwarding
is tested, and checkpoint resume refuses implicit v2-to-v3 migration. Prepared v2
reference hashes still match phase 2. No existing checkpoint is silently retargeted.

## 2. Major physical defect: left and right feet overlap

Native MuJoCo collision detection, 532 off-knot samples, same v21 no-arms model,
same clip 1, duration 1.108329 s:

| Measurement | v2 control | v3 height fix |
| --- | ---: | ---: |
| Samples with sole-sole penetration >1 micrometer | 532/532 | 532/532 |
| Deepest sole-sole penetration | 99.55 mm | 93.85 mm |
| Neutral standing sole-sole collision | None | None |
| Numerical joint derivative RMSE | 6.51e-8 rad/s | 6.48e-8 rad/s |

The two sole capsules have radii about 51.36 mm each. Near-100 mm penetration is
not plausible soft floor contact. It is also not the intentional **5 mm floor
preload**, which remains unchanged. Pure XY root scaling/translation cannot repair
relative foot overlap; the joint/root orientation retargeting and foot geometry
must be checked together. We have not yet isolated which upstream stage first
introduces the collapse, so do not attribute it specifically to a mirrored axis,
source motion, IK weighting or the stability prior without the stage-wise audit.

At the v2 sample with the largest requested actuator force (index 224), sole-sole
penetration is 97.79 mm and its inverse-contact normal force is about 5,365 N.
The largest generalized constraint component is about 8,346, compared with about
552 for the inertial component at that sample (vector components have their own
N/Nm units). The report retains full per-DOF decomposition, not just a mixed-unit
norm. A different root-force peak is dominated by inertia; collision is not the
only remaining problem.

Evidence: `analysis_outputs/recovery_phase3_20261012/collisions.json` and
`baseline/summary.json` in that directory.

## 3. Controlled XY-scale / foot-lock comparison

Four full-speed v2 variants, same source, model, rewards and other settings:

| Root XY scale | Lock | Net root X per cycle | Near-floor support-point speed median, left / right |
| --- | --- | ---: | ---: |
| 0.35 | on | -0.05479 m | 0.1057 / 0.0991 m/s |
| 1.00 | on | -0.06184 m | 0.1064 / 0.0989 m/s |
| 0.35 | off | +0.57123 m | 0.6463 / 0.5787 m/s |
| 1.00 | off | +1.63209 m | 1.6875 / 1.6134 m/s |

The lock reduces this speed proxy but substantially changes net translation.
Simply turning it off or setting XY scale to 1 does not solve the reference.
With locking, the reference no longer represents the original forward progression
in simulator X. V3 plus the same lock has net X -0.00443 m/cycle, almost in place.

The lock follows a selected geom **center**, not a rotating contact material point.
It constrains one foot and keeps that foot active while it stays within its 3.5 cm
height threshold. In v3 the right foot is within this threshold for the entire
cycle; this warrants a phase/contact-selection audit. Our 1 cm near-floor labels
are not ground-truth stance or force labels. CSVs retain per-foot heights, center
and support-point speeds, node flags and positions for further inspection.

## 4. Required dynamics versus available actuator control

We differentiate v2/v3 qvel analytically at off-knot midpoints, then run native
inverse dynamics. Actuator transmission/gear, passive forces, affine servo bias,
enabled/asymmetric control limits and force limits are handled separately. Tests
cover nonunit/negative gear, passive damping, contact, gravity and free fall.

MuJoCo inverse dynamics returns the forces required for the prescribed
configuration/velocity/acceleration under its current soft-contact model. It is
not an optimizer over alternative contact geometry or nearby motions.
See [MuJoCo inverse dynamics](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#inverse-dynamics)
and [actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model).

| At least one actuator exceeds its required limit | v2 | v3 |
| --- | ---: | ---: |
| Force limit, fraction of 532 samples | 99.25% | 100% |
| Control-target limit, fraction of samples | 100% | 100% |
| Maximum free-root linear acceleration | 186.42 m/s² | 108.44 m/s² |

These are exact-reference requests, **not observed saturation during replay**.
The model has 18 actuators and 21 unactuated velocity coordinates, including the
free root and passive joints. Required unactuated forces are reported separately.

Large nominal `forcerange` does not imply reachable torque from a position servo.
For example ankle_z has kp160 and ctrl ±0.20944 rad: at q=0 it can request only
±33.51 Nm from legal controls, despite a nominal ±260 Nm force clamp. Control
targets and force capacity must both be considered.

A diagnostic model copy disables **only sole-sole collision**, keeping floor
contact. Even then, required force limits are exceeded in 87.41% of v2 samples
and 88.53% of v3 samples; control limits in 100% for both. This supports multiple
remaining defects, including excessive trajectory acceleration. The production
model is unmodified. Removing collisions would hide intersecting feet and is not
an accepted fix or a training preset.

## 5. Actual free-root replay, not just a kinematic movie

Exact reference qpos/qvel initializes each run once. Native physics then uses
unchanged position actuators with an unassisted free root. Four starting phases
times three control timings times two versions = **24 trials**, each requesting
three cycles (3.325 s). Physical termination follows the environment's root-height
and torso-up thresholds, plus finite-state checks; no optional pose termination.

| Start phase | v2 survival, 50 Hz current target | v3 survival, same controller |
| --- | ---: | ---: |
| 0.00 | 1.310 s | 1.595 s |
| 0.25 | 1.040 s | 1.455 s |
| 0.50 | 0.810 s | 1.070 s |
| 0.75 | 1.125 s | 2.020 s |

V3 improves all four matched trials, but **none finishes three cycles**. Updating
at every 5 ms physics step or using one 20 ms control-step lookahead does not make
any trial complete either. All trials show zero measured actuator force-limit
saturation. This is consistent with restricted position-servo authority and the
robot deviating from the prescribed motion; it does not contradict large inverse
forces that would be needed to remain exactly on that motion.

This is open-loop position tracking, not a learned balance controller. Failure
does not prove that no RL policy could learn a nearby viable gait. It does show
that these raw reference/controller pairs are not yet valid multi-cycle trackers.

## Tests, files and reproduction

104 regression tests pass. New diagnostics and tests pass Ruff E/F/I and formatter
checks; touched production files compile. No new dependency was installed. Existing
unrelated lint warnings in the large environment are left untouched.

Saved evidence under `analysis_outputs/recovery_phase3_20261012/`:

- `baseline/summary.json` and four per-variant contact CSVs: complete v2 2x2 audit.
- `corrected/summary.json` and one contact CSV: isolated v3 normalization change.
- `collisions.json`: self-collision checks, standing control, numerical curve gates
  and explicitly labeled diagnostic-only collision ablation.

Inputs, configs, source/processed-reference hashes and limitations are recorded.
The audit head is the parent commit of the measured working-tree changes.

Windows Git Bash, repository root:

```bash
./.venv-win/Scripts/python.exe -m unittest discover -s tests -v
./.venv-win/Scripts/python.exe -m tools.check_reference_contacts --output-dir analysis_outputs/recovery_phase3_20261012/baseline
./.venv-win/Scripts/python.exe -m tools.check_reference_contacts --baseline-only --recovery-physics-version consistent_v3 --output-dir analysis_outputs/recovery_phase3_20261012/corrected
./.venv-win/Scripts/python.exe -m tools.check_reference_collisions
```

No checkpoint is loaded. Re-running replaces the named generated audit reports.

## Next experiment, before a training budget

1. Trace foot lateral spacing and collision depth through source mapping, raw
   pose, IK targets/solution, stability prior and final curve preparation. Neutral
   standing is the geometric control. Find the first stage introducing overlap.
2. Correct that stage with left/right and coordinate-frame regression fixtures;
   require adequate capsule separation and measure actual Cartesian IK residuals.
   Do not just disable foot-foot collision or increase force limits.
3. Revisit phase-aware stance selection, root translation and acceleration
   regularity, then repeat contact/dynamics/replay from all four phases.
4. Only after a physically credible reference, audit reward discrimination and
   freeze a pure-tracker training preset. Keep joystick and new curricula out of
   this experiment. No 80M-step training is justified by the current gate results.
