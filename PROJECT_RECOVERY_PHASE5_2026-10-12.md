# Recovery phase 5: one corrected WRAP cycle and the next physical defects

Branch: `recovery/project-v2-imitation`, parent `96bee03`.
Implementation is committed separately as phase 5A, `d2e90b7`
(`feat: select reference source cycles before retargeting`). Phase 5B contains
only diagnostics, regression tests for those tools, measured evidence and reports.
This phase adds explicit input-clip selection and diagnostic tools. It does not
change rewards, actuator settings, collision masks, IK weights or training presets.
No PPO run, push or checkpoint migration was performed.

## One valid numerical cycle, with source provenance

`reference_clip_ids=[1]` now selects original **loader batch clip 1 before duration
filtering, IK and curve preparation**. It becomes runtime clip 0. The source
remains CMU `02_01.bvh`, frames `[79:213)`, 134 frames, duration 1.108329 s.
This is the same diagnosed cycle, not a newly chosen easier source motion.

`reference_forced_clip_id` still means a runtime sampling index. It is not used
to pretend that rejected source clips have been removed. Unselected clip 0 still
fails its original bounded-loop check; this patch does not relax that check.

The optional selector preserves requested order and all clip-aligned arrays,
source frame ranges, weights and metadata. Empty, duplicate, noninteger, boolean
or out-of-range IDs fail. Explicitly selected short clips fail rather than being
silently dropped. `None` preserves the old duration-filter behavior, including its
historical all-short fallback. Saved config and training CLI forward the choice.

Actual v3 control with no selection still matches its phase-3/4 digest:
`0fd0f0099cbf9f69fd35a1ef985e20122c8eecf3e095f06e29a25ecc9f9653d9`.
Selected v4 WRAP digest:
`12ae4d6c90078d151670c96a9c46079d9d957da708d6edb2a0fe387dc2d27d0f`.

## Numerical gates pass; physical tracking does not

| Check | Selected v4 WRAP result |
| --- | --- |
| Joint dq/dt versus reported qvel RMSE | 7.09e-8 rad/s |
| Root derivative versus reported velocity RMSE | 4.34e-8 m/s |
| Prepared joint endpoint pose / velocity gap | 0 / 0 |
| Exact reference reset versus runtime query | Pass |
| Repeated positive/negative-cycle query consistency | Pass |
| Sole-sole penetration, 533 nodes/intermediate samples | None |
| Separate 532 off-knot sole collision check | None |
| Net reference root XY displacement | (+1.37426, -0.09930) m/cycle |
| Mean reference XY travel speed | 1.24317 m/s |
| Deepest sole-floor penetration | 10.511 mm, including 5 mm intended preload |

The phase-4 finite CLAMP result had shallower penetration (5.680 mm); that did not
certify WRAP. Bounded seam correction changes geometry and must be rechecked.
Near-floor support-point speed medians are 0.1273 / 0.1380 m/s for left/right,
with much larger peaks. These height labels are proxies, not true force-based
stance annotation or a no-slip certificate.

Native unassisted position replay, requested three cycles (3.325 s):

| Start phase | Physics-rate target | 50 Hz current target | 50 Hz one-step lookahead |
| --- | ---: | ---: | ---: |
| 0.00 | 1.310 s | 1.330 s | 1.290 s |
| 0.25 | 1.120 s | 1.125 s | 1.105 s |
| 0.50 | 1.125 s | 1.120 s | 1.120 s |
| 0.75 | 1.000 s | 1.075 s | 0.905 s |

All 12 stop before the requested horizon. None shows actuator force-clamp
saturation. These are open-loop native position controls with unchanged gains,
not a learned balance controller or a test of MJX/Warp equivalence. The corrected
reference now moves forward; comparing survival directly to the near-stationary
v3 reference is not a like-for-like locomotion success metric.

Exact-reference inverse dynamics exceeds at least one actuator force range in
**91.17%** of 532 samples and at least one control-target range in **100%**.
Required free-root force reaches 123,140 N and root linear acceleration reaches
**1,516.81 m/s^2**. These are prescribed-motion requirements, not forces observed
in replay. Finite derivatives and a continuous loop can still describe a wildly
nonphysical trajectory.

## Root locking introduces sharp motion changes before interpolation

The observer now measures the real input/output of each stance-lock call, keeping
the production method unchanged. XY second differences are a nodal roughness
diagnostic; they are explicitly distinct from the exact interpolant acceleration.

| Lock pass | Peak XY second difference before | After | Worst center frame |
| --- | ---: | ---: | ---: |
| Initial floor alignment, before IK | 27.48 m/s^2 | 398.74 m/s^2 | 108 |
| Final alignment, after IK/prior/XY scaling | 139.56 m/s^2 | 369.06 m/s^2 | 1 |

For example, final-lock root X values at frames 0/1/2 are
0.00000, -0.01247, +0.00047 m, spaced only 8.333 ms apart. The sharp reversal
exists at nodes before curve preparation. The largest exact inverse-force sample
is the first off-knot sample, near the cycle seam. Interpolation and the cyclic
tangent construction further shape the acceleration; they are not the sole
origin of the rough input. No smoothing or contact-state rule was silently added.

This identifies a concrete pipeline problem, but does not yet isolate which
stance transitions, startup conditions, or upstream IK changes should be corrected.
The report retains local root-position windows around each peak for that next audit.

## Why IK still misses the correct foot targets

An independent SciPy least-squares diagnostic evaluates nine evenly spaced saved
frames: 0, 16, 33, 49, 66, 83, 99, 116, 133. It uses native MuJoCo marker Jacobians,
the same root, passive/trunk coordinates, legal actuator-control bounds, raw seed
and actual previous production IK pose. Nothing is fed back into the reference.
No dependency installation was needed.

Two counterfactuals separate solver convergence from its objective:

1. A properly bounded solve of the existing objective (pose penalty 0.20,
   previous-pose penalty 0.08) barely changes most sampled frames. Simply running
   more of the same iterations would not remove their centimeter-scale errors.
   At frames 0/116/133 it improves the right-foot fit; clipping an unconstrained
   step is not identical to optimizing the bounded problem.
2. With both pose penalties removed, both foot markers fit within 1e-8 m in
   **6/9 sampled frames**, at the same frozen root and control bounds. This is
   evidence that the old pose/previous-frame preference explains much of their
   error, not that the robot geometry makes every target unreachable.

Exceptions and costs matter:

- Frame 49 retains 40.79 mm left-foot error at the left ankle_z lower bound.
- Frame 116 retains 49.25 mm right-foot error at right knee_z/ankle_z upper bounds.
- Frame 66 reaches its 300-evaluation budget: left/right errors are 61.77/7.59 mm.
  It is explicitly unsuccessful; do not call this a proven minimum or conclude
  global unreachability from a local optimizer.
- Removing the penalties introduces **1.151 mm sole-sole penetration at frame 33**,
  although production IK has none there. Two toe positions alone do not constrain
  full foot geometry, orientation, temporal smoothness or balance.
- Physical joint limits are checked separately; the largest tiny excess in the
  unregularized sample fits is about 4.90e-7 rad, not an authority to widen bounds.

Therefore the experiment does **not** justify simply setting regularization to zero
or replacing production IK with independently optimized frames. It motivates a
bounded, collision-aware, temporally coherent retargeter with explicit residual
acceptance, followed by fresh physical validation.

## Tests and reproducible artifacts

147 tests pass. New/modified diagnostics and tests pass Ruff E/F/I and formatting;
production modules compile and diff whitespace checks pass. Coverage includes
selection/provenance/weights, CLI and saved-config propagation, native Jacobian
finite differences, bounded unreachable fixtures, immutable inputs, distinct
control/physical limits, nonfinite input and root-jump localization.

`analysis_outputs/recovery_phase5_20261012/` contains:

- `selected_stages.json`: source IDs, full config, reference/code/input hashes,
  stage geometry, independent WRAP gates, dense off-knot collisions, root-lock
  windows and all nine counterfactual solver results.
- `corrected_cycle/summary.json`: numerical gates, geometric contact proxies,
  exact inverse-force decomposition, 12 native replays and source provenance.
- `corrected_cycle/xy0.35_lock1_contacts.csv`: per-sample contact measurements.

Windows Git Bash, repository root:

```bash
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
./.venv-win/Scripts/python.exe -m tools.check_retarget_stages --version consistent_v4 --reference-clip-ids 1 --dense --ik-fit --output analysis_outputs/recovery_phase5_20261012/selected_stages.json
./.venv-win/Scripts/python.exe -m tools.check_reference_contacts --baseline-only --recovery-physics-version consistent_v4 --reference-clip-ids 1 --clip-id 0 --output-dir analysis_outputs/recovery_phase5_20261012/corrected_cycle
```

Re-running replaces the explicitly named generated reports. None of these commands
loads a checkpoint, trains a policy or changes the model. `reference_clip_ids=[1]`
must be recorded with source files/order and hashes; loader IDs are not universal
identifiers across arbitrary changes to a motion library.

## Next step

1. Diagnose and correct periodic IK/stance-lock startup and support transitions;
   require bounded root/foot accelerations without hiding displacement as skating.
2. Fit the motion jointly with meaningful Cartesian residuals, contact geometry,
   joint limits and temporal/loop continuity. Keep old versions reproducible and
   compare a single controlled opt-in variant, not several unrelated changes.
3. Repeat these numerical, contact, inverse-authority and replay measurements.
   Only then decide whether a pure-tracker RL pilot is informative. Native replay
   is diagnostic evidence, not a theorem that no feedback policy could walk.

The three obsolete local branches remain archived/deleted as requested in phase 4.
Their remote branches, all checkpoints and the diagnostic stash remain untouched.
