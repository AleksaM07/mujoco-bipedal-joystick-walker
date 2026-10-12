# Recovery phase 6C: cyclic IK helps geometry, not yet walking

Implementation: `0d362bb` (phase 6B), observer baseline: `a0971ad` (phase 6A).
Branch: `recovery/project-v2-imitation`. This separate phase records measurements
and adds v5 selection to the root-lock observer. It changes no production logic.

**Decision: keep v5 as an isolated experimental correction, not a validated
walking preset. Do not start the long PPO run on the strength of these results.**
The startup asymmetry is removed and acceleration extremes fall, but prescribed
motion remains dynamically problematic and open-loop replay does not improve.

## Controlled comparison

Both versions use the same v21 no-arms XML, CMU `02_01.bvh`, source frames
`[79:213)`, selected loader clip 1/runtime clip 0, 134 nodes, 1.108329 s cycle,
bounded WRAP closure, full source speed, XY scale 0.35 and stance locking enabled.
No reward, controller gain, force limit, collision mask or timestep changed.
The only production difference is v5's converged cyclic IK startup history.

| Metric | v4 control | v5 warmup |
| --- | ---: | ---: |
| Final-lock peak nodal XY second difference | 369.063 m/s^2 | 186.053 m/s^2 |
| Peak exact curve root acceleration | 1516.807 m/s^2 | 684.615 m/s^2 |
| Peak required free-root residual force | 123139.907 N | 57962.976 N |
| Samples exceeding any required actuator force limit | 91.17% | 93.42% |
| Samples exceeding any required actuator control target | 100% | 100% |
| Maximum sole-floor penetration | 10.511 mm | 8.722 mm |
| Penetration beyond the intentional 5 mm preload | 5.511 mm | 3.722 mm |
| Near-floor point speed median, left / right | 0.1273 / 0.1380 m/s | 0.1247 / 0.1304 m/s |
| Net root translation per cycle, X / Y | +1.37426 / -0.09930 m | +1.38703 / -0.09203 m |
| Mean reference XY travel speed | 1.24317 m/s | 1.25421 m/s |
| Pre-closure joint endpoint mismatch | 0.055607 rad | 0.043368 rad |
| Pre-closure root Z mismatch | 6.183 mm | 1.235 mm |
| Pre-closure joint velocity mismatch | 6.67286 rad/s | 1.28918 rad/s |
| Free-root trials completing three cycles | 0 / 12 | 0 / 12 |

Contact metrics sample 533 nodes/intermediate times; inverse dynamics uses 532
off-knot midpoints with exact analytic qacc. Near-floor point speeds use a 10 mm
height threshold, not force-based stance. Required inverse forces describe exact
prescribed motion with the current soft-contact model, not forces observed during
replay and not an optimized contact-force allocation. Nodal second differences
are distinct from analytic curve acceleration.

The smaller force peak does **not** imply improved overall feasibility: the
fraction of samples exceeding at least one force limit actually increases.
All replay force-clamp saturation fractions remain zero. The controller fails to
follow the trajectory; it does not realize the enormous inverse demands.

## All replay timings and phases, including regressions

Native free-root position replay requests three cycles (3.325 s), initializes
exact reference qpos/qvel once, and applies no root assistance afterward. Same
native gains, limits, floor and physical failure conditions in both versions.

| Start phase | Physics-rate v4 / v5 | 50 Hz current v4 / v5 | 50 Hz lookahead v4 / v5 |
| --- | ---: | ---: | ---: |
| 0.00 | 1.310 / 1.100 s | 1.330 / 1.110 s | 1.290 / 1.090 s |
| 0.25 | 1.120 / 1.120 s | 1.125 / 1.120 s | 1.105 / 1.110 s |
| 0.50 | 1.125 / 1.125 s | 1.120 / 1.125 s | 1.120 / 1.120 s |
| 0.75 | 1.000 / 0.950 s | 1.075 / 1.020 s | 0.905 / 0.895 s |

Phase-zero survival worsens by about 0.2 s. This must not be presented as a
walking improvement. These trials lack a learned balance controller and do not
establish that RL could never track a nearby trajectory; they diagnose this
particular reference/controller pairing.

## What the new trace isolates

V5's four warmup cycles converge to identical float32 joint poses (maximum cycle
difference 0). This is startup convergence, not exact equality of original source
endpoint targets. Bounded curve closure is still applied and its guards remain.

The original final-lock frame-1 reversal disappears: root X at frames 0/1/2 is
now 0, 0.0099598, 0.0198907 m. However, the changed initial IK pose also changes
the height-based support choice: v5 starts on the left foot, changes to the right
at frame 4, then back to the left at frame 76. V4 starts on the right and changes
to the left only at frame 76. No support labels were manually forced.

At a support switch the current lock inherits the preexisting input-root
increment; in continued support it instead cancels the selected foot-center
motion. These rules do not guarantee continuous root velocity across a switch.

The remaining measured v5 spikes align with those transitions:

- Peak nodal XY second difference: frame 4, 186.053 m/s^2.
- Largest absolute required motor force/torque sample: 12, source-relative
  fractional frame 3.125 (just before the frame-4 switch).
- Largest free-root residual force and root linear acceleration: sample 303,
  fractional frame 75.875 (just before the frame-76 switch).
- Final-lock X velocities into frames 75/76/77 are +0.7133, -0.2950, +1.2111 m/s,
  with an 8.333 ms node interval. The backward step is already in the nodes.
- The earlier, pre-IK frame-108 transition-to-flight spike is unchanged. Warmup
  does not alter that first alignment stage.

This supports investigating the support-handoff rule next. It does not isolate
it as the sole cause of all failures: centimeter IK residuals, stability-pose
blending, floor preload, soft contacts and limited actuation remain relevant.

## Numerical and compatibility checks

- V5 joint derivative RMSE is 9.13e-8 rad/s; root derivative RMSE is
  4.55e-8 m/s. Prepared joint endpoint pose/velocity gaps are zero. Exact reset
  matches runtime; multi-cycle consistency remains within tested tolerances.
- No sole-sole penetration in 533 contact samples or the separate 532 off-knot
  geometry samples. This is sampled evidence, not an exhaustive certificate.
- The fresh v4 control matches phase 5 **exactly** for digest, contact metrics,
  numerical checks, collision checks, inverse dynamics and all replay records.
- All 170 tests pass. Source hashes in the five new JSON reports match the
  measured implementation. The old phase-6A trace remains historical evidence
  of its own recorded source revision; it was not rewritten.

Selected v4 reference digest:
`12ae4d6c90078d151670c96a9c46079d9d957da708d6edb2a0fe387dc2d27d0f`.
Selected v5 reference digest:
`3ed1fac27b400954b8ac4aef847fa27e0302bfdd232e8aef0f7395300592b015`.

## Evidence and next isolated phase

Files under `analysis_outputs/recovery_phase6_20261012/`:

- `v4_control_stages.json` and `v5_stages.json`: stage geometry, closure and
  actual root-lock measurements; v5 includes a dense off-knot collision check.
- `v4_control_physics/summary.json` and `v5_physics/summary.json`: full numerical,
  contact, inverse-demand and 12-trial replay results; adjacent contact CSVs
  preserve the per-sample measurements for each version.
- `v5_root_lock_trace.json`: per-frame actual support/pose/root transitions.

Next: a controlled support-handoff/root-trajectory experiment while keeping the
IK objective fixed. First separate initial pre-IK locking from final locking in
a diagnostic ablation; then assess continuous handoffs against both root
acceleration and stance-point motion. Do not accept a smoother root curve if it
only hides sliding or causes new penetration. Recheck the periodic seam, limits,
contacts and identical multi-phase replay before promoting a training preset.
Keep implementation and measured-result commits separate.

No new PPO, checkpoint overwrite, branch deletion, stash operation or push.

## Reproduce (Windows Git Bash, repository root)

```bash
./.venv-win/Scripts/python.exe -m tools.check_reference_contacts --baseline-only --recovery-physics-version consistent_v4 --reference-clip-ids 1 --clip-id 0 --output-dir analysis_outputs/recovery_phase6_20261012/v4_control_physics
./.venv-win/Scripts/python.exe -m tools.check_reference_contacts --baseline-only --recovery-physics-version consistent_v5 --reference-clip-ids 1 --clip-id 0 --output-dir analysis_outputs/recovery_phase6_20261012/v5_physics
./.venv-win/Scripts/python.exe -m tools.check_root_lock_transitions --version consistent_v5 --output analysis_outputs/recovery_phase6_20261012/v5_root_lock_trace.json
```
