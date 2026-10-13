# Phase 10B: knee kink removed, full-reference candidate not promoted

Branch: `recovery/project-v2-imitation`. Diagnostic implementation: `8f956c4`.
Production remains opt-in `consistent_v6`; historical/default behavior is not
rewritten. This phase records results separately from diagnostic code.

## Decision and current project state

Do not promote the offset-only candidate as a better training reference. It
removes the targeted left-knee clipping kink, but worsens root acceleration,
free-root residual force and maximum floor penetration. All 24 native replay
trials fail. A locally cleaner seed is not a globally validated physical gait.

We have **not fixed all identified issues**. Distinguish these checkpoints:

| Area | Current evidence |
| --- | --- |
| Versioned timing, coordinate measurements, derivative consistency, toe aliases, cyclic IK initialization, support handoff | Implemented and regression-tested on the recovery path; old semantics preserved |
| Current retargeted motion and contacts | Still problematic: high accelerations, sliding/penetration and large exact force demands |
| Knee bind-offset clipping | Mechanism reproduced; this controlled correction removes its local kink but is not promoted because of regressions |
| Post-IK pose prior / root-height envelope | Diagnosed through isolated experiments; no universally better replacement yet |
| Reward discrimination and normalization | Original planned audit still outstanding; the reward still uses mean hinge position/velocity errors |
| Learned feedback tracker, natural gait, joystick | Not validated by these diagnostics; no new PPO run or walking-success claim |
| Full-body/arm motion | Not certified: these experiments use the fixed v21 no-arms model |

Passing unit tests checks code behavior, not whether the robot can walk.

## Controlled intervention

Keep source `02_01.bvh` clip 1 -> runtime clip 0, the same physical/control
bounds, prior, root alignment, gains, physics, reward config and curve gates.
Only the constant knee-seed offset is projected into its feasible interval.

Left knee: preferred +0.257272512 -> fitted +0.140589744 rad, a change of
-0.116682768 rad. The raw angular variation is preserved without per-frame
clipping. Right knee's offset already fits and its seed positions/velocities
are unchanged. The loop heuristic remains WRAP. No height smoothing is used.

At local frame 114, left-knee nodal second differences are:

| Stage, rad/s^2 | V6 | Offset candidate |
| --- | ---: | ---: |
| Loader seed | -346.827 | +6.209 |
| Raw IK | -303.338 | +18.041 |
| Prepared reference nodes | -303.422 | +17.966 |

This demonstrates removal of the targeted kink through the actual pipeline.
It does not separately identify the contribution of each IK penalty, because
the seed also changes upstream floor alignment and downstream coupled poses.

## Whole-trajectory outcomes

All maxima below are sampled, not certified continuous extrema. Inverse
dynamics uses the existing soft-contact model, not optimized contact forces.

| Measurement | V6 | Offset candidate |
| --- | ---: | ---: |
| Peak exact root linear acceleration, m/s^2 | 216.962 | 281.912 |
| RMS root linear acceleration, m/s^2 | 39.312 | 42.769 |
| Peak free-root residual force, N | 16599.387 | 21594.337 |
| Peak free-root residual torque, Nm | 9604.617 | 9192.267 |
| Samples exceeding any actuator force limit | 91.917% | 92.857% |
| Samples exceeding any required control range | 100% | 100% |
| Largest actuator scalar-force demand | 7335.333 | 6276.282 |
| Maximum penetration, 533-point contact grid, mm | 8.722 | 9.830 |
| Maximum penetration, 532-point midpoint grid, mm | 8.749 | 9.861 |
| Three-cycle replay completions | 0/12 | 0/12 |

Both largest actuator demands are abdomen-Z, whose configured scalar-force
range is +/-180. The maximum's location moves from frame 113.875 to 0.125; this
does not prove a new position/velocity discontinuity at the loop seam.

The root-acceleration peak remains frame 69.875. Its XYZ components change
from `[-87.49,-28.31,+196.51]` to `[-87.25,-29.09,+266.49] m/s^2`, worsening the
remaining vertical peak. The final support selector now starts with its first
foot and switches only at frame 76; v6 also switched at frame 4. Input changes
can affect the whole contact sequence, not just the seed's local knee kink.

No sole-sole penetration or geometric flight is observed on either sampling
grid. The intentional 5 mm preload is included in floor-penetration figures.
These findings neither certify continuous clearance nor all-body collisions.

Prepared-node diagnostic marker residual medians, right/left, are
94.14/85.13 mm for v6 versus 98.07/83.28 mm for the candidate. Maximum left
residual improves 163.43 -> 144.77 mm, while median right residual worsens.
These targets follow pipeline root shifts and are comparators, not a new IK
solve after closure. The overall marker fit is still centimeter-scale.

## Native replay details

Requested horizon: 3.325 s (three cycles). No root assistance after exact
reference initialization; no learned balance feedback.

| Timing | Start phase | V6 duration, s | Candidate duration, s |
| --- | ---: | ---: | ---: |
| Physics rate | 0.00 | 1.095 | 1.095 |
| Physics rate | 0.25 | 1.120 | 1.115 |
| Physics rate | 0.50 | 1.125 | 1.070 |
| Physics rate | 0.75 | 0.945 | 1.070 |
| 50 Hz current | 0.00 | 1.105 | 1.115 |
| 50 Hz current | 0.25 | 1.120 | 1.115 |
| 50 Hz current | 0.50 | 1.135 | 1.080 |
| 50 Hz current | 0.75 | 1.015 | 1.165 |
| 50 Hz lookahead | 0.00 | 1.085 | 1.095 |
| 50 Hz lookahead | 0.25 | 1.110 | 1.100 |
| 50 Hz lookahead | 0.50 | 1.120 | 1.070 |
| 50 Hz lookahead | 0.75 | 0.895 | 0.915 |

Some starts last longer and others shorter. None completes the requested
three cycles; this does not establish improved walking. Actual force-clamp
saturation remains zero because replay departs from the prescribed state.
Failure of open-loop playback does not prove no feedback policy can learn.

## Invariants and remaining work

The raw source targets, roots and quaternions are identical; only left-knee
seed arrays change. All captured orientations remain identical. Initial root
alignment shifts by up to 0.179 m and changes aligned world targets, as expected
from reprocessing a changed seed. Final prepared root differences are up to
13.38 mm. Do not describe the final motion as a one-joint-only change.

Numerical curve checks still pass: exact reset, joint/velocity endpoint closure,
joint derivative RMSE 7.08e-8 rad/s, root derivative RMSE 3.90e-8 m/s, and the
existing repeated-cycle tolerances. Both candidates therefore pass numerical
consistency while remaining poor physical targets.

Next priority is the already-planned **reward discrimination audit** on the
frozen v6 control: exact reference, standing, wrong phase, swapped leg poses
and sliding states, with every reward component and normalization visible.
This is independent of whether the current open-loop replay survives. It will
separate target-quality problems from a learning objective that may reward
standing or wrong motion too generously. No reward coefficient change should
be made without those comparisons.

Retarget/contact improvements remain a parallel open issue. Keep the measured
knee correction as an experimental candidate rather than combining it blindly
with height smoothing or prior removal. A subsequent fixed feedback-tracker
pilot and serious training budget need explicit success metrics; neither a
short pilot nor a higher scalar reward alone establishes walking.

## Evidence

All 248 regression tests pass (26 new). The final report is
`analysis_outputs/recovery_phase10_20261013/knee_offset.json`, rerun after the
diagnostic commit and carrying source/input hashes. The disabled control
reproduces the committed v6 reference and measurements.

Candidate digest:
`bef3431b3e425505a7a2343e33e14257fe6888f7a5fe53c5ed415cc479f3b545`.

Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_knee_offset_ablation
```

No production code, preset, reward, checkpoint, stash or remote changes in
phase 10B. No PPO run started.
