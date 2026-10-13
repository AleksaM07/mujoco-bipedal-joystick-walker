# Phase 9B: prior attribution and an upstream clipped knee seed

Branch: `recovery/project-v2-imitation`. Production reference remains v6.
Diagnostic implementation: `091a90f`; measurements were rerun after this commit.
The diagnostic captures the existing reference once and compares native FK on
frozen pre-final-alignment geometry. There is no new prepared trajectory,
physics rollout or trained policy in this phase.

## What this isolates

Two mechanisms are now better localized:

1. The post-IK prior changes multiple joint amplitudes and partly destroys
   their compensation in Cartesian foot space. Restoring torso/pelvis or hip-Y
   reduces the frame-69 foot-path curvature; restoring ankle-Z does not.
2. The left-knee seed hits its upper limit before the IK solver. The loader's
   standing-pose offset makes that knee request positive angles in frames
   114-129. Clipping creates a plateau and a large nodal curvature change,
   which persists in modified form in the IK output.

These are not claims that either mechanism alone explains all failed walking
runs. Bounds enforce the model's physical convention; simply removing them is
not a valid repair.

## Frozen-geometry group comparison

Same v21 **no-arms**, same source clip 1 (source frames 79:213), 134 nodes,
same initial-aligned roots and orientations, same two normalized foot-marker
targets. This audit does not validate arm motion or full-body imitation.

All curvature entries below are root-relative sole-center XY **nodal second
differences** in world axes, not exact reference accelerations or required forces.

| Pose array | Right sole at frame 69, m/s^2 | Left whole-cycle peak, m/s^2 | Right whole-cycle peak, m/s^2 | Median right marker error, mm | Median left marker error, mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing prior | 61.636 | 94.097 | 226.237 | 93.469 | 81.558 |
| Raw IK | 14.188 | 106.395 | 231.103 | 67.679 | 65.456 |
| Restore torso/pelvis | 38.419 | 91.464 | 225.979 | 69.332 | 77.268 |
| Restore hip-Y pair | 38.126 | 105.925 | 229.998 | 90.919 | 79.096 |
| Restore ankle-Z pair | 62.126 | 94.424 | 227.607 | 92.889 | 77.934 |

The global peaks remain frames 9 (left) and 111 (right), not frame 69. The
relevant foot need not be the selected support at every peak. Removing the
whole prior is therefore not uniformly better: local curvature and marker fit
improve, but both whole-cycle maxima get worse. Group effects are conditional
and nonlinear; do not sum their apparent improvements.

The torso/pelvis group is the stronger candidate for a later complete-pipeline
test: it improves frame 69 and the right marker median without worsening the
whole-cycle center-curvature maxima. Nevertheless, the maximum left marker
error grows from 161.56 to 164.31 mm. This is not a production promotion.

No sole-sole penetration was reported at the 134 measured nodes in any variant.
That is not a dense/interpolated collision check or a complete-body collision
certificate. Joint and position-target violations above 1e-6 rad are absent;
raw IK/ankle restoration have at most 4.83e-7 rad of float32 physical-bound
excess. Existing target-bound clipping and centimeter-scale IK residuals remain.

## A seed defect upstream of the knee/hip spike

The knee reconstruction reproduces the captured loader seed **exactly** for
both knees. The standing-like bind frame is local frame 60. Its left-knee offset
is +0.2572725 rad; right offset is +0.0608497 rad.

| Local frame | Left knee before clipping, rad | Loader seed after clipping, rad | Raw IK output, rad |
| --- | ---: | ---: | ---: |
| 112 | -0.0457724 | -0.0457724 | -0.5291288 |
| 113 | -0.0240850 | -0.0240850 | -0.5321790 |
| 114 | +0.0010006 | 0 | -0.5317366 |
| 115 | +0.0265173 | 0 | -0.5523592 |
| 116 | +0.0495539 | 0 | -0.5735148 |

Positive left-knee targets violate the configured upper bound 0. They are
clipped for 16 frames, 114-129; the largest pre-clipping value is +0.1166828 rad.
The right knee has no upper-clipped frame in this cycle. At frame 114, the
left-knee nodal second difference changes:

`before clip +6.21 -> clipped seed -346.83 -> IK -303.34 rad/s^2`.

The normalized left foot target has zero root-relative Z second difference at
frame 114, while its XY components are about [+3.97, +5.26] m/s^2. Thus the
large knee-seed kink is not just the same sharp change in foot-target height.
There is separate target-height roughness nearby, including frame 115; neither
targets nor solver can be declared generally clean from this local result.

The IK objective explicitly regularizes to this clipped seed (weight 0.20) and
the preceding solution (0.08). It is plausible that this transmits the seed
kink into the fitted knee/hip motion, but the observed sequence and code do not
quantify causality or prove that the ten-iteration solver has converged. A
controlled seed intervention is needed before attributing the force peak to it.

## Next experiment and limits

Prioritize a **limit-preserving knee bind-offset** diagnostic: retain the source
angular variation but constrain its constant offset to the interval that fits
the complete knee sequence inside the existing bounds. Do not allow positive
knee motion or increase torque/control limits. If the source amplitude itself
cannot fit, reject the candidate rather than silently flatten it.

This would be a change to the IK seed, not to the physical limits or source
foot-marker targets. Rebuild the actual pipeline and compare closure, marker
fit, dense foot contacts, exact derivatives, inverse dynamics and native replay
against unchanged v6. Keep the prior unchanged for that experiment; otherwise
the two mechanisms are confounded. A separate torso/pelvis-prior ablation can
follow if useful. Do not combine them into an unexplained new preset.

Reward discrimination and a fixed feedback-tracker experiment remain on the
plan. Perfect open-loop replay is not a theoretical requirement for learning.
No long PPO should be justified solely by a smaller diagnostic peak.

## Evidence

All 222 regression tests pass. Source/input hashes in the completed report
match the measured files; native FK counterfactuals leave the runtime reference
unchanged.

`analysis_outputs/recovery_phase9_20261013/attribution.json` records joint arrays,
frozen geometry, per-foot paths and axis signs, named-marker residuals, limits,
local windows, loader knee reconstruction, input/source hashes and Git head.
The production reference digest is unchanged:
`2cf2f8aeab70f77ed02099fab2d53285b9b2d75f7fb0a42bec3e7ce44441211e`.

Reproduce in Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_stability_prior_attribution
```

No production code, reward, training configuration, policy, checkpoint, stash
or remote was changed in phase 9B.
