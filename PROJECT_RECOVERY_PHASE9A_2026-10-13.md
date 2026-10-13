# Phase 9A: attribute Cartesian damage and reconstruct knee-seed clipping

Branch: `recovery/project-v2-imitation`. Production control is unchanged v6.
This commit adds the diagnostic and tests; measurements are recorded separately
in phase 9B. No production behavior, preset, reward or policy is modified.

## One capture, five counterfactuals

`tools/check_stability_prior_attribution.py` captures the real reference once.
It freezes the normalized-IK-input roots, orientations and marker targets, then
compares these joint arrays with native MuJoCo forward kinematics:

1. Existing post-IK stability prior.
2. Entire raw IK solution.
3. Existing prior with raw-IK abdomen/pelvis XYZ restored (six joints).
4. Existing prior with raw-IK hip-Y restored (two joints).
5. Existing prior with raw-IK ankle-Z restored (two joints).

Restoration is global across the cycle, not a splice around a problematic
frame. Groups use joint names rather than assumed actuator order. The report
checks that changes outside these groups are no larger than 1e-6 rad; alpha=1
still has float32 roundoff in `neutral + (pose-neutral)`. It records the actual
per-joint differences, not just the classification.

All variants use the same frozen geometry; there is no new IK solve, floor
alignment, XY locking, closure, dynamics or replay. A locally improved foot
path is a conditional nonlinear effect, not an additive attribution or a
certificate that a new prepared reference would work.

## Measurements and boundaries

- Both capsule sole centers relative to root translation in world axes.
- Interior-node XY second differences, with explicit center-frame indexing.
  These are not the final curve's exact accelerations. Raw endpoints are not
  wrapped when computing differences.
- Capsule longitudinal-axis Z components, to inspect endpoint-selection changes.
- Named-marker residuals against the same normalized IK targets.
- Sole-sole penetration under the model's existing contact masks. Absence of
  reported contact means zero censored depth, not positive clearance. This does
  not check every body pair for self-collision.
- Physical joint and position-target bounds, not required dynamic controls.
- Whole-cycle arrays plus explicit windows 64-80 and 108-120.
- Seed joint curvature and root-relative normalized-target curvature.

The production reference digest is checked before/after. Every variant is a
detached pose copy. Source/input hashes, Git head, selected original clip and
source frame range make the measurements reproducible. Partial output remains
marked incomplete if a run fails.

## Upstream knee-seed reconstruction

The diagnostic also reproduces the loader's existing knee calculation from
the selected BVH segment: hinge conversion, standing-like bind frame, neutral
offset, then clipping to the original control limits. It requires the resulting
clipped arrays to match the captured loader seed within 1e-6 rad and records
pre/post-clipping arrays and affected frames.

This can distinguish a seed mapping/clipping kink from a defect introduced by
the IK solve. It does not prove how much each IK penalty contributes to the
later actuator demand; that needs a controlled pipeline experiment.

## Usage

All 222 regression tests pass, including 21 new tests. Coverage includes
named-column restoration, finite and real input validation, integer-safe second
differences, native capsule FK under root rotation/translation, center-frame
indexing, non-mutation, and independent knee-offset/clipping fixtures. Ruff
E/F/I and formatting pass for the diagnostic and tests.

Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_stability_prior_attribution
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
```

Default output:
`analysis_outputs/recovery_phase9_20261013/attribution.json`.

No PPO, policy overwrite, stash operation, branch deletion or push.
