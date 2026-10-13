# Phase 8A: isolate the final floor-height envelope

Branch: `recovery/project-v2-imitation`. Control implementation: `86f3a5b`,
verified in phase 7C (`fcc723e`). This phase adds a diagnostic and its tests;
it does not modify production behavior or introduce a training preset.

## Question and intervention

The remaining largest sampled root acceleration is near frame 69.875. At frame
70, the lowest sole changes from right to left and the hard floor-height rule
reverses vertical nodal speed from -0.509 to +0.423 m/s. The existing 0.75 m/s
speed limiter permits both velocities; it does not remove their sharp change.

For two required root heights a and b, test this upper envelope:

`smooth_max(a,b) = (a+b+hypot(a-b,epsilon))/2`

The bias above the hard maximum is
`(hypot(a-b,epsilon)-abs(a-b))/2`, at most epsilon/2. Because only the height
difference matters, it can be computed directly from the two sole heights.
Default candidates are epsilon 0, 5 and 10 mm; zero is the unchanged control.

`tools/check_root_height_ablation.py` temporarily adjusts both measured foot
heights by this common bias ONLY inside final floor alignment. The original
minimum root height, forward/backward speed limiter, XY lock, velocities and
curve closure still execute. XY locking and its observer see REAL geometry,
including any changed support labels. Initial alignment, IK and joint poses
are untouched. Methods and scope flags restore on normal or exceptional exit.
This process-local patch is only for single-threaded diagnostic construction.

The pre-limiter envelope never lowers the hard required height, but this is NOT
a guarantee about final clearance: the limiter, interpolation and closure follow
it. Lifting may also weaken contacts or create flight. A reference that looks
smoother is not automatically a better physical target.

## Measurements

Each candidate rebuilds the actual selected-cycle reference: v21 no-arms XML,
source `02_01.bvh` clip 1 -> runtime clip 0, v6, bounded closure, pure imitation.
No source-cycle, gain, physics or reward setting changes.

The report includes:

- Reference digest and per-stage joint/root/orientation/target differences.
- Both actual root-lock traces and support transitions using real heights.
- Numerical derivative, seam, exact-reset and repeated-cycle checks.
- 533 contact-grid samples and 532 midpoint samples for sole clearance,
  penetration and collision checks, including both soles above the floor.
- Exact interpolant derivatives at the midpoint grid, inverse dynamics and
  contact-force decomposition; maxima are sampled, not continuum bounds.
- Twelve native free-root open-loop replays per candidate, without learned
  balance feedback. Replay success is not a theoretical prerequisite for RL.
- Config, input/source hashes, Git head, partial results and completion flag.

The feet are capsules. This experiment smooths the left/right minimum, not each
capsule's endpoint selection. PCHIP remains C1, not C2. Do not keep increasing
epsilon to hide different remaining discontinuities.

## Separate actuator-demand issue

The phase-7C maximum abdomen-Z demand occurs at frame 113.875, not the height
cusp. Its inverse force decomposes as approximately
`1342.71 + 12.87 - 1.925 - (-5981.67) = 7335.33` in scalar actuator-force units
(gear 1): inertia + bias - passive - constraint. The single left-sole contact
has 3.933 mm penetration, 6132.67 N normal and 4889.45 N tangential force.
Abdomen-Z acceleration itself is only 2.70 rad/s^2; left-hip-Z is 401.28 rad/s^2.
This is largely a transmitted contact wrench plus coupled inertia, not simply
a huge abdomen trajectory spike. Height smoothing need not solve it.

## Validation and reproduction

All 201 regression tests pass (17 new tests). The new cases cover smoothing
math, exact zero identity, unchanged inputs, per-environment/pass scope, real
geometry inside locks, keyword passthrough, exception cleanup and clearance
validation. A nonfinite second-foot value must not disappear behind the first
foot's lower finite value. Ruff E/F/I and formatting pass for both new files.

Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_root_height_ablation
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
```

The default output is
`analysis_outputs/recovery_phase8_20261013/height_ablation.json`.
Measured results and the accept/reject decision belong in phase 8B, separately
from this implementation commit. No PPO, policy overwrite, stash operation,
branch deletion or push.
