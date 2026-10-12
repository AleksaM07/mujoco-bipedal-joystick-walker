# Phase 7A: isolate pre-IK locking and support handoff

Parent `80fe52e`, branch `recovery/project-v2-imitation`. This commit contains a
scoped diagnostic experiment and its evidence. Production v5 is unchanged.

## The error and isolated candidate

During continued support, the root cancels the selected sole-center movement.
At a support change, the old rule instead inherits the input-root increment,
even when the incoming sole was already in contact in the previous frame.
Thus a known previous world-space anchor is discarded for one interval.

The candidate preserves that previous anchor whenever the incoming foot is
height-classified as contacting at both interval endpoints. With root XY `r`
and FK foot position relative to root `f`, it uses
`r[i] = r[i-1] + f[i-1, incoming] - f[i, incoming]`.
The candidate retains the old displacement rule for genuine landings and flight.
First frame, root Z, joint poses, orientations, support threshold and hysteresis
are unchanged. This constrains a center, not the rotating sole's material points.

The 2x2 experiment separately disables initial pre-IK XY locking and applies the
new final-pass handoff. All four variants use the same selected v5 cycle, v21
no-arms model, root scale, source speed, bounded closure and native controller.
Overrides exist only during diagnostic construction and restore on exceptions.

## Measured comparison

| Variant | Peak exact root acceleration | Peak inverse free-root force | Any required actuator force limit exceeded |
| --- | ---: | ---: | ---: |
| v5 baseline | 684.615 m/s^2 | 57962.976 N | 93.42% |
| Skip initial locking only | 359.518 m/s^2 | 30800.429 N | 92.48% |
| Fix final handoff only | 216.962 m/s^2 | 16599.387 N | 91.92% |
| Both changes | 216.962 m/s^2 | 16599.387 N | 91.92% |

For handoff-only, peak final-lock nodal XY second difference drops from 186.053
to approximately 61.64 m/s^2. Every joint node at every captured pipeline stage
is bit-identical to baseline. Floor penetration remains 8.722 mm, including
the intentional 5 mm preload. No sole-sole penetration is detected on the 533
contact samples or 532 off-knot samples.

At the 10 mm near-floor height threshold, material-point speed statistics:

| Sole | Baseline median / p95 / max | Handoff median / p95 / max |
| --- | ---: | ---: |
| Left | 0.1247 / 0.5615 / 1.9400 m/s | 0.1226 / 0.5385 / 0.8253 m/s |
| Right | 0.1304 / 0.4429 / 1.8711 m/s | 0.1304 / 0.3973 / 0.9555 m/s |

The improvement therefore does not come from moving the feet farther through
the floor or increasing the measured peak point speeds. It is still far from
a full stance/no-slip or dynamic-feasibility certificate.

All 48 native three-cycle replay trials fail. At 50 Hz current-target timing,
phases 0/.25/.5/.75 survive:

- Baseline: 1.110, 1.120, 1.125, 1.020 s.
- Skip initial: 1.110, 1.120, 1.130, 1.020 s.
- Handoff only: 1.105, 1.120, 1.135, 1.015 s.
- Both: 1.105, 1.120, 1.135, 1.015 s.

These small changes do not demonstrate more stable walking. Required inverse
controls still exceed ranges throughout the sampled exact trajectory. Actual
replay force-clamp saturation remains zero. Open-loop failure is not proof that
a learned feedback policy cannot track a nearby trajectory.

## Why the initial lock is not promoted as a second fix

Pre-IK root and target translations cancel in Cartesian residuals and joint
Jacobians. Removing that lock changes solved joint poses by only float rounding
(maximum 1.1921e-7 rad here). The initial trajectory affects final root motion
primarily through the arbitrary old handoff increment. Once the handoff is
fixed, removing the first lock adds no measured peak-dynamics benefit.

Decision: promote only the final handoff rule to an explicit new version, with
the historical path untouched. Keep the initial-lock ablation as evidence.
The next implementation commit must reproduce this candidate and retain all
numerical closure guards. Further physical defects remain before long training.

## Validation and reproducibility

Eleven small native MuJoCo tests cover stationary-center handoff, genuine
landing/flight, legacy continuous support, translation and yaw transforms,
padding, immutability, independent experiment flags, per-environment pass scope,
and exception restoration. The full regression suite contains 181 tests.

The baseline exactly reproduces phase-6C contact metrics, inverse-dynamics values,
numerical checks, collision checks and all replay records. The report preserves
input/source hashes, every condition, stages' joint differences, full force
decompositions and all 48 replay results:
`analysis_outputs/recovery_phase7_20261012/ablation.json`.

The candidate reconstructs the root recurrence using native FK and float64
height decisions; tiny rounding differences from the old float32 lock are
possible in addition to the deliberate handoff correction. Old versions do not
use it. Only one selected source cycle has been physically measured.

Windows Git Bash reproduction:

```bash
./.venv-win/Scripts/python.exe -m tools.check_root_lock_ablation
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
```

No PPO, checkpoint migration, model-limit edits or push.
