# Phase 11B: measured weak reward discrimination on frozen v6

Implementation: `896908c` (phase 11A). Branch: `recovery/project-v2-imitation`.
No production reward change and no PPO run in this phase.

## Decision

The reward audit identifies a second concrete problem beyond reference physics:
the current pose/velocity terms are weakly discriminative with their existing
mean normalization and scales. A static neutral pose retains substantial reward;
instantaneous knee-velocity errors with large near-floor foot-speed changes
barely affect the objective.

This is evidence about the objective, not a proof of why any particular past
policy stopped moving. Exact reference still scores highest. Dynamic stability,
state visitation, optimizer behavior, action scaling and cumulative returns are
not measured here. Mean normalization is not inherently wrong, but copying
coefficients without matching the error definition changes their meaning.

The user explicitly requested moving to real training rather than extending
diagnostics. Next: a fresh 80M-step pure tracker on this frozen v6 control,
with the existing mean-normalized reward recorded explicitly. That establishes
an actual learned baseline; reward-normalization A/B remains a later comparison,
not another prerequisite. Do not combine the baseline with the rejected knee
candidate, new priors or joystick rewards. The offline sum comparison below is
evidence, not an adopted training preset.

## Fixed experimental definition

Use the same v21 **no-arms** XML, `02_01.bvh`, selected source clip 1/runtime
clip 0, bounded WRAP closure and production `consistent_v6`. The processed
reference digest remains:

`2cf2f8aeab70f77ed02099fab2d53285b9b2d75f7fb0a42bec3e7ce44441211e`.

Score 128 uniformly spaced midpoint phases, 13 cases each: **1,664 native
MuJoCo snapshots**. Native FK/cvel/Jacobians are computed from full qpos/qvel;
actual production JAX reward and termination methods score the snapshots. No
physics integration, root assistance, learned controller or optimizer is run.
The scoring clock always stays at the original phase; standing fallback is
disabled. Every position/control-limit excess is zero on this sampling grid.
No sampled case triggers the configured physical/pose/motion termination.

The pure branch's actual step reward is three times imitation total for these
nonterminal states. Physical failure would instead return -25. Pose termination
is disabled in this fixed config. Mixed task-plus-imitation reward is out of
scope and must not be assigned these results.

## Measured rewards

| Snapshot case | Mean imitation, max ~1 | Observed imitation min..max | Mean actual pure reward, max 3 |
| --- | ---: | ---: | ---: |
| Exact reference | 0.999999 | 0.999999..0.999999 | 2.999998 |
| Configured neutral standing pose, zero velocity | 0.785682 | 0.747780..0.848238 | 2.357047 |
| Frozen phase-zero reference pose, zero velocity | 0.800841 | 0.742884..0.922142 | 2.402524 |
| Current reference pose, all velocity zero | 0.917707 | 0.901070..0.949197 | 2.753120 |
| Quarter-cycle wrong phase | 0.806860 | 0.750909..0.875203 | 2.420579 |
| Half-cycle wrong phase | 0.801513 | 0.736998..0.917388 | 2.404539 |
| Swapped sagittal leg pose/velocity channels | 0.859956 | 0.832008..0.985834 | 2.579869 |
| Rigid root XY translation control | 0.999999 | 0.999999..0.999999 | 2.999998 |
| Extra world-X root velocity, +0.5 m/s | 0.977880 | 0.977879..0.977880 | 2.933639 |
| Either knee velocity, +2 or -2 rad/s (four cases) | 0.999777 | approximately constant | 2.999332 |

The standing case is the configured neutral `_init_q`, not necessarily the
identical fallback reset (which additionally overwrites root height). Frozen
standing poses are candidate kinematic states, not certified stable balances.
The stopped-current-pose and velocity-only cases are instantaneous probes;
their phase sequences are not asserted to be physically realizable motions.

Swapping hip-Z/knee-Z/ankle-Z is explicitly not a full anatomical reflection.
The small exact-state deficit from one comes from the existing clipped
quaternion-distance calculation, not a significant FK inconsistency.

## Why the velocity probe matters

With 18 actuated hinges, the production definitions are:

`pose = exp(-0.25 * mean((q - q_ref)^2))`

`velocity = exp(-0.01 * mean((dq - dq_ref)^2))`

Their normalized total weights are 0.50 and 0.10. Root pose/velocity and key
positions have weights 0.15/0.10/0.15. Key position uses a sum of coordinate
squared errors, unlike the hinge means. Pure mode has no explicit foot-speed
or contact-conditioned slip term.

A single knee velocity error of 2 rad/s produces mean squared error 4/18, so
the velocity component is `exp(-0.01 * 4/18) = 0.997780`. With the other
components effectively unchanged, total reward loses only about **0.000222**
(0.0222% of the full imitation scale). Native measurements show this is not
merely an imperceptible foot-speed perturbation:

| Geometrically near-floor samples | Left knee probe | Right knee probe |
| --- | ---: | ---: |
| Affected foot sample count, of 128 | 63 | 71 |
| Median change in foot horizontal velocity vector magnitude, m/s | 0.978383 | 0.980856 |
| Maximum change in that vector, m/s | 1.001662 | 0.998687 |
| Baseline median horizontal point speed, m/s | 0.131821 | 0.134495 |
| Median speed with +2 rad/s probe, m/s | 0.891487 | 0.867308 |
| Median speed with -2 rad/s probe, m/s | 1.091171 | 1.091977 |

This is material-point velocity at the native lowest sole point, not finite
difference of the moving minimum point. Near floor means height <=1 cm,
including penetration; it is not force-labeled stance. The velocity change is
the norm of the vector difference, not the difference between two speed norms.
No claim is made that a controller can sustain these exact probe states.

## Pose discrimination and normalization counterfactual

Standing's mean pose component is **0.990649**, even though its mean key-position
component is 0.153110 and root-velocity component 0.194593. The high-weight pose
term contributes about 0.4953 out of the available 0.5. Zero joint velocities
still receive a mean velocity component of 0.982481. These measurements explain
why the per-state score stays relatively high despite large tracking omissions.

Replacing only hinge means with sums in an **offline algebraic rescore** gives:

| Case | Current mean-hinge total | Offline sum-hinge total |
| --- | ---: | ---: |
| Neutral standing | 0.785682 | 0.690463 |
| Frozen phase zero | 0.800841 | 0.701484 |
| Stopped current reference pose | 0.917707 | 0.894235 |
| Quarter-cycle wrong phase | 0.806860 | 0.663932 |
| Half-cycle wrong phase | 0.801513 | 0.636301 |
| Swapped sagittal legs | 0.859956 | 0.715482 |

This broadens separation but does not establish a sufficient objective: the
single-knee velocity probe would still retain approximately 0.99608 total.
Summed hinge errors are also not automatically equivalent to a body quaternion
error used by another implementation. Do not label this a verified DeepMimic
replication or choose new coefficients from one attractive scalar result.

Absolute XY translation intentionally leaves the total unchanged (measured
max difference zero); local-root imitation ignores global path origin. That
invariance is not a slip bug. The added root velocity is penalized separately.
Reference key positions are reconstructed from the processed pose's FK, so
an exact score does not prove fidelity to the original BVH markers.

## What remains red and the next controlled step

1. Reference/contact physics remain unresolved, as documented in phases 7-10.
   The phase-10 knee candidate is not promoted. Numerical consistency is not
   physical feasibility, and this reward audit does not change that finding.
2. Reward discrimination is now measured, not fixed. The next real training
   run deliberately retains this known objective to establish a baseline.
   Any later normalization change needs an explicit saved option and separate
   comparison, keeping reference/model/actions fixed. Do not silently patch a run.
3. Start the isolated learned feedback tracker with actual rollout metrics:
   multi-cycle survival, phase/pose errors, root speed and near-floor slip,
   rather than scalar reward alone. Short smoke/pilot runs validate execution;
   they do not replace an adequately budgeted learning experiment.

Open-loop playback failure remains a diagnostic warning, not a theorem that
feedback RL cannot learn. Full-body arms and joystick control are still not
validated by these no-arms, pure-imitation checks. **Not all bugs or project
problems have been solved.**

## Evidence and validation

`analysis_outputs/recovery_phase11_20261013/reward_discrimination.json` contains
per-phase rows, complete reward components, geometry, config, runtime versions,
Git HEAD and source/input hashes. It is generated again after the implementation
commit. All 274 regression tests pass; targeted Ruff and format checks pass.

Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_reward_discrimination
```

No production code, reward, preset, checkpoint, stash or remote mutation in
phase 11B. The audit itself launches no training; the following operational
step starts the user-requested 80M baseline separately.
