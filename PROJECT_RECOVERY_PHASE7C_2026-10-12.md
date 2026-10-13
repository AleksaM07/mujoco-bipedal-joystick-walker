# Phase 7C: verify the production handoff fix and locate remaining acceleration

Implementation `86f3a5b`, experiment `55ab5b7`, branch
`recovery/project-v2-imitation`. This commit records results separately from code.

The production `consistent_v6` reference exactly reproduces the phase-7A
handoff-only candidate: reference digest, contact measurements, numerical checks,
collision checks, inverse-dynamics values and all 12 native replay records match.
The diagnostic report additionally labels its midpoint sampling. Switching the
contact classification back to historical float32 precision changes none of
these selected-cycle results.

## What improved, and what did not

| Selected-cycle measurement | v5 | v6 |
| --- | ---: | ---: |
| Peak exact root linear acceleration | 684.615 m/s^2 | 216.962 m/s^2 |
| Peak required free-root residual force | 57962.976 N | 16599.387 N |
| Peak nodal root XY second difference | 186.053 m/s^2 | 61.636 m/s^2 |
| Required actuator force-limit exceedance | 93.42% | 91.92% |
| Required actuator control-range exceedance | 100% | 100% |
| Maximum sole-floor penetration | 8.722 mm | 8.722 mm |
| Near-floor left point-speed maximum | 1.9400 m/s | 0.8253 m/s |
| Near-floor right point-speed maximum | 1.8711 m/s | 0.9555 m/s |
| Successful three-cycle native replays | 0/12 | 0/12 |

Frame-76 root X velocity is now +1.0526 m/s, between the neighboring +0.7133
and +1.2111 m/s intervals. The old -0.2950 m/s backward interval is gone. The
frame-4 interval is likewise +1.2104 instead of +0.1302 m/s. Selected support
transitions remain frames 4 and 76; the rule changes their anchors, not labels.
Maximum stationary-center recurrence error, including supported handoffs, is
7.12e-6 m/s. Joint poses, root Z and orientations are unchanged throughout.

Both versions remain imperfect physical targets. Height labels are geometric
proxies, and foot-center anchoring does not eliminate rotation-induced point
motion. At the broader 35 mm threshold, point-speed peaks remain above 5 m/s;
the smaller values in the table refer specifically to the 10 mm near-floor
subset. The floor penetration includes a 5 mm intentional preload.

## Native replay, all timings

| Start phase | Physics-rate v6 | 50 Hz current v6 | 50 Hz lookahead v6 |
| --- | ---: | ---: | ---: |
| 0.00 | 1.095 s | 1.105 s | 1.085 s |
| 0.25 | 1.120 s | 1.120 s | 1.110 s |
| 0.50 | 1.125 s | 1.135 s | 1.120 s |
| 0.75 | 0.945 s | 1.015 s | 0.895 s |

Requested horizon is 3.325 s. These are unassisted free-root, open-loop position
controls after exact reference initialization, with unchanged native gains.
Differences from v5 are only about 0-10 ms and provide no meaningful evidence of
improved walking. Actual actuator force-clamp saturation remains zero, despite
large exact inverse demands: replay departs from the prescribed motion.

Open-loop success is not a theoretical prerequisite for learning a feedback
policy. These results show that the current reference/controller pairing remains
poor; they do not prove that no policy can walk. V6 is retained as a tested local
bug correction, with physical feasibility and training readiness still open.

## Remaining peaks

The largest inverse free-root force and linear acceleration now occur at
fractional frame 69.875, away from the support switches. Exact root acceleration
there is approximately `[-87.49, -28.31, +196.51] m/s^2`; vertical motion dominates.
The largest nodal XY second difference is at frame 69 during continued right-foot
support. Root Z remains identical to v5; this handoff fix could not resolve its
remaining roughness.

The native trace exposes a separate floor-height cusp at frame 70. Root Z at
frames 68-72 is `[1.479289889, 1.474909306, 1.470669866, 1.474192977,
1.476799011] m`; vertical nodal speed changes from -0.5087 to +0.4228 m/s.
The geometrically lowest sole changes from right to left at frame 70, while
the hysteretic XY support selector keeps the right foot until frame 76.
At frame 69, left/right minimum heights are approximately +3.670/-5.000 mm;
at frame 70 they are -5.000/-3.764 mm. Selecting the lowest sole independently
at each frame makes the required root-height curve change slope abruptly.
The existing Z speed limiter bounds velocity but does not smooth this cusp.
This is a separate mechanism from the XY handoff repaired in v6.

The largest required actuator scalar force and free-root torque occur separately
at fractional frame 113.875. Abdomen-Z demand reaches about 7335 in scalar
actuator-force units, against its configured +/-180 range. These demands assume
exact qpos/qvel/qacc and the model's current soft contacts; no contact allocation
optimization was performed. Raising limits based on these peaks is not justified.

Next analysis should isolate this floor-height cusp and the different
actuator-demand peak. A candidate root-height
change must be checked for penetration and flight, and any pose change for marker
error, joint limits and sole overlap. The reward discrimination audit and a fixed
pure-tracker training setup from the original diagnosis are still outstanding.

## Numerical checks and evidence

- Joint derivative RMSE 9.13e-8 rad/s; root derivative RMSE 4.40e-8 m/s.
- Prepared joint position/velocity endpoint gaps zero; exact reset and repeated
  cycle checks pass at the existing tolerances.
- No sole-sole penetration in 533 contact samples and 532 off-knot samples.
- Production root-lock capture explicitly records initial `False`, final `True`
  for the requested contact-history option.
- 184 regression tests pass; no further production changes in this phase.

Artifacts under `analysis_outputs/recovery_phase7_20261012/`:

- `v6_stages.json`: real pipeline geometry and root-lock stages.
- `v6_root_lock_trace.json`: support labels and frame-level root/foot data.
- `v6_physics/summary.json`: numerical checks, contacts, forces and all replay trials.
- `v6_physics/xy0.35_lock1_contacts.csv`: per-sample contact measurements.

Source hashes in the new reports match the measured implementation. Phase-7A
artifacts remain immutable historical measurements, with their own source hashes.
V6 digest:
`2cf2f8aeab70f77ed02099fab2d53285b9b2d75f7fb0a42bec3e7ce44441211e`.

Reproduce in Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_reference_contacts --baseline-only --recovery-physics-version consistent_v6 --reference-clip-ids 1 --clip-id 0 --output-dir analysis_outputs/recovery_phase7_20261012/v6_physics
./.venv-win/Scripts/python.exe -m tools.check_root_lock_transitions --version consistent_v6 --output analysis_outputs/recovery_phase7_20261012/v6_root_lock_trace.json
```

No new PPO, policy overwrite, branch deletion, stash operation or push.
