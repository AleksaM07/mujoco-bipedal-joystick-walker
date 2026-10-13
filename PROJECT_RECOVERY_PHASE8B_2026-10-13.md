# Phase 8B: height smoothing helps one defect, but does not produce walking

Diagnostic implementation: `b50c449`, branch `recovery/project-v2-imitation`.
Production remains `consistent_v6`; no new behavior version or training preset
is promoted. This phase records the controlled measurements separately.

## Decision

Keep the 5 mm envelope as an experimental candidate, not a solved reference.
It reduces the height cusp and largest sampled root acceleration without
sampled geometric flight. The 10 mm candidate offers little additional
acceleration reduction and introduces a small positive gap beneath both feet.
Neither candidate materially improves native replay or resolves the high
actuator demands. Do not increase smoothing or actuator limits blindly.

All 36 native replay trials fail. No PPO was started. Open-loop reference
playback is not a learned feedback controller, and these failures do not prove
that RL cannot learn a nearby feasible gait.

## Controlled results

Same v21 **no-arms** XML, source clip 1, cycle duration 1.108329 s, gains,
contacts, physics and reward configuration in all conditions. Epsilon is the
smooth-envelope width, not the amount the whole robot is raised.

| Measurement | Unchanged v6 | Epsilon 5 mm | Epsilon 10 mm |
| --- | ---: | ---: | ---: |
| Largest sampled exact root acceleration, m/s^2 | 216.962 | 160.662 | 158.264 |
| Largest sampled free-root inverse residual force, N | 16599.387 | 11991.173 | 10373.790 |
| Largest sampled free-root inverse residual torque, Nm | 9604.617 | 9583.216 | 9517.085 |
| Samples exceeding any actuator force limit | 91.917% | 91.917% | 91.165% |
| Samples exceeding any actuator control range | 100% | 99.812% | 99.812% |
| Maximum penetration on the 533-point contact grid, mm | 8.722 | 7.966 | 6.720 |
| Maximum penetration on the 532-point midpoint grid, mm | 8.749 | 7.992 | 6.711 |
| Highest lowest sole on the 532-point midpoint grid, mm | -2.578 | -2.109 | +0.115 |
| Maximum sampled absolute root Z speed, m/s | 0.830 | 0.844 | 0.802 |
| Three-cycle native replay completions | 0/12 | 0/12 | 0/12 |

Penetration includes the intentional 5 mm preload. The global maximum gets
smaller, but the right foot's maximum on the contact grid gets slightly worse:
6.302 -> 6.388 -> 6.459 mm. Smoothing is therefore not a uniform clearance
improvement. The original limiter constrains nodal steps; it does not guarantee
a 0.75 m/s bound on the final interpolant's derivative, even for unchanged v6.

The 10 mm candidate has both soles above the floor at one contact-grid sample
(0.067 mm gap) and one midpoint sample (0.115 mm gap). This is a small geometric
gap, not evidence that this alone causes falling. Five millimeters has no such
sample in either grid. These finite grids do not certify continuous clearance.

The 5 mm candidate lowers the largest sampled acceleration by about 26%, but
the RMS decreases only from 39.31 to 37.85 m/s^2. The largest remaining
acceleration moves to frame 69.125, approximately
`[111.84, 111.19, -30.68] m/s^2`: it is now predominantly horizontal. At 10 mm,
the horizontal components are the same while Z becomes -13.28 m/s^2. This
explains the diminishing return from further height smoothing.

## Replay results, not policy evaluation

Requested duration is three cycles, 3.325 s. The entries are time until the
unchanged height/uprightness termination, in seconds.

| Target timing | Start phase | V6 | 5 mm | 10 mm |
| --- | ---: | ---: | ---: | ---: |
| Physics rate | 0.00 | 1.095 | 1.095 | 1.095 |
| Physics rate | 0.25 | 1.120 | 1.120 | 1.125 |
| Physics rate | 0.50 | 1.125 | 1.125 | 1.125 |
| Physics rate | 0.75 | 0.945 | 0.950 | 0.965 |
| 50 Hz current | 0.00 | 1.105 | 1.105 | 1.100 |
| 50 Hz current | 0.25 | 1.120 | 1.120 | 1.125 |
| 50 Hz current | 0.50 | 1.135 | 1.135 | 1.130 |
| 50 Hz current | 0.75 | 1.015 | 1.020 | 1.025 |
| 50 Hz lookahead | 0.00 | 1.085 | 1.085 | 1.085 |
| 50 Hz lookahead | 0.25 | 1.110 | 1.110 | 1.115 |
| 50 Hz lookahead | 0.50 | 1.120 | 1.125 | 1.125 |
| 50 Hz lookahead | 0.75 | 0.895 | 0.900 | 0.905 |

Changes are -5 to +20 ms, at a 5 ms native timestep. They do not demonstrate a
meaningful walking improvement. All actual actuator force-clamp saturation
fractions remain zero; the replay deviates from the exactly prescribed motion.

## Isolation and numerical checks

- The zero-epsilon reference digest is exactly the committed v6 digest.
- All joint and orientation arrays are identical at every captured stage.
- All arrays before final alignment are identical, including initial alignment
  and the IK solution. This is not a different IK experiment.
- Support transitions remain frames 4 and 76; initialization is frame 0.
- Maximum final pre-closure root change is 1.957 mm / 4.420 mm. Bounded closure
  makes maximum prepared-node changes 1.994 mm / 4.486 mm.
- Closure's original root Z gap grows from 1.235 to 2.168 / 2.912 mm, still
  accepted by the existing gate. This confirms why closure must be rechecked.
- Joint derivative RMSE remains 9.13e-8 rad/s, root derivative RMSE below
  4.40e-8 m/s; exact resets, endpoint joint/velocity closure and existing
  repeated-cycle tolerances remain satisfied.
- No foot-foot penetration on either sampling grid. Capsule endpoint selection
  is not smoothed by this experiment and remains a potential cusp mechanism.

## Next bounded investigation

Investigate how the post-IK stability blend changes local foot geometry and
how root locking transfers that motion to the pelvis. The blend is constant
and affine per joint; it cannot introduce a new time-local joint discontinuity
by itself. However, changing several joint amplitudes can spoil an IK fit and
change the nonlinear foot path. Compare stages locally before changing that
blend globally, then check marker fit, self-collision, limits and full curves.

A supplementary native stage inspection localizes this effect: at frame 69,
the right sole center's root-relative XY second-difference norm is 14.19 m/s^2
in the IK solution and 61.64 after the blend. Capsule endpoint selection does
not switch there. The blend disrupts cancellation between joint contributions
to the Cartesian foot path; the underlying joint variations already exist.

Next, freeze the v6 roots, orientations and marker targets. Compare native FK
for the existing blend, raw IK, and separately restoring raw-IK values for the
torso/pelvis, hip-Y pair or ankle-Z pair. Measure the whole cycle, emphasizing
frames 64-80 and 108-120. These are analysis-only arrays, not frame-window pose
splices or new policy references. Only a promising group goes on to a complete
pipeline ablation.

The separate large motor/contact-wrench peak near frame 113.875 is not solved
by reducing the height cusp. See the force decomposition in phase 8A and the
per-candidate extreme samples. Keep reward discrimination and a fixed pure
tracker experiment on the recovery plan; do not make perfect open-loop replay
an endless prerequisite for any learning experiment.

The same supplementary inspection finds left-knee and left-hip-Z nodal second
differences near frame 114 already in raw IK, approximately -303.34 and
+139.15 rad/s^2. The blend leaves these two joints unchanged. Their normalized
targets, seed, solver convergence and clipping need a separate local audit.
The largest abdomen force hardly changes with smoothing: 7335.33 -> 7318.47
-> 7266.73, against its configured +/-180 scalar-force range.

## Evidence and reproduction

`analysis_outputs/recovery_phase8_20261013/height_ablation.json` contains all
three conditions, both root-lock traces, stage differences, closure, geometry,
derivatives, inverse dynamics and replay records. The final run was made after
committing the diagnostic; source/input hashes identify its measured code.

Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_root_height_ablation
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
```

201 regression tests pass. No production code, policy, checkpoint, reward,
training config, stash or remote was changed in phase 8B.
