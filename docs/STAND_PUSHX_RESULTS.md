# Negative-X training: no complete robustness pass

The run `runs/stand_pushx_20261106_1m_lr3e-4` completed 1,024,000 new training
steps at 19:14 local time on 2026-11-06. Training used commit `8294630` and
the randomized training-only pushes specified in
[STAND_HORIZON60_RESULTS.md](STAND_HORIZON60_RESULTS.md). All short gates and
four selected-candidate videos are complete. The robustness process has now
completed all 25 cases. **No checkpoint passes the full suite.** Complete
evidence: `analysis_outputs/skills/stand_pushx_robustness_20261106.json`.

## Completed deterministic short gates

All four seeds agree within each checkpoint. The complete reports are saved
in `analysis_outputs/skills/stand_pushx_short_gates_20261106.json`.

| Checkpoint | Steps | Final success | Final dwell | Max drift | Verdict |
| ---: | ---: | --- | ---: | ---: | --- |
| 204,800 | 500 / 500 | yes | 10 s | 2.122 cm | pass |
| 409,600 | 500 / 500 | yes | 10 s | 2.988 cm | pass |
| 614,400 | 500 / 500 | yes | 10 s | 0.990 cm | pass |
| 819,200 | 500 / 500 | yes | 10 s | 3.292 cm | pass |
| 1,024,000 | 209 / 500 | no | 0 s | 1.505 m | physical fall |

The selected short-gate checkpoint is 614,400, with 3,686,400 total training
steps in its lineage. Reviewed spaced frames and the final frame of
`videos/000000614400/stand_seed_11.mp4`: upright stance is maintained through
the ten-second clip. This does not establish longer recovery performance.

The final checkpoint, with 4,096,000 steps in its lineage, fails at 4.18 s
under the deterministic acceptance policy. Its stochastic PPO evaluation
reported a 3,000-step episode and reward 8679.4004. These are different tests;
the stochastic evaluation does not override the deterministic failure.
Do not use the latest checkpoint automatically as the next training source.
The previously confirmed horizon60 source is retained unchanged.

## Completed robustness results

| Checkpoint | Hold | +X | -X | +Y | -Y |
| ---: | --- | --- | --- | --- | --- |
| 204,800 | pass, 60 s | pass | fall, 6.98 s | pass | pass |
| 409,600 | pass, 60 s | pass | fall, 6.94 s | pass | pass |
| 614,400 | pass, 60 s | fall, 9.10 s | fall, 6.82 s | pass | pass |
| 819,200 | fall, 40.06 s | pass | fall, 6.88 s | pass | pass |
| 1,024,000 | fall, 4.18 s | fall before pulse | fall before pulse | fall before pulse | fall before pulse |

The first two checkpoints retain four of five successful cases, as did the
horizon60 source. The training did not establish negative-X recovery. The
last checkpoint regressed even in nominal stance: all its cases end before
5 s, so their actual applied impulses are zero. Preserve the earlier source
and do not promote the latest weights. Both reports now have `status=complete`;
the parent correctly records an empty `robustness_candidates_passed` list.

## Kinematic diagnosis and next measurement

`analysis_outputs/skills/stand_pushx_kinematics_20261106.json` reconstructs
centre-of-mass position/velocity and foot contact geometry from saved MJX
qpos/qvel using native MuJoCo forward kinematics. This is not a new physics
rollout. The source and checkpoint 409,600 show the same failure pattern:
continued negative-X motion after the pulse ends at 5.2 s, followed by loss
of front foot contacts and knee extension. At 5.5 s, checkpoint 409,600 has
COM X = -0.0260 m and velocity X = -0.0948 m/s, behind the remaining contacts
at approximately -0.0161 and -0.0184 m. Its right knee approaches the extension
limit by 6.0 s.

The ankle residual prior is narrow relative to the XML joint/control range.
This motivates checking control saturation, but geometry alone does not prove
insufficient control authority. The next measurement reruns only the existing
negative-X probe on the horizon60 final checkpoint and push-trained 409,600,
with unchanged weights, action mapping, reward, model and physics.

The evaluator now accepts opt-in `--control-trace` and `--probes push_x_neg`.
The CSV additionally records each actuator's raw and smoothed action, applied
position target, resulting joint position and actuator force. The report
includes reference targets, effective residual target bounds and XML bounds.
The default full suite is unchanged. A subset always has `full_suite=false`
and cannot produce `passed=true` for overall robustness, even if its requested
cases pass. Fifteen focused tests pass, including JIT trace indexing and this
subset-certification guard; Ruff, formatting and diff checks pass.

Diagnostic output: `runs/stand_control_diagnostic_20261106`. Wait for its two
cases and compare control limits before choosing the next training change.
Walk-to-stand and A-to-B remain pending.
