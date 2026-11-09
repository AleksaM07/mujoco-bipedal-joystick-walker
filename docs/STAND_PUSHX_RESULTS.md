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

Diagnostic output: `runs/stand_control_diagnostic_20261106`. Both cases are
complete; the findings and next experiment are below. Walk-to-stand and A-to-B
remain pending.

The diagnostic was launched at 20:16 UTC (21:16 local) on 2026-11-06 from
clean commit `f34efc2`, with hidden Windows launcher PID 28940 at startup.
The sibling `.launch.json` contains the exact command. Its initial report
confirms a diagnostic subset with control tracing and no full-suite claim.

## Completed control diagnosis, reviewed 2026-11-09

Saved evidence: `analysis_outputs/skills/stand_control_diagnostic_20261106.json`.
The source falls at 6.96 s and push-trained 409,600 at 6.94 s. Every physical
summary field and every replay array (`qpos`, `qvel`, `steps`, `dt`) agrees
exactly with the original non-instrumented case. The report remains
`full_suite=false`, `passed=false`.

The same 51 control samples from 5.0 through 6.0 s cover the force pulse and
initial balance loss in both cases. Neither ankle-z target reaches its
effective range, neither raw ankle-z action reaches absolute 0.99, and neither
sampled ankle-z actuator force reaches the XML force limit.

| Candidate | Left/right maximum raw ankle-z action | Left/right maximum ankle-z target (rad) | Right knee target at extension bound |
| --- | --- | --- | --- |
| horizon60 final | 0.559 / 0.153 | 0.08874 / 0.05944 | 13 / 51 samples |
| push-trained 409,600 | 0.325 / 0.078 | 0.07229 / 0.05075 | 19 / 51 samples |

Both ankle-z target ranges are approximately [-0.03, 0.13] rad. These samples
do not support ankle command saturation as the cause of early balance loss.
They do not prove that all control constraints are irrelevant: the right knee
target reaches its physical extension bound. No action-range or XML change is
justified by the ankle saturation hypothesis on this evidence alone.

## Active controlled continuation

Run: `runs/stand_pushx_continue_20261109_1m_lr1e-4`.

Continue push-trained checkpoint 409,600, which retains the 60-second hold and
+X/±Y recovery, with one additional 1M-step budget. Its lineage contains
3,481,600 steps before this continuation. Change learning rate from `3e-4` to
`1e-4`, retaining the reward, 3,000-step episodes, seed 7, 64 environments,
training pulse distribution, physics, action mapping and acceptance criteria.
Restore actor, critic and observation normalizer with a fresh optimizer.

The late deterministic regression motivates testing smaller updates while
continuing recovery training. This is a continuation from an earlier checkpoint,
not a matched causal demonstration that learning rate caused the regression.
The failed final checkpoint is not a training source. The actual WSL dry-run
validates the source checkpoint, XML hash and matching environment/PPO settings.

The existing runner performs four-seed short gates for all five new checkpoints,
four selected-candidate videos, and all 25 long-hold/push cases automatically.
Success requires a candidate in `robustness_candidates_passed` plus video review;
pipeline completion or reward alone is insufficient. No additional training
is queued after this single run.

```powershell
$sourceRun = @(Get-ChildItem runs/stand_pushx_20261106_1m_lr3e-4/resume -Directory)
if ($sourceRun.Count -ne 1) { throw 'Expected one source run' }
$sourcePath = $sourceRun[0].FullName.Substring((Get-Location).Path.Length + 1).Replace('\', '/')
wsl -e env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_stand_ablation.py --variants resume --resume-from "$sourcePath/checkpoints/000000409600" --timesteps 1000000 --stand-reward-mode balance_v1 --learning-rate 0.0001 --episode-length 3000 --stand-push-max-fraction 0.05 --robustness-check --output runs/stand_pushx_continue_20261109_1m_lr1e-4
```

Before running this command, check whether that output/process already exists.
The background launcher uses the same arguments with hidden Windows process
execution and separate stdout/stderr logs.

Launched on 2026-11-09 at 13:51:49 UTC (14:51 local) from clean commit
`4373829`, after pushing the diagnosis and plan. Windows launcher PID at
startup: 31584; WSL parent/worker at the 14:52 check: 432/433. The sibling
`.launch.json` records the exact command and source commit. `comparison.json`
records `status=running`, `stage=training`, `robustness_requested=true`.
The saved `resume/*/config.json` matches the WSL dry-run plan in every env and
train field.

At the 15:57 local check on 2026-11-09, training has completed all 1,024,000
steps and exited with code 0 at 15:52. The final stochastic PPO evaluation
reports a 3,000-step average episode with no falls; this is not an acceptance
gate. The parent is running `evaluate_stand.py` over the five checkpoints
(WSL parent/evaluator PID 432/6708 at that check). The first seed-11 rollout
passes 500/500 steps with final dwell 10 s and max drift 0.018938 m. The full
short gates and long robustness suite are not complete yet. The parent report
still says `stage=training` until the variant advances, so inspect
`resume_gate.log`, stdout and the process as well as `comparison.json`.

The user's latest scope is to finish this existing run and its automatic
checks only, then transfer to a new chat. No further training is queued or
authorized by that latest instruction. Save the completed results and video
review before proposing another experiment.
