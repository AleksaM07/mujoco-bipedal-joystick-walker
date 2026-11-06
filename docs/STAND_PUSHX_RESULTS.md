# Negative-X training: completed short gates, robustness still running

The run `runs/stand_pushx_20261106_1m_lr3e-4` completed 1,024,000 new training
steps at 19:14 local time on 2026-11-06. Training used commit `8294630` and
the randomized training-only pushes specified in
[STAND_HORIZON60_RESULTS.md](STAND_HORIZON60_RESULTS.md). All short gates and
four selected-candidate videos are complete. The 25-case robustness process
is still active; **this is not a completed robustness result**.

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

## Pending robustness results

At the 20:10 local review, five of 25 cases had completed. Checkpoint 204,800
passes the 60-second hold and +X/±Y pushes, but fails -X at 349 steps (6.98 s).
The process was evaluating checkpoint 409,600 next. Its parent/evaluator PIDs
were 429/9115 when checked; re-check process identity rather than relying on
these PIDs later.

The parent `comparison.json` is at `status=running`, `stage=robustness`.
`robustness/report.json` is updated after every case. Let the current process
finish all 25 cases before selecting another source or starting training.
A checkpoint must pass the complete longer suite and visual review before
moving to walk-to-stand. If none passes, inspect the failed traces and training
distribution before choosing the next controlled change; do not infer success
from reward or the ten-second video.
