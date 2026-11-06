# Stand: 60-second balance and remaining negative-X recovery failure

The 1M continuation with 60-second training episodes completed successfully.
All five checkpoints pass the original four-seed ten-second gate. Four also
pass the full 60-second hold with continuous stand eligibility. The last two
pass three of the four push directions. Every checkpoint still fails -world X.

Run: `runs/stand_horizon60_20261106_1m_lr3e-4`, launched from `c37f924`.
The pipeline finished at 15:35 UTC (16:35 local) on 2026-11-06, including
all short gates, four short-gate videos and all 25 robustness cases.
The complete evidence is in
`analysis_outputs/skills/stand_horizon60_20261106.json`.

| New checkpoint steps | 60-second hold | Hold max drift | +X | -X | +Y | -Y |
| ---: | --- | ---: | --- | --- | --- | --- |
| 204,800 | fall at 28.98 s | 1.495 m | fail | fail | pass | pass |
| 409,600 | pass | 6.276 cm | fail | fail | pass | pass |
| 614,400 | pass | 2.395 cm | fail | fail | pass | pass |
| 819,200 | pass | 2.338 cm | pass | fail | pass | pass |
| 1,024,000 | pass | 2.436 cm | pass | fail | pass | pass |

The final checkpoint has 3,072,000 actual training steps in its lineage.
Its nominal hold has mean speed 0.001803 m/s and final dwell 60 s. Its -X
case falls at 6.96 s after the 5.0–5.2 s pulse. The pulse is 36.861076 N at
the thorax centre of mass, with impulse 7.372216 Ns. None passes the complete
robustness suite; `robustness_candidates_passed` is correctly empty.

The runner's short-gate selection is checkpoint 614,400. It is not the best
recovery candidate: it still fails +X. For the next continuation use the
**final 1,024,000 checkpoint**, which passes the long hold and three directions.
This preserves a clear training lineage; the penultimate checkpoint also
passes those four cases and is retained.

## Visual review

Rendered saved physical states for the final checkpoint at 10 Hz, from four
cameras, without re-running or resetting physics. Reviewed six spaced frames
including the last frame of each recording. They show upright stance through
60 s and a fall after the negative-X pulse, consistent with the numeric traces.

Videos under the run directory:

- `robustness/000001024000/hold_60s.mp4`
- `robustness/000001024000/push_x_neg.mp4`

The failed case includes its terminal state. Its final sampling interval is
shorter than 0.1 s; fixed-rate playback places that last frame on the next
10 Hz frame boundary. Timestamps in the video show the original physical time.

## Next controlled experiment

Continue the final weights for one additional 1M requested steps. Keep
60-second episodes, `balance_v1`, LR `3e-4`, seed 7, 64 environments and the
same PPO settings. Restore actor, critic and observation normalizer; the
optimizer and step counter start afresh as in previous continuations.

The only new training condition is opt-in
`TrainConfig.stand_push_max_fraction=0.05`. A training-only wrapper samples:

- A nominal episode with probability 0.5.
- Otherwise a -world X force uniformly between zero and 5% of body weight.
- One pulse phase uniformly from 2.0 to 8.0 s, on the control-step grid.
- A 0.2 s thorax pulse at that phase in each ten-second interval.

Magnitude and phase are sampled anew on every episode reset and held within
that episode. A full 60-second episode therefore has six pulses or none.
The nominal fraction is probabilistic, not an exact split of the 64 environments.
Weaker pushes provide easier recovery examples; nominal episodes preserve
exposure to unforced stance. Neither phase nor force is added to observations.

Forces use `xfrc_applied` in world coordinates at the thorax centre of mass.
All other entries and explicit torques are zero; force is cleared outside
the pulse. No velocity edits, changed reference, reward, contact parameters,
XML, gravity, domain randomization or ERFI are introduced. The default
fraction is zero, so old configurations retain their previous behavior.

PPO evaluation remains unforced. The independent acceptance protocol is
unchanged: four-seed ten-second gates, then the 60-second hold and four
fixed 5%-weight pushes on every new checkpoint. Passing only training reward,
the short gate or three directions is insufficient. Walk-to-stand and A-to-B
remain pending.

Planned output: `runs/stand_pushx_20261106_1m_lr3e-4`.

```text
.venv/bin/python -u scripts/run_stand_ablation.py --variants resume --resume-from <horizon60_final_checkpoint> --timesteps 1000000 --stand-reward-mode balance_v1 --learning-rate 0.0003 --episode-length 3000 --stand-push-max-fraction 0.05 --robustness-check --output runs/stand_pushx_20261106_1m_lr3e-4
```

The new tests cover real MJX impulse response and force removal, timing and
repetition, seeded magnitude/phase sampling, automatic reset of force and
schedule, nominal evaluation, unchanged humanoid observation layout, argument
forwarding and compatibility with the saved source configuration. All **412
tests pass** (439.561 s), including the new real-MJX and humanoid checks.
The log is `runs/stand_pushx_20261106_full_tests.log`. Ruff checks pass for the
stand wrapper, runner and changed stand tests; `train.py` and `config.py`
retain their seven pre-existing lint findings, with no additional findings.
Formatting of the new module/runner tests and `git diff --check` pass.
A WSL dry-run against the actual final checkpoint confirms the intended
source, 1M budget, 3,000-step horizon, LR, seed, environment count, critic
restore and 0.05 maximum push fraction.
