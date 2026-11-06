# Stand robustness v1 — frozen checkpoint validation

The follow-up with 60-second training episodes is complete. See
[STAND_HORIZON60_RESULTS.md](STAND_HORIZON60_RESULTS.md): sustained stance now
passes, while negative-X recovery remains the next training target. The
acceptance protocol below is unchanged.

The user reviewed the successful ten-second stand video and authorized the
next step. This protocol tests the selected continuation checkpoint 409,600
and the final checkpoint 1,024,000. No weights, reward, reset, model XML or
training settings are changed.

`tools/evaluate_stand_robustness.py` runs five cases per checkpoint:

| Case | Duration | Disturbance |
| --- | ---: | --- |
| `hold_60s` | 60 s / 3,000 steps | None |
| `push_x_pos` | 20 s / 1,000 steps | +world X |
| `push_x_neg` | 20 s / 1,000 steps | -world X |
| `push_y_pos` | 20 s / 1,000 steps | +world Y |
| `push_y_neg` | 20 s / 1,000 steps | -world Y |

Each push starts at 5.0 s and ends at 5.2 s: control steps 250 through 259
(zero-based). Force magnitude is 5% of total model weight: `0.05 * mass * |g|`.
For a 75 kg model this is 36.7875 N and a 7.3575 Ns impulse. The force acts at
the thorax centre of mass through MJX `xfrc_applied[:, :3]`, in world coordinates.
All other external forces and explicit torques are zero. Force is cleared
on every control step outside the pulse. Joint torque injection retains its
saved disabled configuration; velocities are never directly overwritten.

The 60-second case requires the full horizon without a physical failure and
continuous eligibility for stand. Push cases allow transient loss of eligibility,
but require a fresh one-second eligible dwell within five seconds after release,
completion of the full twenty seconds without falling, and final stand success.
Recovery timing excludes all dwell accumulated before the push. Passing this
small deterministic suite is not a claim of broad stochastic robustness or a
successful transition from walking.

The reset uses seed 11 in each case, isolating direction and checkpoint as the
changed variables. The earlier four-seed gate already establishes repeatability
from the same neutral physical start; additional identical seeds would not add
different perturbations.

## Execution and outputs

The first suite was launched from commit `823bc51` on 2026-11-06 at 08:36 UTC
(09:36 local time), as a hidden WSL process. Its output directory is
`runs/stand_robustness_20261106_selected_final`; launcher logs are the sibling
`.stdout.log` and `.stderr.log` files. The Windows launcher PID was 27984 at
startup. The suite is now complete: all ten cases were evaluated, and neither
checkpoint passed the longer hold or any complete push-case criterion.

Run with the existing WSL virtual environment after prior GPU processes exit:

```text
.venv/bin/python -u tools/evaluate_stand_robustness.py --checkpoints <selected_checkpoint> <final_checkpoint> --output-dir runs/<fresh_robustness_run> --device gpu
```

The script refuses existing output directories and mismatched environments or
network settings. It saves `report.json` after each case, including actual
impulse, recovery time, physical metrics and verdict. Every case also saves a
CSV containing all control-step metrics and force values, plus compressed
`*_replay.npz` states at 10 Hz and at termination for later rendering. No video
rendering blocks the physical tests. Exit code 0 means every case passed;
1 means at least one physical case failed. A partial `status=running` report
must never be treated as completed validation.

The original evaluator's CLI remains the fixed 500-step gate. Only its internal
summary accepts explicitly longer horizons, with a minimum of 500 steps, and
its compiled physics step accepts an optional external-force array.

## Validation

Thirteen focused tests pass: seven original checkpoint gate tests and six new
tests covering exact pulse timing, sign and impulse; native MuJoCo free-body
velocity response; force removal; post-push dwell timing; late recovery; final
failure; and rejecting a short trajectory as a long-horizon success. Ruff lint,
formatting and diff checks pass. Installed MJX source confirms the force/torque
ordering and world-space application at `xipos` used by this protocol.

## Completed result — 2026-11-06

The complete report is preserved in
`analysis_outputs/skills/stand_robustness_20261106.json`. The actual model mass
is 75.15 kg, giving a 36.861076 N pulse and 7.372216 Ns impulse.

| Checkpoint | Case | Physical steps | End time | Fresh recovery dwell | Final success |
| ---: | --- | ---: | ---: | ---: | --- |
| 409,600 | hold | 971 / 3,000 | 19.42 s | n/a | no |
| 409,600 | +X | 360 / 1,000 | 7.20 s | not reached | no |
| 409,600 | -X | 365 / 1,000 | 7.30 s | not reached | no |
| 409,600 | +Y | 984 / 1,000 | 19.68 s | 1.00 s after release | no |
| 409,600 | -Y | 947 / 1,000 | 18.94 s | 1.00 s after release | no |
| 1,024,000 | hold | 967 / 3,000 | 19.34 s | n/a | no |
| 1,024,000 | +X | 915 / 1,000 | 18.30 s | 1.02 s after release | no |
| 1,024,000 | -X | 349 / 1,000 | 6.98 s | not reached | no |
| 1,024,000 | +Y | 1,000 / 1,000 | 20.00 s | 1.00 s after release | no |
| 1,024,000 | -Y | 926 / 1,000 | 18.52 s | 1.00 s after release | no |

The final checkpoint's +Y case survives the full horizon but loses stand
eligibility: max drift is 1.239 m and final success is false. All other cases
end in physical failure. A brief recovery dwell alone is insufficient.

The first 500 steps of the selected checkpoint's longer hold reproduce its
earlier successful gate exactly: max drift 0.0206152033 m, mean speed
0.0054618480 m/s, and eligibility on every step. Thus the longer evaluator
preserves the earlier ten-second result. The later instability is outside
the previous training horizon. This supports testing a longer horizon; it
does not prove that horizon length is the only cause.

## Next controlled experiment: 60-second training episodes

Continue the morning run's **final 1,024,000 checkpoint** for one additional
1M requested steps, restoring actor, critic and observation normalizer.
Brax initializes a fresh optimizer and step counter. The source has the most
training in this lineage and better +X recovery in this suite; it is not
uniformly superior in every case.

Change only `episode_length` from 500 to 3,000 (10 to 60 seconds). Keep
`balance_v1`, learning rate `3e-4`, seed 7, 64 environments, PPO settings,
neutral reset, model and physics. Do not add training pushes in this run.
The final weights will have 3,072,000 actual steps in the lineage if the
new requested 1M again rounds to 1,024,000.

```text
.venv/bin/python -u scripts/run_stand_ablation.py --variants resume --resume-from <morning_final_checkpoint> --timesteps 1000000 --stand-reward-mode balance_v1 --learning-rate 0.0003 --episode-length 3000 --robustness-check --output runs/stand_horizon60_20261106_1m_lr3e-4
```

The runner checks every new checkpoint with the original four-seed 500-step
gate, renders the best short-gate candidate, then runs all five robustness
cases on **every** new checkpoint. With five saved checkpoints this gives
25 longer cases. `stand_gate_passed` describes only the short gate;
`robustness_candidates_passed` separately lists checkpoints passing all five
longer cases. A completed physical failure is a valid experiment result.
Missing or partial robustness results make the pipeline incomplete.

Eight runner tests and seven overnight regression tests pass, covering the
explicit horizon change, preservation of environment/PPO configuration,
worker argument forwarding, complete physical failures, passing candidates
and incomplete case sets. Ruff and formatting checks pass. A WSL dry-run
against the actual source checkpoint confirms the intended configuration.
The old overnight queue is not restarted. Walk-to-stand and A-to-B remain
pending evidence of sustained balance and recovery.

This continuation was launched on 2026-11-06 at 13:14 UTC (14:14 local) from
clean commit `c37f924`, with Windows launcher PID 26156 at startup. Output:
`runs/stand_horizon60_20261106_1m_lr3e-4`. The sibling `.launch.json` records
the exact arguments and source commit. Check this existing run and its
`comparison.json` before starting anything else; no result is available yet.
The worker's saved `config.json` confirms the 3,000-step episodes, source
checkpoint, critic restore, `balance_v1`, LR `3e-4`, seed 7 and 64 environments.
