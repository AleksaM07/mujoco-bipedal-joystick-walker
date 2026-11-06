# Stand robustness v1 — frozen checkpoint validation

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
