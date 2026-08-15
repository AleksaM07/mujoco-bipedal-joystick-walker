# Phase Audit

## Finding

Severity: low for the suspected phase-speed bug

The reference phase/time update appears semantically correct.

Evidence:

- Policy/control timestep is `0.02s`.
- Physics timestep is `0.005s`.
- `n_substeps=4`.
- `biomechanics_env.py:1935` increments `motion_step` once per environment step.
- `biomechanics_env.py:3335` computes reference time from `bvh_reference_time_offset + motion_step * dt`.
- `biomechanics_env.py:3343` queries phase/frame from that control-rate time.

This matches the expected design: four physics steps per one control step, not four reference-frame advances per action.

## Remaining Phase Risk

The phase implementation is not the main issue, but clamp-clip semantics create damaging episode timing:

- `biomechanics_env.py:2729` marks non-looping clips over at their motion length.
- Random reset can sample late phases.
- Short clamp clips can end quickly after reset.

Training effect:

- PPO sees many short terminal episodes.
- The agent may be optimized against reset/end statistics rather than a sustained walk cycle.

Minimal fix:

- Keep the current time-step math.
- Build a one-clip cyclic debug task.
- For finite clips, avoid sampling reset times too close to the end unless the algorithm treats motion end as a proper success/time-limit transition.

Test:

```powershell
python tools\debug_reference_playback.py --reference-gait bvh --mode kinematic --clip-id 0 --steps 100
```

Pass condition: reported frame indices advance at control-rate time and clamp only at the declared clip duration.

