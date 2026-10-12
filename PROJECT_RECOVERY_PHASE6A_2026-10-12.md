# Recovery phase 6A: distinguish support transitions from IK startup

Parent: `babb4e0`, branch `recovery/project-v2-imitation`.
Observation-only phase. No reference, IK, physics or reward behavior changed.

The new `tools/check_root_lock_transitions.py` observes both actual stance-lock
calls for the phase-5 selected v4 WRAP cycle (source clip 1, runtime clip 0).
It reconstructs height-based support labels from native FK and compares output
motion with the stationary-foot-center recurrence. It does not replace the
locking implementation or infer force-based stance from geometry.

## Evidence and causal distinction

- Before IK, the largest XY second difference is 398.739 m/s^2 at frame 108.
  Left sole height crosses the 0.035 m threshold at frame 109; both feet are
  classified airborne through frame 112. Locking switches from following the
  stationary foot center to inheriting the input root displacement.
- After IK, the largest XY second difference is 369.063 m/s^2 at frame 1.
  This is **not a support switch**: the right foot remains selected through
  frame 75; the only change of supporting foot is frame 76.
- The right-foot X offset relative to the root at frames 0/1/2 is
  0.3723121, 0.3847813, 0.3718438 m. Final root X consequently becomes
  0, -0.0124692, +0.0004682 m. The lock transmits an upstream IK pose transient.
- The IK solver treats frame 0 differently: it starts from the raw seed without
  a previous-pose penalty; later frames use a previous-pose initialization and
  regularizer. This is appropriate for a finite clip but asymmetric at a
  repeating cycle boundary. Periodic warmup is the next isolated experiment.
- Maximum continued-support recurrence error is 1.10e-5 / 1.62e-5 m/s for the
  initial/final pass, consistent with float32 output rounding.

Evidence: `analysis_outputs/recovery_phase6_20261012/root_lock_trace.json`.
The selected v4 reference digest remains
`12ae4d6c90078d151670c96a9c46079d9d957da708d6edb2a0fe387dc2d27d0f`.
Five native small-model tests cover recurrence, support switching, airborne
frames, reinitialization, inconsistent-output detection, immutability and invalid
arguments. The complete suite has 152 tests.

These second differences are nodal roughness, not analytic curve acceleration.
Warmup alone cannot certify contact feasibility, remove target mismatch, resolve
the other support transition, or prove that the robot can walk. No PPO started.

## Reproduce (Windows Git Bash)

```bash
./.venv-win/Scripts/python.exe -m tools.check_root_lock_transitions
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
```

Next phase is a separate opt-in cyclic-IK change with the same solver, targets,
weights and limits; then a separate measured comparison, including free-root
replay. Old reference versions must remain reproducible.
