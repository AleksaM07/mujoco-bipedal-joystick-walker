# Phase 11A: frozen-reference reward discrimination audit

Branch: `recovery/project-v2-imitation`. This phase adds diagnostic code and
tests only. Production reward, reference v6 and saved policies remain unchanged.
Measurements and interpretation are recorded separately in phase 11B.

## Purpose

Return to the original diagnosis's reward audit without making open-loop
survival a prerequisite for testing the learning objective. Measure the actual
production reward on explicitly defined states, not a duplicate approximation.
Higher imitation reward alone must not be interpreted as learned walking.

`tools/check_reward_discrimination.py` pins the same v21 no-arms XML,
`02_01.bvh` selected source clip 1/runtime clip 0, `consistent_v6`, bounded
closure and pure imitation. It rejects other recovery versions, mixed mode,
multiple clips and non-WRAP input rather than silently changing semantics.

## Measurement

Sample 128 uniformly spaced midpoint phases over one cycle. Assemble full
native MuJoCo qpos/qvel using named actuator indices and query-derived free-root
coordinates. Run `mj_forward` independently for each state and copy the native
kinematic fields into JAX arrays. Call the real environment methods:

- `_get_bvh_deepmimic_reward`: all components and errors;
- `_get_reward`: actual pure step reward, including terminal handling;
- `_get_done` and `_get_physical_fail_done`: report termination separately.

Every candidate is scored against the original phase, with standing fallback
false and joystick command zero. Zero current/previous actions are passed to
the pure branch, where action costs do not contribute. This is not a policy
rollout, action-space test or new native-physics training backend.

Thirteen cases per phase (1,664 total snapshots):

| Case | Construction |
| --- | --- |
| Exact reference | Queried full qpos/qvel |
| Standing default | Fixed configured neutral `_init_q`, all velocities zero; not a claim of identical fallback reset |
| Frozen phase zero | Fixed phase-zero reference pose, all velocities zero |
| Stopped reference pose | Current reference pose, all velocities zero; instantaneous probe |
| Wrong phase, quarter / half | Full state from shifted reference phase, still scored against original time |
| Swapped sagittal legs | Exchange named hip-Z, knee-Z, ankle-Z pose and velocity channels; not a full anatomical mirror |
| Translated XY control | Add world XY `[1,-0.5]` m; unchanged velocities; tests intentional translation invariance, not slip |
| Added root X velocity | Add 0.5 m/s world X; pose unchanged |
| Four knee velocity probes | Add +2 or -2 rad/s to either knee alone; pose/root velocities unchanged |

No candidate is clamped. Position/control-limit excess is reported. Native
Jacobians measure the horizontal material-point velocity at each lowest sole
point. Height <=1 cm defines a near-floor proxy, including penetration, not
force-confirmed stance. Velocity probes are not coherent trajectories.

The exact-state reward must be >=0.9999 and translated-control total must match
within 1e-6. Reference digest is checked unchanged before/after the audit.

## Normalization visibility and provenance

Report the actual class scales/weights, every scalar reward/error, mean/min/max/
percentiles, per-phase foot measurements, and source/input hashes. Python,
MuJoCo, JAX and NumPy versions and Git HEAD accompany the output.

An explicitly labelled offline counterfactual replaces only mean squared
hinge pose/velocity errors with sums by multiplying the reported means by the
actuator count. Other components and normalized weights stay fixed. This does
not alter runtime rewards, is not a saved training configuration, and is not
a faithful replication of a quaternion-based external method.

Exact-reference reward tests the internally processed pose/FK target; it does
not certify source BVH marker agreement, contact quality or physical tracking.

## Verification and usage

All 274 regression tests pass, including 26 new tests in two files:

- `tests/test_deepmimic_reward_contract.py`: actual inherited reward math,
  mean normalization, error monotonicity, root XY invariance versus height,
  velocity sensitivity, normalized weights and single clocked query;
- `tests/test_reward_discrimination.py`: all candidate definitions, reordered
  noncontiguous actuator indices, source immutability/array independence,
  native snapshot ownership, finite summaries and scope guards.

Ruff E/F/I and formatting pass for all three new Python files.

Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_reward_discrimination
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
```

Output: `analysis_outputs/recovery_phase11_20261013/reward_discrimination.json`.
No PPO, preset changes, checkpoint writes, stash operations or push.
