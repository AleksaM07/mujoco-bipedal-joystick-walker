# Phase 10A: limit-preserving knee-offset pipeline experiment

Branch: `recovery/project-v2-imitation`; production remains `consistent_v6`.
This commit adds an opt-in diagnostic, not a new training behavior. Phase 10B
records measurements and the decision separately.

## Intervention

For the source knee-angle series q and the intersection [L,U] of physical and
position-control bounds, the allowable constant offset is:

`[L - min(q), U - max(q)]`.

Project the loader's preferred standing-pose offset into this interval. This
preserves the source angular variation without per-frame clipping. If the
interval is empty, reject the candidate; do not reduce amplitudes or widen
limits. A valid unchanged knee is left bit-identical, including its velocity.

`tools/check_knee_offset_ablation.py` applies the intervention after the original
clip selector but before initial alignment, IK, the stability prior, final root
locking and curve preparation. It changes only the selected batch's knee seed
position/velocity columns. Right knee, non-knee seeds and all raw marker/root
metadata stay unchanged when their offsets already fit.

Velocity for a changed knee uses the existing loader finite difference with
the parsed source interval. Position padding repeats the last valid node;
velocity padding is zero, as in the loader. The original batch is not mutated.
The source segment and seed are checked through the phase-9 reconstruction.

The automatic loop classifier is rerun. This experiment rejects a changed
classification rather than silently switching from WRAP to CLAMP. Source
selection/ranking remains frozen. The normal later closure gate still applies.

## What is intentionally not frozen downstream

Changing a knee pose can change the initial floor-aligned roots. Existing code
then shifts the IK targets and solves the coupled joints. Downstream world
targets, other joint poses and support choices may legitimately change. The
experiment reports those changes; it does not promise an isolated one-joint
change in the final reference.

There is no height-envelope smoothing, prior change, controller gain change,
reward change or physical-limit change. The established phase-8 measurement
function is reused with epsilon zero for numerical checks, both contact grids,
inverse dynamics and twelve native open-loop replay trials per condition.
Additional evidence includes marker/sole residuals, nodal pose limits and
left-knee second differences at frame 114 through all captured stages.

The disabled control does not even replace the clip-filter method. Enabled
patches restore on success or exception. These unversioned, process-local
overrides must never be used to resume a policy or as a saved training preset.

## Tests and usage

All 248 regression tests pass, including 26 new tests. Ruff E/F/I and formatting
pass for both new Python files.

New tests cover feasible/infeasible constant offsets, preservation of temporal
differences, input validation, scoped restoration, exact unchanged columns,
physical/control bound intersection, source interval, position/velocity padding,
metadata identity, non-mutation and loop-classification rejection.

Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m tools.check_knee_offset_ablation
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
```

Default evidence:
`analysis_outputs/recovery_phase10_20261013/knee_offset.json`.

No PPO, policy overwrite, stash operation, branch deletion or push.
