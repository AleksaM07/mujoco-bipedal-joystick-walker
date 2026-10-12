# Recovery phase 6B: opt-in cyclic IK startup correction

Parent: phase 6A `a0971ad`, branch `recovery/project-v2-imitation`.
Code and regression tests are committed separately from the phase-6C physical
comparison. No training, push or checkpoint migration is part of this change.

## What changed

`recovery_physics_version="consistent_v5"` inherits v4 and changes only the
initialization history of the existing causal IK fit for WRAP references.
The default is still `legacy`; v1-v4 retain their original behavior.

- Four copies of the cycle's **count - 1 unique nodes**, then its original full
  node sequence, are passed to the same IK kernel in one call per clip.
- The endpoint is not duplicated into the warmup period: phase zero's causal
  predecessor is node `count - 2`, not the endpoint at the next phase zero.
- Every joint of the last warmup period must match the retained period within
  1e-6 rad; a nonconvergent or nonfinite fit fails explicitly.
- The retained output has the original frame count; padded frames hold its last
  valid pose. CLAMP and singleton clips have no warmup.
- Copies isolate source arrays from the callback. Input dimensions, finite
  values, counts, modes and convergence settings are checked.
- Per-clip diagnostics record convergence and the float32 predecessor pose used
  for the frame-zero diagnostic objective. The independent IK-fit audit now
  uses that predecessor, rather than assuming frame zero has no history.

The original kernel moved to `_fit_reference_ik_once` **without any body change**
(AST comparison verified). The wrapper retains the original public internal
entry point, so retarget-stage capture observes the original-length output.
No IK weights, regularization, iteration count, control limits, collision masks,
root-lock rule, rewards or PPO settings changed.

## Compatibility and validation

- All 170 regression tests pass; touched standalone modules/tests pass Ruff
  E/F/I and format checks, and the environment/config/train modules compile.
- Training CLI accepts v5; saved config restores and forwards it in evaluation.
  Resume still rejects implicit v4-to-v5 and v5-to-v4 migration.
- Unit tests cover endpoint indexing, convergence failure, mixed clip lengths,
  finite clips, singleton/two-frame cycles, padding, input immutability, invalid
  values, callback output, old-version bypass and v5 wrapper wiring.
- Existing height-normalization and constructor-flag tests now also cover v5;
  three saved-config tests exercise CLI, roundtrip and migration protection.
- The actual selected v4 control retains digest
  `12ae4d6c90078d151670c96a9c46079d9d957da708d6edb2a0fe387dc2d27d0f`.

This is cyclic **startup convergence**, not a globally periodic IK optimization.
Original endpoint seeds/targets can still differ. Existing bounded seam guards
remain mandatory. Repeating world transforms is valid here because both root
and markers restart together and the solver retains only joint-pose history.
It would not be valid for a solver retaining world-space contacts or velocities.

V5 is an experimental opt-in, not a promoted training preset. Physical contact,
inverse-demand and native replay results are recorded separately in phase 6C.

## Reproduce (Windows Git Bash)

```bash
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
./.venv-win/Scripts/python.exe -m tools.check_retarget_stages --version consistent_v5 --reference-clip-ids 1 --dense --output analysis_outputs/recovery_phase6_20261012/v5_stages.json
./.venv-win/Scripts/python.exe -m tools.check_reference_contacts --baseline-only --recovery-physics-version consistent_v5 --reference-clip-ids 1 --clip-id 0 --output-dir analysis_outputs/recovery_phase6_20261012/v5_physics
```
