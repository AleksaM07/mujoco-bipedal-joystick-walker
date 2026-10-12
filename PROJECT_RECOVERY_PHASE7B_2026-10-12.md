# Phase 7B: version the contact-history handoff

Parent `55ab5b7` (phase 7A), branch `recovery/project-v2-imitation`.
Implementation is separate from the diagnostic experiment and final measurements.

`consistent_v6` inherits v5 and preserves an incoming support foot's previous
world-space center if it was height-classified as in contact in both frames.
It enables this rule only during final root alignment, after IK and the stability
prior. Initial alignment still uses the historical lock. Both final call sites
(with and without Cartesian IK targets) explicitly pass the versioned option.

`reference_root.lock_with_contact_history` is the shared native FK implementation.
The diagnostic ablation now uses the same function. The helper keeps the
historical float32 contact-height comparison; phase 7A used float64 there.
The tested source cycle selects the same support in both precisions. A native
boundary regression covers values which classify differently without the cast.

The first valid root pose and root Z are preserved; continuing contact cancels
foot-center XY motion, existing-contact handoffs use the incoming foot's previous
anchor, and genuine landing/flight retain the input-root displacement. Padding
holds the last valid pose. This does not enforce material-point contact, optimize
force allocation or impose velocity continuity on arbitrary input poses.

## Compatibility and evidence

- All 184 tests pass; touched standalone modules/tests pass Ruff E/F/I and
  formatting checks. Environment, configuration and training modules compile.
- The default remains `legacy`; versions through v5 use the historical locking
  body unchanged (AST comparison checked), including its existing rounding.
- A keyword-only option routes final locking explicitly. Observation wrappers
  forward it and report the requested option; diagnostic overrides are recorded
  separately in the ablation report.
- V6 inherits continuous curves, per-frame height normalization, strict BVH
  marker mapping and cyclic IK warmup. The training CLI accepts v6. Saved config
  forwards it to evaluation; v5/v6 checkpoint migration is refused.
- Native fixture tests cover supported handoff, flight, genuine landing, root
  yaw, translation, unchanged input/model, padding, method dispatch and threshold
  classification. Configuration tests exercise v6 and old versions.
- Actual selected-cycle v5/v6 capture confirms every joint, quaternion and root-Z
  array is bit-identical at every stage. Every array before final alignment is
  bit-identical. V6 observed alignment flags are `[False, True]`.
- The actual v5 digest is unchanged:
  `3ed1fac27b400954b8ac4aef847fa27e0302bfdd232e8aef0f7395300592b015`.
- The resulting selected v6 digest is
  `2cf2f8aeab70f77ed02099fab2d53285b9b2d75f7fb0a42bec3e7ce44441211e`.

V6 is an explicit experimental correction. Phase 7A already showed that this
handoff repair alone does not produce successful free-root replay. Production
v6 contacts, dynamics and replay are recorded separately in phase 7C. No source
cycle, joint objective, controller gain, actuator range, reward or PPO change is
included. No long training or policy migration was started.

Reproduce in Windows Git Bash:

```bash
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
./.venv-win/Scripts/python.exe -m tools.check_retarget_stages --version consistent_v6 --reference-clip-ids 1 --dense --output analysis_outputs/recovery_phase7_20261012/v6_stages.json
./.venv-win/Scripts/python.exe -m tools.check_reference_contacts --baseline-only --recovery-physics-version consistent_v6 --reference-clip-ids 1 --clip-id 0 --output-dir analysis_outputs/recovery_phase7_20261012/v6_physics
```
