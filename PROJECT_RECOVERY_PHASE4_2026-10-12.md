# Recovery phase 4: both feet were targeting the pelvis

Branch: `recovery/project-v2-imitation`, phase-3 parent `18515c7`.
No PPO, checkpoint migration, controller retuning or collision-mask changes.

## Confirmed cause and controlled correction

The actual `02_01.bvh` skeleton names its toes `RightToe` and `LeftToe`.
`BVH_IK_TARGET_MAP` expected `RightToeBase` and `LeftToeBase`.
`_bvh_ik_target_positions` silently substituted the Hips position for each absent
joint. Thus **both foot targets were identical in every frame**. The subsequent
height normalization lowered these identical targets; it could not recover their
missing fore/aft motion or lateral separation.

The stage audit wraps the real loader, floor alignment, IK and stability-prior
methods without modifying their inputs or returned outputs. The causal sequence
for all three clips (136 + 134 + 131 = 401 frames) is:

| Stage | V3: coincident source targets | V4: actual source toes |
| --- | ---: | ---: |
| Raw mapped joint pose: sole-sole penetrating frames | 0/401 | 0/401 |
| Initial floor alignment: penetrating frames | 0/401 | 0/401 |
| IK solution: penetrating frames | 401/401 | 0/401 |
| Post-IK stability prior: penetrating frames | 401/401 | 0/401 |
| Final floor/root alignment: penetrating frames | 401/401 | 0/401 |

This isolates the first physical overlap to fitting the corrupted Cartesian
targets. It does not establish that raw joint-angle mapping or IK is otherwise
accurate. Neither removing collision nor increasing actuator authority was needed
to remove this measured overlap.

For clip 1, corrected target left-minus-right lateral spacing is 0.02478-0.17392 m
along the neutral model's lateral axis. Full 3D target distance is 0.12480-0.78795 m;
small lateral distance alone is not a collision test. The diagnostic uses native
sole geometry to measure penetration, not a minimum-width approximation.

## Narrow versioned implementation

- `consistent_v4` inherits v3 height normalization and v2 continuous curves.
- Its BVH loader resolves `RightToeBase` then `RightToe`, and `LeftToeBase` then
  `LeftToe`. If both spellings exist, the canonical ToeBase spelling wins. This is
  explicit and tested. Missing either foot raises a descriptive `ValueError`;
  there is no pelvis or ankle substitute in v4.
- V4 also rejects missing/reordered simulator IK markers before fitting. Previously
  filtering an unresolved XML marker could shift target indices for later markers.
  That latent problem did not cause this v21 failure: both XML sites exist.
- Legacy/v1/v2/v3 and the historical direct-loader default remain unchanged.
  The default environment version is still `legacy`. No existing checkpoint is
  silently given new targets; v3-to-v4 resume is refused.

The real v3 processed-reference digest remains identical to phase 3:
`0fd0f0099cbf9f69fd35a1ef985e20122c8eecf3e095f06e29a25ecc9f9653d9`.

## The next red lights are visible, not bypassed

With correct targets the unchanged bounded WRAP gate rejects clip 0. The audit
records the rejection and inspects each clip separately without increasing limits:

| Clip | Frames | Joint endpoint gap | Root-Z gap | Bounded WRAP gate |
| --- | ---: | ---: | ---: | --- |
| 0 | 136 | 0.209577 rad | 19.953 mm | Reject: joint limit is 0.2 rad |
| 1 | 134 | 0.055607 rad | 6.183 mm | Pass |
| 2 | 131 | 0.071046 rad | 2.658 mm | Pass |

**The full v4/auto batch does not initialize.** `reference_forced_clip_id=1` only
selects runtime sampling; it does not remove clip 0 before curve preparation.
The tool deliberately reports the partial host-FK audit and exits nonzero.
The partial object is never stepped or used for runtime reference queries.

A separate explicitly labeled **finite CLAMP diagnostic** initializes and checks
the final curves at 540 + 532 + 520 = **1,592 off-knot samples**. There are no
sole-sole collisions at these samples or at the 401 prepared nodes. Floor contacts
remain enabled. Worst floor penetration is 5.680 mm, including the unchanged
intentional 5 mm preload. CLAMP does **not** repair or certify a repeating gait;
no multi-cycle replay or learned policy success is claimed for v4.

IK still leaves material Cartesian error. For clip 1:

| Marker residual | Right median / max | Left median / max |
| --- | ---: | ---: |
| Immediately after IK | 67.68 / 138.24 mm | 65.46 / 158.84 mm |
| After stability blending | 93.47 / 158.91 mm | 81.56 / 161.56 mm |

These compare the exact named sites to their same-frame normalized targets.
The solver still uses regularization, a fixed root and only two foot-position
constraints; it has no final Cartesian acceptance threshold. Stability blending
can worsen the fit. Those are the next issues to measure, not reasons to alter
PPO hyperparameters. Raw joint-angle seeds are also poor Cartesian fits here.

Correct source-foot motion also changes the stance-lock result: clip 1's final
pre-closure root X translation is now **+1.37426 m** instead of v3's approximately
-0.00443 m. This is kinematic reference travel, not actual simulated walking.
Foot locking, slip, acceleration and dynamic feasibility need fresh measurement
on the corrected reference; old exact-force/replay numbers are not v4 results.

## Evidence, validation and reproduction

123 regression tests pass. New/modified diagnostic and test files pass Ruff
E/F/I and formatter checks. Production Python files compile; no new dependency
was installed. Tests include independent synthetic FK with translation/yaw,
left/right alias ordering, legacy fallback, missing source and XML markers,
config restoration, migration refusal, observation-only stage capture, copied
arrays, selective failure handling and monkeypatch restoration.

Evidence under `analysis_outputs/recovery_phase4_20261012/`:

- `stages.json`: v3 control, original target collapse and preserved reference hash.
- `corrected_stages.json`: v4 auto, pre-curve stages and all individual gates;
  initialization is explicitly false because clip 0 is rejected.
- `corrected_clamp_diagnostic.json`: finite-only curves and dense collision checks.

Reports include full config, source/model/code hashes, per-frame collision depth,
spacing, named-marker residuals and limitations. Absence of contact is encoded as
zero penetration, not a claim of measured positive clearance. The lateral axis
is fixed to neutral standing, not a moving heading frame.

Windows Git Bash, repository root:

```bash
./.venv-win/Scripts/python.exe -m unittest discover -s tests -q
./.venv-win/Scripts/python.exe -m tools.check_retarget_stages --version consistent_v3 --output analysis_outputs/recovery_phase4_20261012/stages.json
./.venv-win/Scripts/python.exe -m tools.check_retarget_stages --version consistent_v4 --output analysis_outputs/recovery_phase4_20261012/corrected_stages.json
./.venv-win/Scripts/python.exe -m tools.check_retarget_stages --version consistent_v4 --loop-mode clamp --dense --output analysis_outputs/recovery_phase4_20261012/corrected_clamp_diagnostic.json
```

The v4/auto command intentionally exits nonzero after saving the rejected-gate
report. Re-running replaces only the named generated reports.

## Next controlled step

1. Explicitly select one accepted source cycle before preprocessing, retaining
   its original source-frame provenance. Do not use runtime forced-clip selection
   as though it filtered the input batch, and do not relax the closure limits.
2. Measure whether remaining Cartesian error comes from root placement, reachable
   limb geometry, joint bounds or the regularized solver. Compare fitting before
   and after the stability prior; retain source-foot separation and collision tests.
3. Repeat exact derivative, floor/contact/slip, acceleration, actuator-authority
   and free-root tracking checks on that corrected repeating reference.
4. Only then freeze a pure-tracker preset and allocate training; no joystick or
   unrelated reward/curriculum changes in the same experiment.

## User-requested branch cleanup

The three obsolete local branch refs were removed only after archiving all tips
in a verified incremental bundle. See `BRANCH_ARCHIVE_2026-10-12.md` for full hashes,
retained lessons, archive SHA256, prerequisites and restore commands. Remote refs,
main, reference-only, recovery, the diagnostic stash and checkpoints are untouched.
