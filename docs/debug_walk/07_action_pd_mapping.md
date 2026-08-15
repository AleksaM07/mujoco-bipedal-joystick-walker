# Action And PD Mapping Audit

## Action Mapping

Relevant code:

- `biomechanics_env.py:2096` maps policy actions to motor targets.
- In `reference_action_mode="mimickit"`, the action is center plus scaled actuator range.
- In `reference_action_mode="residual"`, the action is added around the current reference control target.

Finding:

- The default BVH reference joint targets are representable by the MimicKit-style action mapping.
- The action oracle found maximum normalized BVH reference action around `1.0`, meaning some target joints are at the edge of the allowed target range.

Training effect:

- This is not the strongest default-BVH failure.
- Bound-saturated reference targets can still make early learning brittle, especially with exploration noise and smoothing.
- For SMPL residual runs, a residual scale of `0.25` means the policy is not being asked to output absolute reference actions, but the PD oracle still must be physically survivable.

Minimal fix:

- Keep MimicKit action mapping for BVH debug runs.
- For residual SMPL runs, require `debug_reference_pd_tracking.py` to pass before training.
- Log normalized oracle action saturation per clip and exclude clips that live on actuator bounds.

Test:

```powershell
python tools\debug_reference_pd_tracking.py --reference-gait bvh --clip-id 0 --steps 120
```

Pass condition: pose RMSE stays bounded and no early low-height/tipping failure occurs.

