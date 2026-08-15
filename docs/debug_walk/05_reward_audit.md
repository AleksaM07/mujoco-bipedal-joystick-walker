# Reward Audit

## Finding 1 - Exact Reference Reward Ceiling Is About 0.915

Severity: medium-high

Evidence:

- `config.py:644` sets `deepmimic_root_velocity_weight_scale=0.15`.
- `biomechanics_env.py:2970` multiplies the root velocity reward weight by that scale.
- The other reward weights are not renormalized.
- Deterministic reward diagnostic for BVH clip 0 at phase 0:

```text
exact_reference total=0.914938 pose=1.0 vel=1.0 root=0.999996 key=1.0
perturb_1_deg   total=0.914851
perturb_5_deg   total=0.912782
perturb_20_deg  total=0.884014
random_pose     total=0.737344
```

Semantic difference:

- DeepMimic and MimicKit use reward weights that sum to `1.0`.
- Local code changes one weight but leaves the total unnormalized.

Training effect:

- Exact imitation no longer maps to `1.0` DeepMimic reward or `3.0` environment reward.
- Reward dashboards understate perfect tracking.
- Any threshold or early-stop logic based on reference reward can become misleading.

Minimal fix:

- Restore `deepmimic_root_velocity_weight_scale=1.0`, or
- Normalize by the active sum of weights after scaling, or
- Move the intended weakening into the root velocity error scale instead of the reward weight.

Test:

```powershell
python tools\debug_reward_landscape.py --reference-gait bvh --clip-id 0 --phase 0.0
```

Pass condition: exact reference total is within a small tolerance of `1.0` for the DeepMimic component.

## Finding 2 - Reward Ordering Is Mostly Sane

Severity: low

The same diagnostic shows exact reference reward is higher than 1 degree, 5 degree, 20 degree, and random pose perturbations. This argues against a gross sign error in the main pose/key reward.

