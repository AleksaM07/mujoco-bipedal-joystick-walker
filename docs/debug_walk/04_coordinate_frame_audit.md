# Coordinate Frame Audit

## BVH Conversion

Relevant code:

- `bvh_reference.py:1023` builds root position, root rotation, root velocity, and root angular velocity targets.
- `bvh_reference.py:1119` converts BVH Y-up centimeters to MuJoCo Z-up meters with `[Z, -X, Y]`.
- `bvh_reference.py:1125` converts BVH rotations into the MuJoCo frame.
- `bvh_reference.py:290` retargets joint roles to MuJoCo hinge coordinates.

Initial verdict:

- No single obvious global axis swap was found in the reward invariant tests. Exact reference reset produces near-zero pose and key errors for sampled BVH frames.
- This reduces the likelihood of a catastrophic frame mismatch in the BVH reward path.

## Root/Anchor Velocity Mismatch

Severity: medium

Evidence:

- Exact BVH states can still show root angular velocity error up to about `1.75` in the diagnostic sweep.
- Reset assigns free-root velocity from reference in `biomechanics_env.py:3536`.
- Reward root velocity is computed from the anchor body velocity, not directly from free-joint qvel.

Training effect:

- The root velocity component can penalize exact reference states.
- Because the root velocity weight is already down-scaled, this is unlikely to be the only failure. It is still a broken invariant.

Minimal fix:

- Either compare free-root velocity to the free-root reference, or compute the reference anchor-body angular velocity through the same MuJoCo forward-kinematics path used for the simulated body.

Test:

```powershell
python tools\debug_reward_landscape.py --reference-gait bvh --clip-id 0 --phase 0.0
```

Pass condition after fix: exact reset has near-zero root linear and angular velocity errors when measured in the same frame/source.

## SMPL Retarget Risk

Severity: high for recent runs

The existing SMPL playback artifact indicates that the physically simulated target does not remain valid even when driven toward reference targets. This points to retargeting, morphology, height alignment, or PD/gain mismatch before PPO.

