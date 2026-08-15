# Reset And RSI Audit

## Reset Path

Relevant code:

- `biomechanics_env.py:1560` implements reset.
- `biomechanics_env.py:1741` initializes `motion_step` to zero.
- `biomechanics_env.py:1750` stores `bvh_reference_time_offset`.
- `biomechanics_env.py:3536` samples exact or projected initial states from the reference.

Finding:

- Exact reset can produce near-perfect BVH reward at the sampled state.
- Therefore reset is not globally disconnected from the reward target.

## Failure Mode

Severity: high

Randomized reference-state initialization becomes harmful when combined with many short clamp clips:

- A reset near the end of a clamp clip can terminate quickly via `motion_over`.
- Existing SMPL playback evidence shows all sampled episodes failed or ended early, with `valid=0/4`.
- Training logs from recent runs show short evaluation episodes and frequent low-height, motion-over, and pose-termination endings.

Training effect:

- The rollout distribution is dominated by short, terminal fragments.
- Learning pressure can be fragmented across many transient poses rather than one stable walking manifold.

Minimal fix:

- For the first debug training target, use one clip and deterministic reset at phase `0.0`.
- After PD tracking survives, add randomized phase reset.
- Only then add multiple clips.

Test:

```powershell
python tools\debug_reference_pd_tracking.py --reference-gait smpl --clip-id 0 --steps 180
python tools\debug_reference_pd_tracking.py --reference-gait bvh --clip-id 0 --steps 180
```

Pass condition: no low-height or tipping failure during at least one meaningful gait segment.

