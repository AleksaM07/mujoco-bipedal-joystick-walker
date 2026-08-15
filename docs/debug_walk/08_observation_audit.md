# Observation Audit

## Local Observation Structure

Relevant code:

- `biomechanics_env.py:2292` builds the observation vector.
- Observation includes local root velocity, angular velocity, projected gravity, command, phase, reference target observation, joint position/velocity, and last action.

## Comparison With MimicKit

Reference:

- `.tmp_mimickit/data/envs/deepmimic_humanoid_env.yaml:12` enables target observations.
- `.tmp_mimickit/data/envs/deepmimic_humanoid_env.yaml:14` uses target steps `[1, 2, 3]`.
- `.tmp_mimickit/data/envs/deepmimic_humanoid_env.yaml:15` enables randomized reset.

Local defaults:

- `config.py:673` sets `bvh_target_observation_steps=(0,)` in the default environment config.
- A later dataclass default at `config.py:814` uses `(1, 2, 3)`, but the main default object uses `(0,)`.

Finding:

Severity: medium

Observation design is not obviously broken, but the default target-step setting differs from MimicKit. Current-step-only target observations can make tracking more reactive and less anticipatory than the MimicKit baseline.

Training effect:

- This probably does not explain total walking failure by itself.
- It can make a difficult reference tracking task harder once the lower-level oracle issues are fixed.

Minimal fix:

- For debug parity with MimicKit, use future target steps `[1, 2, 3]`.
- Keep the phase observation only if it empirically helps after the reference oracle passes.

Test:

- Check observation size and target-step values in the env config printed at training startup.
- Train only after PD tracking and reward exact-match invariants pass.

