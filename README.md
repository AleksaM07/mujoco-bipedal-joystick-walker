# MuJoCo Bipedal Joystick Walker

Reinforcement-learning pipeline for joystick-controlled walking on a generated
MuJoCo human biomechanics model.

The project trains PPO policies, evaluates the best checkpoints, compares them
with a Berkeley/MuJoCo Playground humanoid baseline, and exports analysis tables
and plots.

## What This Project Shows

- PPO training works on the generated humanoid model.
- Checkpoints can be resumed, evaluated, and compared.
- Berkeley humanoid is a useful tuned baseline.
- The generated biomechanics model is harder: reward, contacts, actuators, joint
  limits, and XML variants matter a lot.
- High reward alone is not trusted; the analysis also checks falls, command
  tracking, posture, foot/contact behavior, action smoothness, and actuator load.

## Main Files

| File | Purpose |
| --- | --- |
| `train.py` | PPO training entry point. |
| `evaluate.py` | Load and visualize/evaluate checkpoints. |
| `biomechanics_env.py` | RL environment, observations, rewards, contacts. |
| `biomechanics_model.py` | Generated MuJoCo model/XML construction. |
| `bvh_reference.py` | BVH walking reference support. |
| `walking_analysis.py` | Reusable checkpoint and rollout analysis code. |
| `walking_analysis.ipynb` | Report notebook with plots and conclusions. |
| `analysis_outputs/` | Generated CSV analysis outputs. |
| `generated_models/` | Classic and training XML variants. |

## Setup

```powershell
python -m uv sync
```

## Common Commands

```powershell
python -m uv run python train.py --help
python -m uv run python evaluate.py --help
```

Run the analysis notebook:

```powershell
python -m uv run jupyter nbconvert --to notebook --execute walking_analysis.ipynb --inplace
```

Quick CPU smoke test:

```powershell
python -m uv run python train.py --debug-run --device cpu --allow-cpu --bare --no-checkpoints --timesteps 1000 --num-envs 4 --num-evals 0 --episode-length 20 --batch-size 4
```

## Current Best-Checkpoint Analysis

The notebook selects the best logged checkpoint from each run under
`runs/successful`, runs fixed command scenarios, and exports:

- `selected_checkpoints.csv`
- `training_history.csv`
- `rollout_steps.csv`
- `episode_metrics.csv`
- `trial_metrics.csv`
- `policy_metrics.csv`
- `actuator_metrics.csv`

Current top in-distribution policies:

| Rank | Policy | Type | Composite | Survival | Tracking RMSE | Torso Up |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `P08_v16_standard_bvh` | biomechanics | 0.7490 | 1.0000 | 0.1618 | 0.9416 |
| 2 | `P09_berkeley_flat` | berkeley | 0.6092 | 0.9502 | 0.5778 | 0.9888 |
| 3 | `P04_v16_forward_bvh` | biomechanics | 0.6043 | 0.9307 | 0.2604 | 0.9414 |

Composite score combines survival, command tracking, upright posture, and action
smoothness. It is more useful than raw training reward for cross-run comparison.

## Classic XML vs `trainfast_v19`

`trainfast_v19` keeps the same basic humanoid structure, but changes the model
to be more suitable for locomotion training.

| Metric | Classic | `trainfast_v19` |
| --- | ---: | ---: |
| Bodies | 16 | 16 |
| Joints | 34 | 34 |
| Geoms | 17 | 19 |
| Position actuators | 18 | 18 |
| Changed joint definitions | baseline | 29 |
| Changed actuator definitions | baseline | 18 |

Important `v19` changes:

- adds `left_foot_sole` and `right_foot_sole` box contacts,
- marks pelvis/thigh/shank contacts as illegal guardrails,
- filters non-foot contacts,
- tightens many joint ranges,
- adds damping/stiffness/friction/armature,
- replaces broad actuator ranges with joint-specific `ctrlrange`, `kp`, and
  `forcerange`.

Short version: same humanoid, stricter and more training-friendly physics.

## Reference/Learning Summary

Reviewed references:

- Gymnasium Humanoid,
- MuJoCo,
- Unitree RL MuJoCo/Gym/Lab projects,
- HumanoidBench,
- CMU/CGSpeed BVH motion data,
- DeepMimic/DRLoco/LocoMuJoCo/GMR-style imitation projects.

Main lessons:

- velocity tracking alone does not guarantee natural walking,
- real foot/contact metrics are necessary,
- BVH imitation should eventually track root, feet, velocities, and body targets,
- domain randomization should come after a stable base gait,
- high reward with sliding is a reward loophole, not success.

## Model Evolution Summary

- Early versions proved the generated model can be trained, but gait quality was
  poor.
- Later versions unlocked/stabilized trunk control and moved to 18 actuators.
- BVH/reference gait was added to push the policy toward human-like motion.
- Contact-aware anti-slip, swing/stance checks, and foot/root reference terms
  were added after sliding was observed.
- `trainfast_v17-v19` added stricter joint/action priors and contact guardrails.

## Practical Conclusion

The project is a working PPO locomotion pipeline, not a claim that human walking
is fully solved.

Best framing:

- Berkeley baseline: cleaner tuned locomotion benchmark.
- Generated biomechanics model: harder custom model successfully trained, but
  still limited by gait/contact quality.
- Next research step: stronger contact-aware imitation or full-body retargeted
  motion data.
