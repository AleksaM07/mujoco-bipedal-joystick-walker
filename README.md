# MuJoCo Bipedal Joystick Walker

Reinforcement learning project for a biomechanical humanoid walker controlled by
joystick-style velocity commands. The active training stack is MuJoCo/MJX,
MuJoCo Playground wrappers, Brax PPO, JAX, and BVH walking references.

## Active Entry Points

- `train.py` - PPO training and checkpoint resume flow.
- `evaluate.py` - MuJoCo viewer rollout, keyboard joystick control, and headless
  inspection mode.
- `biomechanics_env.py` - custom humanoid locomotion environment and reward.
- `biomechanics_model.py` - generated humanoid XML adaptation for training.
- `training_wrappers.py` - vectorized training wrappers and per-episode domain
  randomization.
- `bvh_reference.py` - BVH loading and retargeting to MuJoCo actuator order.
- `tools/reference_sanity.py` - quick checks for BVH reference target ranges.
- `legacy/barkley_legacy_walking.py` - isolated legacy Berkeley/Playground training
  path, kept out of the active biomechanics code.
- `docs/` - research notes, optimization notes, references, and old comparison
  writeups.
- `assets/` - small checked-in model/input files used for inspection or resume
  reproducibility.
- `artifacts/` - local videos and other presentation outputs, ignored by Git.

## Setup

Use Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The model builder expects the sibling `mujoco-biomechanics` repository when it
needs to generate a new XML. For exact resume/evaluation, pass the saved
`--xml-path` from the run config.

## Training Example

```bash
python train.py ^
  --device gpu ^
  --env-version standard ^
  --reference-gait bvh ^
  --reference-gait-list BVH_walking_animation/tier1_forward_walk.txt ^
  --run-tag presentation_bvh ^
  --timesteps 1000000 ^
  --num-envs 512
```

## Evaluation Example

```bash
python evaluate.py ^
  --checkpoint runs/biomechanics_ref_bvh_v10_best87040_erfi_dr_180m_accurate_ppo_BiomechanicsHumanJoystickStandard_20260625_2136_180m_seed7_running/checkpoints/000085196800 ^
  --device gpu ^
  --command-x 0.25
```

For a headless smoke test before opening the viewer:

```bash
python evaluate.py --checkpoint PATH_TO_CHECKPOINT --inspect --inspect-steps 2000
```

BVH target sanity check:

```bash
python -m tools.reference_sanity --reference-gait-list BVH_walking_animation/tier1_forward_walk.txt
```

Regenerate BVH tier lists:

```bash
python -m tools.build_walk_tiers
```

## Presentation Run

Current best candidate from the open run log:

- Run: `runs/biomechanics_ref_bvh_v10_best87040_erfi_dr_180m_accurate_ppo_BiomechanicsHumanJoystickStandard_20260625_2136_180m_seed7_running`
- Notable eval: step `85,196,800`, reward `1038.56`
- Checkpoint: `checkpoints/000085196800`
- XML: `generated_models/human_male_180cm_75kg_standard_trainfast_v16.xml`

## Artifact Policy

Training runs, checkpoints, local videos, scratch clone folders, and generated
artifacts are local outputs. They are ignored for future changes by `.gitignore`.
Some older artifacts are already tracked in Git; if you want a slim presentation
repository, untrack them with `git rm --cached` while keeping the files on disk.
