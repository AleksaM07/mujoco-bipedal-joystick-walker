# MuJoCo Bipedal Joystick Walker

This repository contains a MuJoCo/MJX reinforcement-learning pipeline for a generated human biomechanics model. The goal is joystick-controlled bipedal locomotion: the policy receives a walking command and learns how to move the generated humanoid while staying upright, reducing foot sliding, and keeping physically plausible contacts.

The project also keeps a Berkeley/MuJoCo Playground humanoid baseline. That baseline is useful because it shows what a tuned locomotion benchmark can do, while the generated biomechanics model shows the harder research problem: training a custom anatomical model whose contacts, joint ranges, actuators, rewards, and observations had to be designed and debugged locally.

## What Was Built

- A generated MuJoCo human model pipeline in `biomechanics_model.py`.
- A joystick locomotion environment in `biomechanics_env.py`.
- PPO training and checkpointing in `train.py`.
- Headless and viewer evaluation in `evaluate.py`.
- BVH/reference-gait support in `bvh_reference.py`.
- A full analysis notebook in `walking_analysis.ipynb`.
- CSV exports in `analysis_outputs/` for checkpoint selection, rollout steps, trials, episodes, policy metrics, and actuator metrics.

## Current Research Summary

The PPO pipeline works: generated-human policies can be trained, resumed, evaluated, and compared across checkpoints. The best current policies move under joystick commands, but the core research issue is still gait quality. Raw reward alone is not enough, because a humanoid can exploit contact and velocity rewards by sliding or using unnatural support patterns.

The current direction is therefore not simply "more timesteps". The useful direction is better locomotion structure:

- contact-aware foot slip penalties,
- swing/stance quality metrics,
- stronger joint limits and actuator priors,
- BVH/reference gait targets,
- root and foot tracking,
- comparison against a tuned Berkeley baseline.

## Best Checkpoint Analysis

Run the notebook to reproduce the comparison:

```powershell
python -m uv sync
python -m uv run jupyter nbconvert --to notebook --execute walking_analysis.ipynb --inplace
```

The notebook compares the best checkpoint from every folder under `runs/successful`. It exports reusable CSV files to `analysis_outputs/` and includes filtered top-policy plots so the results are readable for presentation.

Current top in-distribution leaderboard from the generated analysis:

| rank | policy_id | policy_type | composite_score | mean_first_fall_survival_fraction | tracking_rmse | mean_torso_up |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | P08_v16_standard_bvh | biomechanics | 0.7490 | 1.0000 | 0.1618 | 0.9416 |
| 2 | P09_berkeley_flat | berkeley | 0.6092 | 0.9502 | 0.5778 | 0.9888 |
| 3 | P04_v16_forward_bvh | biomechanics | 0.6043 | 0.9307 | 0.2604 | 0.9414 |
| 4 | P05_v16_standard_bvh | biomechanics | 0.5620 | 0.9028 | 0.3229 | 0.9659 |
| 5 | P03_v16_forward_bvh | biomechanics | 0.5582 | 0.9567 | 0.3697 | 0.9332 |

The main score is a composite metric built from survival, command tracking, upright posture, and action smoothness. It is intentionally not just the training reward, because reward scales differ between environment generations and between Berkeley and generated-biomechanics policies.

## XML Model Comparison: Classic vs `trainfast_v19`

The latest training XML is not a completely different skeleton. It keeps the same high-level body, joint, and actuator structure, but it changes the learning surface around contacts, joint limits, damping, stiffness, and actuator ranges.

| Metric | Classic XML | `trainfast_v19` XML |
| --- | ---: | ---: |
| File size, bytes | 27840 | 31256 |
| Bodies | 16 | 16 |
| Joints | 34 | 34 |
| Geoms | 17 | 19 |
| Position actuators | 18 | 18 |
| Equality constraints | 0 | 0 |
| Changed named joint definitions | 0 baseline | 29 |
| Changed actuator definitions | 0 baseline | 18 |

Added named geoms in `trainfast_v19`:

- `left_foot_sole`
- `left_shank_illegal_contact`
- `left_thigh_illegal_contact`
- `pelvis_illegal_contact`
- `right_foot_sole`
- `right_shank_illegal_contact`
- `right_thigh_illegal_contact`

Removed named geoms in `trainfast_v19`:

- None

Practical interpretation:

- The same 18 controlled joints are still present.
- `v19` adds explicit foot sole boxes for stable floor contact.
- `v19` adds illegal-contact geoms for pelvis/thigh/shank guardrails.
- `v19` disables ordinary non-foot body contacts by using collision filtering.
- `v19` tightens many joint ranges and adds damping, stiffness, armature, and friction loss.
- `v19` replaces the very broad actuator `ctrlrange=-3.14 3.14` style with joint-specific ranges and stronger PD gains.

That means `v19` is meant to be a more learning-friendly locomotion XML, not a new anatomical body. It constrains the policy away from easy but bad solutions such as pelvis scraping, lower-leg floor contact, and over-large joint targets.

## Setup

This project uses Python 3.12 and `uv`.

```powershell
python -m uv sync
```

`uv` creates and maintains the local `.venv` from `pyproject.toml` and `uv.lock`.

## Common Commands

Show training options:

```powershell
python -m uv run python train.py --help
```

Show evaluation options:

```powershell
python -m uv run python evaluate.py --help
```

Run a short training smoke test on CPU:

```powershell
python -m uv run python train.py --debug-run --device cpu --allow-cpu --bare --no-checkpoints --timesteps 1000 --num-envs 4 --num-evals 0 --episode-length 20 --batch-size 4
```

Run the analysis notebook:

```powershell
python -m uv run jupyter nbconvert --to notebook --execute walking_analysis.ipynb --inplace
```

## Important Files

| Path | Purpose |
| --- | --- |
| `biomechanics_model.py` | Builds and patches generated MuJoCo human XML models. |
| `biomechanics_env.py` | Defines the RL environment, observations, rewards, contacts, and reset logic. |
| `train.py` | PPO training entry point. |
| `evaluate.py` | Checkpoint loading, viewer evaluation, and compatibility helpers. |
| `bvh_reference.py` | Lightweight BVH walking reference parser/retargeting support. |
| `walking_analysis.py` | Reusable analysis module for selecting best checkpoints and rolling out policies. |
| `walking_analysis.ipynb` | Human-readable report notebook with plots, top-policy filters, and XML comparison. |
| `analysis_outputs/` | CSV outputs created by the notebook. |
| `generated_models/` | Generated MuJoCo XML variants. |
| `runs/successful/` | Successful or useful training runs used by the analysis. |

## Analysis Outputs

The notebook writes these CSV files when the full analysis is run:

- `selected_checkpoints.csv`
- `training_history.csv`
- `rollout_steps.csv`
- `episode_metrics.csv`
- `trial_metrics.csv`
- `policy_metrics.csv`
- `actuator_metrics.csv`
- notebook summary CSVs for leaderboard, scenario top-3, and actuator hotspots.

`rollout_steps.csv` is intentionally large because it stores step-level trajectory, command, posture, contact, action, torque, and power signals.

## Limitations

- The generated model can learn locomotion-like movement, but visually natural walking is still hard.
- Berkeley is a tuned locomotion benchmark; the generated biomechanics model is a harder custom environment.
- Training reward is not enough evidence. Videos, contact metrics, fall rate, command tracking, and actuator diagnostics are all needed.
- The BVH pipeline is a staged approximation, not yet full GMR/DeepMimic-style retargeting.
- `trainfast_v19` improves contact guardrails and action priors, but it does not magically solve motion imitation.

## Short Explanation For A Professor

A PPO reinforcement-learning policy was trained on a generated MuJoCo human biomechanics model. The training environment includes joystick velocity commands, dense stability and tracking rewards, contact-aware gait diagnostics, actuator/action regularization, and later BVH-inspired walking references. A Berkeley/MuJoCo Playground humanoid is kept as a baseline because it is a prepared locomotion benchmark, while the generated human model requires custom reward, contact, joint-limit, and actuator design. The current result demonstrates a working training and evaluation pipeline, but also shows that natural-looking human gait requires stronger contact-aware or motion-imitation signals than velocity tracking alone.

## Merged Reference Log

The former `docs/REFERENCES_USED.md` content is preserved here so the README is self-contained.

This file records external references used while designing and debugging the
MuJoCo humanoid joystick walker. It is meant to be a project log, not a claim
that any code was copied directly.

## Core MuJoCo / Gymnasium References

- [Gymnasium Humanoid-v5](https://gymnasium.farama.org/environments/mujoco/humanoid/)
  - Used as the main simple MuJoCo humanoid baseline.
  - Important details: 17 torque actions, 348-dimensional default observation,
    and dense reward:
    `healthy_reward + forward_reward - ctrl_cost - contact_cost`.
  - Project lesson: simple forward walking can work with dense reward and rich
    physical observations, but this does not by itself produce human-like style.

- [MuJoCo](https://mujoco.org/)
  - Physics engine used by this project.
  - Project lesson: real contact information should be preferred over
    height-only pseudo-contact when penalizing foot slip.

## Unitree Humanoid / Legged RL References

- [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
  - Official Unitree RL implementation based on MuJoCo / mjlab.
  - Supports G1, G1-23DoF, H1_2, H2, Go2, A2, As2, and R1.
  - Local review path: `.tmp_unitree_rl_mjlab`.
  - Important files reviewed:
    - `.tmp_unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py`
    - `.tmp_unitree_rl_mjlab/src/tasks/velocity/mdp/rewards.py`
    - `.tmp_unitree_rl_mjlab/src/tasks/velocity/config/g1/env_cfgs.py`
    - `.tmp_unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py`
    - `.tmp_unitree_rl_mjlab/src/tasks/tracking/mdp/rewards.py`
    - `.tmp_unitree_rl_mjlab/src/tasks/tracking/config/g1/env_cfgs.py`
  - Project lessons:
    - Separate velocity tracking from motion imitation.
    - Velocity tracking uses command tracking, projected gravity, joint state,
      previous action, gait phase, contact sensors, foot clearance, foot slip,
      posture, action smoothness, and termination penalties.
    - G1-style configs use joint-specific posture/action priors: stride joints
      get more freedom, while hip roll/yaw, ankle roll, and waist joints stay
      tighter.
    - Motion imitation tracks root/anchor position, root orientation, relative
      body positions, body orientations, body linear velocities, and body
      angular velocities.
    - Their imitation target is not just a few joint angles; it is a full-body
      motion target.

- [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
  - Official Unitree RL workflow with Train -> Play -> Sim2Sim -> Sim2Real.
  - Supports G1, H1, H1_2, and Go2.
  - Local review path: `.tmp_unitree_rl_gym`.
  - Important files reviewed:
    - `.tmp_unitree_rl_gym/legged_gym/envs/g1/g1_env.py`
    - `.tmp_unitree_rl_gym/legged_gym/envs/g1/g1_config.py`
    - `.tmp_unitree_rl_gym/legged_gym/envs/base/legged_robot.py`
    - `.tmp_unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py`
  - Project lessons:
    - Uses phase observation for walking.
    - Uses contact forces for foot contact, swing height, and foot sliding
      penalties.
    - Uses PD position targets around a default pose, not raw arbitrary motion.
    - Uses realistic robot joint limits and action scale instead of giving RL a
      completely free generated body.
    - Uses domain randomization and pushes after the basic policy is working.

- [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
  - Unitree MuJoCo simulator/interface.
  - Project lesson: useful for testing/deploy-style simulation, less directly
    useful as the main training framework.

- [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
  - IsaacLab-oriented Unitree RL project.
  - Project lesson: useful architectural reference, but not the primary path
    for this MuJoCo-first project.

## Humanoid Benchmarks

- [carlosferrazza/humanoid-bench](https://github.com/carlosferrazza/humanoid-bench)
  - MuJoCo humanoid benchmark with H1/G1-style tasks.
  - Project lesson: useful as a benchmark reference for task definitions and
    evaluation, not necessarily a direct training recipe.

- [HumanoidBench Project Page](https://humanoid-bench.github.io/)
  - Project page for the benchmark.
  - Project lesson: confirms that humanoid locomotion in MuJoCo is treated as a
    difficult benchmark problem, not a trivial reward-only exercise.

## Motion Capture / Animation Sources

- [CMU Graphics Lab Motion Capture Database](https://mocap.cs.cmu.edu/)
  - Main intended source for real human walking motion clips.
  - Project lesson: best long-term path for human-looking walking is real motion
    imitation, but it needs proper retargeting.

- [CGSpeed CMU BVH Conversion](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-3dsmax-friendly-bvh-release-of-cmus-motion-capture-database)
  - BVH-format CMU motion capture files used in `BVH_walking_animation`.
  - Project lesson: BVH files are useful, but raw BVH channels must be retargeted
    carefully to our model.

- [una-dinosauria/cmu-mocap](https://github.com/una-dinosauria/cmu-mocap)
  - GitHub mirror/conversion of CMU mocap data.
  - Project lesson: convenient source for BVH data, but official CMU metadata is
    cleaner for citation.

- [AMASS](https://amass.is.tue.mpg.de/)
  - Large standardized human motion dataset.
  - Project lesson: good future option after the BVH pipeline is correct; too
    heavy for the first stable walker.

## BVH / DeepMimic / Retargeting References

- [rgalljamov/DRLoco](https://github.com/rgalljamov/DRLoco)
  - DeepMimic-style MuJoCo + Stable-Baselines3 project for legged robots using
    reference trajectories / mocap.
  - Local review path: `.tmp_drloco`.
  - Important files reviewed:
    - `.tmp_drloco/README.md`
    - `.tmp_drloco/drloco/mujoco/mimic_env.py`
    - `.tmp_drloco/drloco/mujoco/mimic_walker3d.py`
    - `.tmp_drloco/drloco/config/hypers.py`
  - Project lessons:
    - Imitation should be treated as a reference-trajectory task, not only as a
      forward velocity task.
    - The environment advances a reference frame with the simulation step.
    - Reward tracks reference qpos/qvel and COM-style motion, with alive reward
      and early-termination handling.
    - Ground contact is checked explicitly for left/right feet.
    - This supports adding reference velocity, foot/contact, and root signals to
      our BVH reward instead of relying only on six joint angle targets.

- [mingfeisun/DeepMimic_mujoco](https://github.com/mingfeisun/DeepMimic_mujoco)
  - Older Python/MuJoCo DeepMimic implementation with mocap playback and walk
    examples.
  - Local review path: `.tmp_deepmimic_mujoco`.
  - Important files reviewed:
    - `.tmp_deepmimic_mujoco/README.md`
    - `.tmp_deepmimic_mujoco/src/env/deepmimic_env_mujoco.py`
    - `.tmp_deepmimic_mujoco/src/data/motions/humanoid3d_walk.txt`
  - Project lessons:
    - DeepMimic reward is not just pose reward. It combines pose, velocity,
      end-effector, root, and COM terms.
    - Root position/rotation/velocity are part of the imitation signal.
    - End-effectors are compared relative to the root, which is why our new
      foot-position target is a better signal than joint angles alone.

- [robfiras/loco-mujoco](https://github.com/robfiras/loco-mujoco)
  - Modern MuJoCo/MJX locomotion and imitation benchmark with PPO, GAIL, AMP,
    and DeepMimic-style examples.
  - Local review path: `.tmp_loco_mujoco`.
  - Important files reviewed:
    - `.tmp_loco_mujoco/README.md`
    - `.tmp_loco_mujoco/examples/training_examples/jax_rl_mimic/README.md`
    - `.tmp_loco_mujoco/loco_mujoco/core/reward/trajectory_based.py`
    - `.tmp_loco_mujoco/loco_mujoco/trajectory/handler.py`
    - `.tmp_loco_mujoco/loco_mujoco/smpl/retargeting.py`
  - Project lessons:
    - Serious mimic rewards compare qpos, qvel, relative site positions,
      relative site orientations, and relative site velocities.
    - Trajectory/state handlers are first-class objects; reset and reward are
      aware of which reference trajectory/frame is active.
    - Their MJX humanoid models simplify or add foot primitive collision shapes
      for reliable foot contact, which supports our focus on foot contact
      sanity.
    - Our current code only approximates this: qpos/qvel plus root height,
      forward velocity, and foot relative positions. Full body/site imitation is
      still a future step.

- [Roboparty/GMR](https://github.com/Roboparty/GMR)
  - General Motion Retargeting pipeline for BVH/FBX/SMPL-X to robot motion.
  - Local review path: `.tmp_gmr`.
  - Important files reviewed:
    - `.tmp_gmr/README.md`
    - `.tmp_gmr/scripts/bvh_to_robot.py`
    - `.tmp_gmr/scripts/bvh_to_robot_dataset.py`
    - `.tmp_gmr/general_motion_retargeting/data_loader.py`
    - `.tmp_gmr/general_motion_retargeting/motion_retarget.py`
    - `.tmp_gmr/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json`
  - Project lessons:
    - The clean long-term path is BVH -> IK retargeted robot motion, producing
      full `root_pos`, `root_rot`, `dof_pos`, and body/link targets.
    - GMR already has a `scripts/bvh_to_robot.py` pipeline for LAFAN1/Nokov BVH
      to robot motion.
    - Our current BVH parser is a lightweight approximation for faster
      iteration, not a replacement for GMR-style retargeting.
    - Best next research step after stabilizing the current walker: adapt GMR
      output to our generated humanoid or to a supported robot model first, then
      train imitation on that full retargeted motion.

## Earlier Open-Source RL Repos Reviewed

- [Rishab-Agrawal/humanoid-curriculum-rl](https://github.com/Rishab-Agrawal/humanoid-curriculum-rl)
  - Local review path: `.tmp_rishab_humanoid_curriculum_rl`.
  - Project lesson: curriculum and staged difficulty matter.

- [ritwikrohan/DDPG-SAC-HumanoidWalking](https://github.com/ritwikrohan/DDPG-SAC-HumanoidWalking)
  - Local review path: `.tmp_ritwik_ddpg_sac_humanoidwalking`.
  - Project lesson: alternative algorithms exist, but the main blocker here is
    environment/reward/physics signal quality, not only PPO.

- [rohanpsingh/LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking)
  - Local review path: `.tmp_rohan_lhw`.
  - Project lesson: simpler humanoid models can look much easier because their
    tasks and bodies are easier than the generated biomechanics model.

- [roboterax/humanoid-gym](https://github.com/roboterax/humanoid-gym)
  - Local review path: `.tmp_roboterax_humanoid_gym`.
  - Project lesson: successful humanoid locomotion stacks usually include dense
    style/contact rewards and stronger task structure.

- [Jitu0110/RLMujoco](https://github.com/Jitu0110/RLMujoco)
  - Local review path: `.tmp_jitu_rlmujoco`.
  - Project lesson: MuJoCo examples are useful for reward/observation sanity
    checks, but not enough for human-looking walking alone.

## Berkeley / Legacy Project Reference

- `barkley_legacy_walking.py`
  - Local legacy placeholder for the earlier Berkeley-style walking setup that
    produced a visibly better first proof-of-work.
  - Project lesson: the old model likely worked with less effort because its
    body, contacts, observations, and reward landscape were easier than the
    generated biomechanics model.

## Current Project Lessons From These References

- Do not trust scalar reward alone; videos and decomposed reward metrics matter.
- Foot slip must use real contact/contact-force information when possible.
- Motion imitation should track root, body pose, body velocity, and end-effectors,
  not only six leg joint angles.
- BVH imitation should include a reference frame/phase, root target, foot or
  end-effector target, and eventually reference-state initialization.
- GMR-style IK retargeting is the best long-term BVH path, but it is a separate
  integration task; the current implementation is a staged approximation so we
  can test faster.
- Velocity tracking and motion imitation should be treated as different tasks or
  stages.
- Domain randomization and external pushes should come after the base gait is
  physically valid.
- A high reward with sliding is not success; it is a reward loophole.
- Unitree-style humanoid locomotion does not hard-lock feet. It combines
  physical joint ranges, small joint-position action targets, soft joint-limit
  penalties, foot slip/contact metrics, and illegal pelvis/hip/knee contact
  handling.

## Merged Model Learning Versions

The former `docs/MODEL_LEARNING_VERSIONS.md` content is preserved here so the README is self-contained.

Ovaj fajl je dnevnik glavnih iteracija ucenja. Poenta nije da svaka verzija ima
veci reward, nego da se zabelezi sta je stvarno nauceno i koji reward loophole je
otkriven.

## V0 - Berkeley/Barkley Baseline

**Ideja:** koristiti MuJoCo Playground Berkeley humanoid kao proof-of-concept za
joystick walking.

**Rezultat:** radi sa mnogo manje rucnog podesavanja.

**Zakljucak:** Berkeley nije dokaz da je custom generated human lak problem. To
je vec pripremljen locomotion benchmark sa dobrim modelom, kontaktima,
actuatorima, reward-om i PPO setupom.

## V1 - Generated Biomechanics Human, Initial Setup

**Ideja:** direktno trenirati generated human model.

**Rezultat:** mnogo treninga, malo korisnog hoda.

**Problem:** policy je imala premalo kontrole i previse losih lokalnih resenja:

- samo 12 leg actuatora,
- abdomen/pelvis prakticno zakljucani,
- reward je mogao da nagradi prezivljavanje/klizanje,
- bez gait phase signala,
- bez privileged critic signala.

**Zakljucak:** problem nije bio samo broj stepova. Task je bio lose postavljen za
PPO.

## V2 - 18 Actuators, Unlocked Stiff Trunk

**Ideja:** otkljucati abdomen i pelvis, ali ih drzati stiff kroz male action
scale-ove, jace PD limite i manji reset/random torque noise.

**Rezultat:** model postaje trenabilniji.

**Sta je dodato:**

- abdomen x/y/z actuators,
- pelvis x/y/z actuators,
- ukupno `nu=18`,
- privileged critic state,
- gait phase observation,
- action smoothing,
- veci PPO setup.

**Zakljucak:** ovo je bila potrebna infrastruktura. Bez toga generated human nije
imao dovoljno nacina da balansira.

## V3 - Forward Curriculum

**Ideja:** uciti prvo samo forward hod, bez punog joystick problema.

**Poznat dobar run:** forward run oko 90M stepova, reward oko `1400`.

**Rezultat:** policy ume da se krece napred i prezivi cele epizode.

**Problem:** forward-only policy ne zna pun joystick. Vizuelno i dalje moze da
izgleda zombi/klizavo.

**Zakljucak:** forward curriculum je koristan bootstrap, ali nije finalni cilj.

## V4 - Standard Joystick Fine-Tune

**Ideja:** nastaviti iz forward checkpoint-a na `command_profile=standard`.

**Rezultat:** reward oko `600+`, model reaguje na vise pravaca, ali kretanje i
dalje izgleda kao klizanje.

**Problem:** velocity tracking reward moze da se resi kontakt trikovima. Veci
reward nije automatski lepsi hod.

**Zakljucak:** pun joystick treba, ali ne sme biti jedini signal.

## V5 - Style V1 Reward

**Ideja:** pojacati stil hoda bez mocap-a:

- anti-slip/contact,
- swing foot clearance,
- torso/head posture,
- action smoothness,
- optional sine trajectory.

**Poznat run:** sine/reference trajectory run oko 60M, reward oko `715`.

**Rezultat:** malo bolje, ali nedovoljno. Hod i dalje nije profesor-ready.

**Problem:** proceduralna sine putanja daje ritam, ali ne garantuje dobar kontakt
i stvarne foot placements.

**Zakljucak:** sine reference je dobar debug alat, ali nije dovoljan kao finalni
style prior.

## V6 - BVH Reference Gait, Tier 1

**Ideja:** koristiti CMU/CGSpeed BVH walking clipove kao motion reference.

**Sta je dodato:**

- BVH parser/retargeting u `bvh_reference.py`,
- `--reference-gait bvh`,
- `--reference-gait-file` vise puta,
- `--reference-gait-list`,
- tier liste iz `BVH_walking_animation`:
  - tier1: vanilla forward walk,
  - tier2: lakse varijacije,
  - tier3: kompleksni/stilizovani hod,
  - uneven terrain/stairs posebno.

**Poznat run:**

```text
runs/biomechanics_noerfi_nodr_forward_ref_bvh_bvh_tier1_accurate_ppo_BiomechanicsHumanJoystickStandard_20260617_2118_60m_seed7_rew_1492p6453_best_1493p6858_s
```

**Rezultat:** reward oko `1493`, pune epizode, ali vizuelno i dalje kliza.

**Log signal:** pri kraju run-a:

- `episode_length=500`,
- `tracking ~= 490`,
- `ref_gait ~= 358`,
- `foot_slip ~= 386`,
- `swing_drag ~= 500`.

**Tumacenje:** policy je naucila da dobije visok reward, ali swing noga je cesto
ostajala u kontaktu, a stopala su klizila. Dakle BVH joint-angle tracking sam po
sebi nije dovoljan.

**Zakljucak:** potreban je jaci contact-aware anti-slip signal i kasnije bolji
mocap imitation koji prati stopala/root, ne samo joint uglove.

## V7 - Anti-Slip V2

**Ideja:** direktno zatvoriti reward loophole iz V6.

**Promena:**

- foot slip cost je sada kontakt-aware kvadrat brzine stopala,
- mala free-speed zona ignorise numericki jitter,
- slip penalty je znatno jaci,
- swing-foot drag penalty je znatno jaci,
- dodat je eksplicitni swing clearance deficit cost,
- swing clearance i stance contact reward su pojacani.

**Naknadna dijagnoza:** BVH tier1 model nije samo malo klizao; vizuelno skoro
uopste nije podizao noge. Zato anti-slip v2 sada posebno kaznjava situaciju gde
swing stopalo nije bar oko `8 cm` iznad stance stopala.

**Kriticni review nalaz:** ni ovo nije dovoljno ako policy ne vidi BVH target.
Prethodni BVH setup je nagradjivao pracenje random izabranog BVH clip/frame-a,
ali taj target nije bio u observation-u. To je skriven zadatak: policy ne zna
koji clip i frame treba da prati, pa uci prosecan klizavi kompromis.

## V8 - BVH Target-Conditioned Policy

**Ideja:** ukloniti skriveni BVH target iz reward-a.

**Promena:**

- Novi BVH trening dodaje trenutni reference target u policy observation.
- `state` observation za BVH target-conditioned run raste sa `98` na `116`.
- Command slice ostaje `9:12`, tako da joystick evaluator i dalje menja isti deo
  observation-a.
- Proceduralni gait/no-lift phase za BVH vise ne koristi nezavisni random sine
  clock, nego aktivni BVH phase.
- Swing noga za BVH reward se izvodi iz reference knee flexion-a, pa no-lift
  kazna vise ne moze da bude u konfliktu sa BVH frame-om.

**Ocekivanje:** ovo je prvi run u kome policy zaista zna koji BVH pose treba da
imitira. Stari BVH run-ovi nisu validan dokaz da motion imitation ne radi, jer
su bili delom non-Markov zadatak.

**Training note:** reward moze biti manji nego V6, ali ako hod izgleda bolje, to
je uspeh. Za ovaj projekat vizuelni gait kvalitet je vazniji od broja `1500`.

**Speed/debug note:** za prve provere koristi
`BVH_walking_animation/tier1_debug_10.txt`. To je mali subset od 10 vanilla walk
clipova, svi kraci od 5 sekundi, pa jedan 10s episode vidi ceo clip i loop.

**Foot contact experiment:** v16 XML uklanja dodati box/padding sole. Umesto toga
originalna generated foot capsule geometrija dobija ime `left_foot_sole` /
`right_foot_sole` i koristi se za kontakt/reward. Ako se pokaze gore, v15 box sole
moze da se vrati.

**Run result:** target-conditioned v16 debug run je poceo da dize noge, ali i
dalje pada. To je pomak: problem se prebacio sa no-lift/sliding na balans tokom
koraka.

**Next curriculum:** `forward_slow` sada prvo uci stabilan spor korak, pa tek
onda pun `forward`. Log iz resume run-a pokazao je da je `done_low` skoro uvek
glavni razlog pada, pa je sledeci rez balance-first:

- sporiji command `0.02-0.12 m/s`,
- 25% epizoda sa zero command radi stabilnog stajanja,
- manji velocity/progress reward za `forward_slow`,
- jaci base-height reward,
- jaci low-height cost,
- jaci fall penalty.

**Sledeci test:**

```bash
python train.py --device gpu --command-profile forward_slow --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 10000000 --num-envs 1024 --num-evals 5 --no-erfi --no-domain-randomization --no-checkpoints --run-tag bvh_target_obs_v16_balance_debug10
```

## Current Diagnosis

Najveci problem vise nije "ne ume da hoda". Ume da dobije reward i da prezivi.
Najveci problem je da reward jos uvek nije dovoljno vezan za fizicki uverljiv
kontakt stopala.

Kratko:

```text
velocity tracking + joint reference != natural walking
natural walking needs contact-aware foot constraints
```

## Next Research Direction

Najbolji sledeci pravac nije jos veci broj stepova, nego bolji imitation/contact
reward:

- contact-aware foot sliding penalty,
- stance foot should stay planted,
- swing foot should clear the ground,
- BVH/reference foot position tracking,
- root orientation/height tracking,
- optional behavioral cloning or motion prior later.

## V9 - Reward/Contact Sanity Fix

**Problem:** reward je mogao da bude visok dok video i dalje pokazuje klizanje.
Stari `clip(reward, 0, 3)` je gasio negativne per-step signale, pa su slip i
drag cesto samo smanjivali bonus umesto da stvarno kazne los kontakt.

**Promene:**

- Reward clip je promenjen na `[-5, 3]`.
- Fall penalty ostaje `-25`.
- `_foot_contact` sada koristi MuJoCo contact geom parove sa floor geom-om,
  umesto samo visinu stopala.
- `_minimum_geom_z` sada razlikuje box, sphere i capsule geometriju, pa v16
  capsule foot sole vise ne koristi box-corner pretpostavku.
- Dodan je `REFERENCES_USED.md` sa svim spoljnim referencama i lekcijama.

**Ocekivanje:** sledeci run moze imati manji reward nego stari sliding run, ali
treba da bude iskreniji. Ako reward opadne a video manje kliza, to je pobeda.

**Sud posle Unitree review-a:** nas trenutni BVH sistem je i dalje samo
lightweight pose reference. Unitree-style imitation trazi root/body pose,
orijentacije, brzine i end-effector tracking. To je sledeci ozbiljan korak ako
V9 ne resi klizanje dovoljno.

## V10 - Gymnasium + Unitree Signal Extraction

**Ideja:** izvuci korisne delove iz Gymnasium Humanoid i Unitree RL Mjlab bez
prepisivanja celog framework-a.

**Promene:**

- Privileged critic observation sada dobija Gymnasium-style fizicke signale:
  `cinert`, `cvel`, i `cfrc_ext`.
- Policy observation ostaje manji i cistiji; veliki physics signali idu samo
  critic-u.
- Dodata je mala Gymnasium-style contact-force kazna iz `cfrc_ext`.
- BVH loader sada pravi i `qvel_targets` iz retargetovanih poza.
- BVH reward sada ima `reference_velocity` pored `reference_gait`, sto je blize
  Unitree tracking ideji: ne prati se samo staticka poza nego i brzina pokreta.
- Train log sada prikazuje `ref_vel` i `contact_force`.

**Vazna kompatibilnost:**

- `state` policy obs ostaje `98` za obican run i `116` za BVH target-conditioned
  run.
- `privileged_state` critic obs raste, npr. sa `151` na `525` za obican run.
- Zato je ovo najbolje tretirati kao novi trening iz pocetka, ne kao resume
  starog checkpoint-a.

**Sta jos nedostaje za pravi Unitree-style imitation:**

- root/anchor position target,
- root/anchor orientation target,
- relative body positions,
- body orientations,
- body linear/angular velocities,
- explicit end-effector/foot position tracking.

## V11 - Unitree/Barkley-Style Anatomical Action Prior

**Problem iz run-a:** `gym_unitree_v10_debug10` se popravio sa negativnog reward-a
na oko `106`, ali `episode_length` je ostao oko `100` i `done_low` je prakticno
stalno aktivan. To znaci da policy dobija neke lokalne signale, ali telo i dalje
ima previse slobode da nadje nestabilne/glupe pokrete umesto stabilnog koraka.

**Ideja:** ne zakljucavati noge u animaciju, nego ograniciti prostor akcija kao
kod Unitree/Barkley-style modela:

- stride zglobovi smeju vise da rade: hip_x, knee_z, ankle_y,
- bocne/uvrtajuce ose smeju mnogo manje: hip_y, hip_z, ankle_z,
- abdomen/pelvis ostaju kontrolisani, ali mali i kruti,
- joint range u XML-u vise ne dozvoljava ekstremne generated-model polozaje.

**Promene:**

- Novi XML cache version: `trainfast_v17`.
- Dodan `LEG_ACTION_SCALE` u `biomechanics_env.py`.
- Action scale vise nije uniforman za noge:
  - hip_x `0.35`,
  - knee_z `0.55`,
  - ankle_y `0.24`,
  - hip_y `0.12`,
  - hip_z `0.14`,
  - ankle_z `0.08`.
- `--action-scale` i dalje radi kao globalni multiplier preko ovih odnosa.
- Dodan variable posture prior:
  - dok stoji, svi zglobovi imaju stroge tolerancije,
  - dok hoda, hip/knee/ankle pitch dobijaju veci prostor,
  - trunk i lateral/twist ose ostaju stroze.
- Train log sada prikazuje `var_pose`.
- U `biomechanics_model.py` dodan `LEG_JOINT_SPECS`, pa v17 XML ima strozije
  humanoidne joint range vrednosti.

**Provera:**

- `python -m py_compile biomechanics_model.py biomechanics_env.py train.py`
  prosao.
- Env je generisao i ucitao
  `generated_models/human_male_180cm_75kg_standard_trainfast_v17.xml`.
- JAX reset/step smoke test je prosao, ali je prvi compile bio spor.

**Vazna kompatibilnost:**

- Ovo tretirati kao trening iz pocetka.
- Ne bih resume-ovao stare v16/v10 checkpoint-eve, jer su dinamika, action prior
  i XML joint range promenjeni.

**Sledeci test koji ima smisla:**

```bash
python train.py --device gpu --command-profile forward_slow --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 20000000 --num-envs 1024 --num-evals 10 --no-erfi --no-domain-randomization --run-tag unitree_prior_v11_debug10
```

Ako do `5-10M` stepova `episode_length` ne krene jasno iznad `200`, problem je
jos uvek u stabilnosti/initial pose/contact dinamici, ne u broju trening koraka.

## V12 - DeepMimic/LocoMuJoCo/GMR Reference Review

**Problem:** BVH pipeline je krenuo u dobrom smeru, ali je i dalje bio preslab u
odnosu na ozbiljne imitation sisteme. Nas signal je bio blizu "prati nekoliko
leg joint uglova", dok DeepMimic/LocoMuJoCo-style pristup prati root, brzine,
end-effectors/sites i cesto resetuje epizodu blizu reference state-a.

**Repo-i pregledani:**

- DRLoco,
- DeepMimic_mujoco,
- LocoMuJoCo,
- GMR.

**Sta je nauceno:**

- Pravi imitation reward nije samo joint pose.
- Treba root target: visina, orijentacija i/ili brzina.
- Treba end-effector target: stopala u odnosu na root/trunk.
- Reference frame/phase mora biti eksplicitan.
- Reference-state initialization moze mnogo da pomogne, ali je rizican flag i
  treba ga testirati odvojeno.
- GMR je najbolji dugorocni BVH retargeting put, ali zahteva poseban integration
  sloj za nas model ili izbor podrzanog robot modela.

**Promene u nasem kodu:**

- BVH sada cuva approximate `root_height_offsets`.
- BVH sada cuva approximate `root_forward_velocity_factors`.
- Env precomputuje BVH qpos/qvel targete.
- Env precomputuje trunk-relative left/right foot targete preko MuJoCo FK.
- Reward sada ima `ref_foot`.
- Reward sada ima `ref_root`.
- Dodati su opcioni `reference_phase_randomization` i `reference_state_init`.
- `reference_state_init` je fixovan da ne pocne prvi step sa pogresnim
  `last_action`.
- Dodan je `reference_sanity.py`.

**Current run status:** V10 forward resume nije propao. Do sada je stigao do:

```text
step=53,739,520
reward=960.2457
episode_length=457.5625
done_low=0.312
checkpoint=000053739520
```

**Sledeci test:** ne pustati sve novo odjednom. Prvi gate treba da testira samo
novi root/foot reward signal iz kompatibilnog V10 checkpoint-a:

```bash
python train.py --device gpu --command-profile forward --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --resume-from runs/biomechanics_noerfi_nodr_forward_ref_bvh_v10_18350_resume_forward_accurate_ppo_BiomechanicsHumanJoystickStandard_20260621_0254_80m_seed7_running/checkpoints/000053739520 --xml-path generated_models/human_male_180cm_75kg_standard_trainfast_v16.xml --legacy-action-prior --timesteps 8000000 --num-envs 1024 --num-evals 4 --num-eval-envs 16 --no-erfi --no-domain-randomization --run-tag v10_ref_root_foot_gate
```

**Ne ukljucivati u prvom gate-u:**

- `--reference-state-init`,
- `--reference-phase-randomization`,
- full tier1 list,
- ERFI/domain randomization.

**Zakljucak:** ovo je najbolji sledeci korak ako ne krecemo iz pocetka. Ako gate
ne pomogne, sledeci veliki korak je GMR-style retargeting, a ne jos jedan
nasumicni 80M PPO run.
