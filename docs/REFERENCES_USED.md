# References Used

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
