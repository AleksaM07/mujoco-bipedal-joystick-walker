# Optimization And Locomotion Changes Applied

Ovaj fajl opisuje promene koje su uvedene posle pregleda tvog biomehanickog
humanoid env-a i poredjenja sa referentnim projektima u `.tmp_*` folderima.

Legenda za poreklo:

- `Seen in other projects`: ideja postoji u nekom od pregledanih repo-a.
- `Adapted`: ideja postoji drugde, ali je ovde promenjena zbog naseg XML-a/env-a.
- `Local fix`: uvedeno zbog konkretnog buga ili ogranicenja u ovom projektu.

## Reference Repos Checked

- `.tmp_rishab_humanoid_curriculum_rl`
- `.tmp_ritwik_ddpg_sac_humanoidwalking`
- `.tmp_rohan_lhw`
- `.tmp_roboterax_humanoid_gym`
- `.tmp_jitu_rlmujoco`
- `.tmp_drloco`
- `.tmp_deepmimic_mujoco`
- `.tmp_loco_mujoco`
- `.tmp_gmr`

## High-Level Diagnosis

Stari setup je bio previse tezak za PPO:

- RL je kontrolisao samo 12 leg actuatora.
- Abdomen/pelvis su bili prakticno zakljucani equality constraint-ovima.
- Reward je mogao da nagradi pasivno klizanje i prezivljavanje vise nego pravi hod.
- Policy nije imala gait phase signal, a critic nije imao privileged informacije.
- PPO config je bio mali za ovakav humanoid model.

Drugim recima: nije bilo samo "treba vise koraka". Problem je bio lose postavljen
za ucenje.

## Model / XML Changes

### 1. New XML cache version: `trainfast_v14`

**Files:** `biomechanics_model.py`

**Change:**

- `SCENE_XML_VERSION` je bumpovan na `trainfast_v14`.
- Time se forsira generisanje novog XML-a umesto tihog koriscenja starog v12/v13
  cache-a.

**Source:** Local fix.

**Why:**

- Stari XML je imao trunk lockove.
- v13 je tokom provere uhvacen bez `left_foot_sole/right_foot_sole`, pa je v14
  uveden kao cist cache target.

### 2. Abdomen and pelvis unlocked

**Files:** `biomechanics_model.py`

**Change:**

- Dodati kontrolisani trunk zglobovi:
  - `abdomen_x`
  - `abdomen_y`
  - `abdomen_z`
  - `pelvis_x`
  - `pelvis_y`
  - `pelvis_z`
- Ukupan broj kontrolisanih actuatora je sada 18, ne 12.
- Equality lockovi za abdomen/pelvis se uklanjaju tokom XML build-a.
- Provereno na novom XML-u: `nu=18`, `neq=0`.

**Source:** Adapted.

**Seen in other projects:**

- `Rishab-Agrawal/humanoid-curriculum-rl` i `Jitu0110/RLMujoco` koriste stock
  MuJoCo Humanoid stil gde abdomen ima aktuatore.
- `Jitu0110/RLMujoco/Code/Humanoid_v4.py` dokumentuje abdomen actuators i
  abdomen state u observation-u.

**Local adaptation:**

- Nas generated model ima dodatne pelvis hinge joints. Zato su i pelvis zglobovi
  otkljucani, ali sa manjim opsegom i jacim PD-om da ostanu stiff.

### 3. Stiff trunk joint specs

**Files:** `biomechanics_model.py`

**Change:**

- Abdomen/pelvis vise nisu fiksirani na `-0.001 0.001`.
- Dobili su mali anatomski opseg, damping, stiffness, friction i armature.
- Ideja je: policy sme da koristi trup za balans, ali ne sme da ga pretvori u
  gumeni motor.

**Source:** Adapted.

**Seen in other projects:**

- Rishab custom humanoid XML ima abdomen joints sa damping/stiffness i motorima.
- Stock Humanoid-style env-ovi kontrolisu abdomen.

### 4. Stronger position actuators with joint-specific limits

**Files:** `biomechanics_model.py`

**Change:**

- Uveden `ACTUATOR_SPECS` po joint-u.
- Trunk actuators imaju male `ctrlrange` vrednosti:
  - abdomen oko `0.14-0.18 rad`
  - pelvis oko `0.10-0.12 rad`
- Leg actuators imaju snaznije PD vrednosti nego stari generic `kp=35`.

**Source:** Local adaptation.

**Why:**

- Stari generic actuator setup je bio preslab i previse uniforman.
- Trup mora biti kontrolisan drugacije od kuka/kolena/stopala.

### 5. Compiler angle fix

**Files:** `biomechanics_model.py`

**Change:**

- `compiler angle` se eksplicitno postavlja na `degree`.

**Source:** Local fix.

**Why:**

- Base generated XML joint ranges su u stepenima. Ako se pogresno tretiraju kao
  radijani, model dobije besmislene joint limits.

### 6. Stable foot sole geoms

**Files:** `biomechanics_model.py`

**Change:**

- Dodati/vraceni su:
  - `left_foot_sole`
  - `right_foot_sole`
- To su box geometrije ispod stopala, ne novi zglobovi i ne novi actuatori.
- Koriste se za stabilniji kontakt i za reward signale.

**Source:** Local fix.

**Why:**

- Generated foot capsule geometrije nisu dovoljno jasan kontakt signal za RL.
- v13 provera je pukla jer env ocekuje ova imena za foot reward/critic signal.

### 7. Collision filtering for training

**Files:** `biomechanics_model.py`

**Change:**

- Non-foot geometrije su visual-only za kontakt.
- Teren prima kontakt od foot sole geometrija.

**Source:** Local fix.

**Why:**

- Smanjuje nepotrebne self/contact parove.
- Daje jasniji signal: stopala su primarni kontakt sa podom.

## Environment Changes

### 8. Per-joint action scale

**Files:** `biomechanics_env.py`

**Change:**

- Action scale vise nije jedan broj za sve.
- Trunk ima mali action scale:
  - abdomen: `0.06-0.08`
  - pelvis: `0.04-0.05`
- Leg joints zadrzavaju normalni `action_scale`.

**Source:** Local adaptation.

**Why:**

- Trup treba da balansira sitno, ne da pravi velike pokrete kao noge.

### 9. Reduced trunk torque injection

**Files:** `biomechanics_env.py`

**Change:**

- ERFI/RAO torque injection se skalira po actuatoru.
- Trunk dobija samo `0.2x` torque perturbacije u odnosu na noge.

**Source:** Local adaptation.

**Why:**

- Ako trunk dobije isti random torque kao noge, lako destabilizuje humanoida pre
  nego sto policy nauci osnovni balans.

### 10. Per-actuator reset noise

**Files:** `biomechanics_env.py`

**Change:**

- Reset noise se primenjuje samo na kontrolisane actuators qpos.
- Trunk noise je manji (`0.005`) od leg noise-a (`0.02`).

**Source:** Local adaptation.

### 11. Dict observations: policy state + privileged critic state

**Files:** `biomechanics_env.py`, `train.py`, `evaluate.py`

**Change:**

- Env sada vraca:
  - `state`: observation za policy.
  - `privileged_state`: siri observation za critic.
- Policy state shape: `98`.
- Privileged critic state shape: `151`.

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- `roboterax/humanoid-gym` koristi privileged observations za critic.
- Njihov runner koristi critic obs ako postoje.

**Local adaptation:**

- U Brax PPO-u je to mapirano kroz:
  - `policy_obs_key="state"`
  - `value_obs_key="privileged_state"`

### 12. Privileged critic signals

**Files:** `biomechanics_env.py`

**Change:**

`privileged_state` ukljucuje dodatno:

- root `qpos[:3]`
- root `qvel[:6]`
- `qfrc_actuator` za kontrolisane DoF-ove
- foot positions
- pseudo foot contact
- action scale vector

**Source:** Adapted.

**Seen in other projects:**

- Stock Humanoid-style env-ovi dokumentuju/use `qfrc_actuator`.
- `Jitu0110/RLMujoco` i `Rishab-Agrawal/humanoid-curriculum-rl` imaju
  Humanoid-v4 style observation dokumentaciju sa actuator/contact signalima.
- `roboterax/humanoid-gym` koristi privileged critic obs.

### 13. Gait phase observation

**Files:** `biomechanics_env.py`

**Change:**

- Policy state sada ukljucuje `sin(phase)` i `cos(phase)`.
- Command slice ostaje `9:12`, tako da evaluator i dalje moze da menja joystick
  komandu bez pomeranja indeksa.

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- `roboterax/humanoid-gym` koristi gait phase (`sin`, `cos`) i stance masks.
- `.tmp_rohan_lhw` walking tasks koriste phase/clock reward.

### 14. Action smoothing

**Files:** `biomechanics_env.py`, `config.py`, `train.py`, `evaluate.py`

**Change:**

- Policy action se filtrira:
  - `smoothed = action_smoothing * new + (1 - action_smoothing) * previous`
- Default `action_smoothing=0.5`.
- CLI opcija: `--action-smoothing`.

**Source:** Adapted.

**Seen in other projects:**

- Humanoid locomotion projekti cesto filtriraju/clipuju actions ili PD targets.
- `roboterax/humanoid-gym` i sim2sim kod imaju action clipping/torque limiting.

## Reward Changes

### 15. Dense humanoid reward scaffold

**Files:** `biomechanics_env.py`

**Change:**

Reward sada kombinuje:

- alive reward
- velocity tracking
- forward progress gated by tracking
- upright reward
- base height reward
- posture reward
- gait reward
- action cost
- action rate cost
- trunk posture cost
- foot slip cost
- height cost
- overspeed cost
- vertical velocity cost
- angular velocity cost
- stuck penalty
- fall reward

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- Stock Humanoid-style reward: `healthy_reward + forward_reward - ctrl_cost`.
- `Jitu0110/RLMujoco` dodatno pojacava healthy/forward reward i smanjuje ctrl
  cost.
- `ritwikrohan/DDPG-SAC-HumanoidWalking` dodaje standing/forward velocity bonus.

**Local adaptation:**

- Nas zadatak nije samo "idi +x", nego joystick tracking. Zato je forward
  progress vezan za command tracking, a ne puko nagradjivanje brzine.

### 16. Positive reward clipping

**Files:** `biomechanics_env.py`

**Change:**

- Non-terminal reward se clipuje na `[0, 3]`.
- Fall ostaje negativan (`-10`).

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- `roboterax/humanoid-gym` ima `only_positive_rewards=True` i clip negativnog
  total reward-a pre termination reward-a.

### 17. Softer fall penalty

**Files:** `biomechanics_env.py`

**Change:**

- Fall penalty je smanjen sa ekstremno negativnog na `-10`.

**Source:** Local adaptation.

**Why:**

- Prevelika negativna kazna moze da dominira PPO signalom na pocetku i da napravi
  "ne diraj nista" strategiju.

### 18. Foot slip penalty

**Files:** `biomechanics_env.py`

**Change:**

- Dodata kazna za horizontalno klizanje foot sole geometrija dok su u kontaktu.

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- `roboterax/humanoid-gym` ima `foot_slip` reward/cost.

### 19. Foot contact / gait reward

**Files:** `biomechanics_env.py`

**Change:**

- Gait reward nagradjuje:
  - swing foot clearance
  - stance foot contact
- Contact je pseudo-contact iz visine `left_foot_sole/right_foot_sole`.

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- `roboterax/humanoid-gym` koristi gait phase, stance masks, contact rewards i
  foot clearance.
- `.tmp_rohan_lhw` walking task koristi foot force/foot velocity clock rewards.

### 20. Base height and orientation rewards

**Files:** `biomechanics_env.py`

**Change:**

- Dodata mala nagrada za root height blizu pocetne stojece visine.
- Upright signal ostaje deo reward-a.

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- Humanoid/legged locomotion env-ovi tipicno koriste base height/orientation
  stabilizatore.

### 21. Overspeed and stuck penalties

**Files:** `biomechanics_env.py`

**Change:**

- Kazni se prebrzo kretanje u odnosu na command.
- Kazni se stanje gde command trazi hod, a forward velocity ostaje prenizak.

**Source:** Local adaptation.

**Why:**

- Prethodni reward je mogao da nagradi pasivno ubrzanje/klizanje koje nije pravo
  pracenje joystick komande.

## PPO / Training Changes

### 22. Bigger PPO setup for biomechanics humanoid

**Files:** `train.py`

**Change:**

Default biomechanics PPO config:

- `num_timesteps=50_000_000`
- `num_envs=1024`
- `num_eval_envs=32`
- `episode_length=500`
- `learning_rate=3e-4`
- `entropy_cost=3e-3`
- `unroll_length=20`
- `batch_size=512`
- `num_minibatches=8`
- `num_updates_per_batch=4`
- policy/value MLP: `(512, 256, 128)`

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- `roboterax/humanoid-gym` trains with large parallel rollout style setup,
  privileged critic, frame/history buffers and many total environment steps.
- MuJoCo Playground humanoid-style configs are generally much larger than the
  earlier tiny custom config.

### 23. Asymmetric PPO critic

**Files:** `train.py`

**Change:**

- PPO network factory uses:
  - `policy_obs_key="state"`
  - `value_obs_key="privileged_state"`

**Source:** Seen in other projects / Adapted.

**Seen in other projects:**

- `roboterax/humanoid-gym` uses privileged critic observations.

### 24. Checkpoint controls

**Files:** `config.py`, `train.py`

**Change:**

- Added:
  - `--no-checkpoints`
  - `--checkpoint-out`
  - `--resume-from`
  - `save_checkpoints`
  - `checkpoint_out`
  - `resume_from`

**Source:** Local fix.

**Why:**

- Useful for debug runs and for avoiding slow Orbax writes on Windows paths.
- Allows continuing from a compatible v14 checkpoint without changing code.
- Important: old 12-action checkpoints are not compatible with the new 18-action
  abdomen/pelvis setup.

### 25. `walk` command profile support

**Files:** `train.py`, `evaluate.py`, `biomechanics_env.py`

**Change:**

- CLI now accepts `--command-profile walk`.
- `walk` uses forward-style command plus gait phase support.

**Source:** Adapted.

**Seen in other projects:**

- Rohan/Roboterax style walking tasks use explicit phase/clock structure.

### 26. Debug preflight supports dict observations

**Files:** `train.py`

**Change:**

- Debug preflight now prints shape tree:
  - `{'state': (98,), 'privileged_state': (151,)}`

**Source:** Local fix.

### 27. UTF-8 stdout for Windows traceback/logging

**Files:** `train.py`

**Change:**

- Added `configure_stdout_encoding()`.

**Source:** Local fix.

**Why:**

- Loguru/JAX traceback could fail on Windows legacy console encoding while trying
  to print the actual error.

## Evaluation Changes

### 28. Dict observation command update

**Files:** `evaluate.py`

**Change:**

- `set_command()` updates command in both:
  - `obs["state"][9:12]`
  - `obs["privileged_state"][9:12]`

**Source:** Local fix.

**Why:**

- Joystick evaluator must keep working after moving to dict observations.

### 29. Auto command profile detection

**Files:** `evaluate.py`

**Change:**

- Evaluator can infer `command_profile` from run config.
- Fallback handles dict observation metadata.

**Source:** Local fix.

### 30. Evaluator action smoothing arg

**Files:** `evaluate.py`

**Change:**

- Added `--action-smoothing` so viewer env matches training env.

**Source:** Local fix.

### 31. Optional MJDATA/QPOS initial pose

**Files:** `biomechanics_env.py`, `config.py`, `train.py`, `evaluate.py`,
`view_model_pose.py`

**Change:**

- Added `--init-qpos-file`.
- The env can now start from a MJDATA-style `QPOS` block, for example:
  `MJDATA_neutral_poze.TXT`.
- The loaded QPOS is sanitized before training:
  - checks that the file has exactly `nq` values,
  - normalizes the root quaternion,
  - clips limited joints to legal MuJoCo ranges,
  - adjusts root height so the lowest foot sole point starts with preload contact.
- Evaluator can auto-read `init_qpos_file` from a run config, unless overridden.
- Viewer script can show the same initial pose before training.

**Source:** Local fix / experiment option.

**Why:**

- Lets us test the neutral half-squat pose without hard-coding it as the default.
- The current default standing-home pose remains unchanged when the flag is not
  provided.

### 32. BVH reference gait support

**Files:** `bvh_reference.py`, `biomechanics_env.py`, `config.py`, `train.py`,
`evaluate.py`

**Change:**

- Added `--reference-gait bvh`.
- Added repeated `--reference-gait-file`.
- Added `--reference-gait-list` for text files with one BVH path per line.
- Env now loads multiple BVH clips and randomly selects one reference clip per
  episode.
- BVH joint angles are retargeted to the actuated hip/knee/ankle joints.

**Source:** Adapted mocap/reference gait idea.

**Why:**

- Pure velocity reward and sine gait were not enough to remove zombie/sliding
  behavior.
- BVH reference gives the policy a human walking pose prior.

**Limitation found:**

- The first tier1 BVH run reached about `1493` reward, but still slid heavily.
- Current BVH imitation tracks joint angles only; it does not yet force stance
  foot world positions, root trajectory, or foot contact timing strongly enough.

### 33. BVH walking tier lists

**Files:** `BVH_walking_animation/build_walk_tiers.py`,
`BVH_walking_animation/tier1_forward_walk.txt`,
`BVH_walking_animation/tier2_walk_variations.txt`,
`BVH_walking_animation/tier3_style_or_complex_walks.txt`,
`BVH_walking_animation/uneven_terrain_walks.txt`,
`BVH_walking_animation/walk_tiers_summary.md`

**Change:**

- Walking BVH files are split into curriculum tiers:
  - tier1: vanilla forward walk,
  - tier2: simpler walk variations,
  - tier3: complex/stylized walks,
  - uneven terrain/stairs separated.

**Source:** Local curriculum design.

**Why:**

- Training on all walking clips at once can mix too many styles and tasks.
- Tier1 gives the policy a cleaner first objective.
- Tier2 should be introduced only after stable visible walking.

### 34. Anti-slip v2 reward tuning

**Files:** `biomechanics_env.py`

**Change:**

- Foot slip cost is now contact-aware squared foot speed:

```text
P_slide = alpha * max(||v_foot_xy|| - free_speed, 0)^2, when foot is in contact
```

- Initial anti-slip scale was increased aggressively, but later BVH bootstrap
  testing showed that this saturated rewards before the policy could learn.
  Current bootstrap values are documented in
  "BVH Bootstrap Reward Saturation Fix".
- Added explicit swing clearance deficit cost.
- Swing clearance and stance contact rewards were increased.

**Source:** Local fix, matching the contact-aware foot sliding diagnosis.

**Why:**

- The BVH tier1 run showed high reward but also high `foot_slip` and saturated
  `swing_drag`.
- Linear slip cost was too weak and could be overpowered by velocity tracking,
  posture, and reference rewards.
- The BVH tier1 model visibly did not lift its feet, so no-lift behavior is now
  penalized directly instead of only through aggregate gait reward.
- The next objective is not a bigger reward number, but less visible sliding.

### 35. BVH target-conditioned observation and phase sync

**Files:** `biomechanics_env.py`, `config.py`, `train.py`, `evaluate.py`

**Change:**

- Added `reference_target_observation`.
- New BVH training runs expose the current retargeted BVH joint target to the
  policy observation.
- For BVH runs, policy `state` observation grows from `98` to `116`.
- Command slice remains `9:12`.
- BVH gait/no-lift phase now uses active BVH phase instead of an independent
  random sine clock.
- BVH swing side is derived from reference knee flexion.

**Source:** Local critical review fix.

**Why:**

- The previous BVH reward was partly hidden from the policy: each episode picked
  a random clip, but the policy did not observe the clip/frame target.
- That makes the task partly non-Markov and encourages a sliding average motion.
- The no-lift penalty was also using an independent procedural gait phase that
  could conflict with the BVH frame being rewarded.

**Compatibility:**

- Existing non-BVH/legacy eval uses the old observation shape unless the saved
  run config explicitly has `reference_target_observation=true`.
- New BVH training from `train.py` enables it automatically.

### 36. v16 anatomical foot contact experiment

**Files:** `biomechanics_model.py`

**Change:**

- `SCENE_XML_VERSION` bumped to `trainfast_v16`.
- Removed the extra box/padding sole geom added in v15.
- The original generated foot capsule geom is now renamed to:
  - `left_foot_sole`
  - `right_foot_sole`
- Those capsule geoms are used for contact and foot reward signals.

**Source:** Local experiment based on visual review.

**Why:**

- The box sole may have encouraged sliding/anchoring because it made the contact
  patch too artificial.
- This tests whether the original anatomical foot shape produces cleaner swing
  and stance behavior.

**Verification:**

- Generated XML: `generated_models/human_male_180cm_75kg_standard_trainfast_v16.xml`.
- `left_foot_sole/right_foot_sole` are capsule geoms, not boxes.

### 37. Fast BVH debug list

**Files:** `BVH_walking_animation/tier1_debug_10.txt`

**Change:**

- Added a 10-clip vanilla walking subset for faster iteration.
- All clips are shorter than 5 seconds.

**Why:**

- Full tier1 has 102 clips and 7 are longer than a 10s episode.
- The small list reduces compile/device memory pressure and makes visual
  debugging less noisy.

### 38. Slow forward balance curriculum

**Files:** `biomechanics_env.py`, `train.py`, `evaluate.py`

**Change:**

- Added `--command-profile forward_slow`.
- `forward_slow` samples forward velocity in `0.02-0.12 m/s`.
- `forward_slow` uses 25% zero-command episodes for standing balance.
- `forward_slow` downweights velocity/progress reward while keeping gait/contact
  rewards active.
- Base-height reward, low-height cost and fall penalty were increased for the
  balance-first phase.
- Added done-reason metrics:
  - `done_low_height`
  - `done_tipped`
  - `done_invalid`

**Why:**

- The target-conditioned v16 run started lifting feet but still fell early.
- That means the next bottleneck is balance during stepping, not no-lift.
- Slower target velocity should let the policy learn stance/swing balance before
  full forward speed.

## Verification Done

### Python compile

Command:

```powershell
python -m py_compile train.py biomechanics_env.py biomechanics_model.py config.py evaluate.py
```

Result: passed.

### Model/XML sanity

Observed on generated v14 XML:

- `nq=40`
- `nv=39`
- `nu=18`
- `neq=0`
- Actuators:
  - abdomen x/y/z
  - pelvis x/y/z
  - left/right hip, knee, ankle actuators

### Env reset/JIT step

Observed:

- `state`: `(98,)`
- `privileged_state`: `(151,)`
- eager step passed
- JIT step passed

### PPO integration smoke test

Command:

```powershell
python train.py --debug-run --device cpu --allow-cpu --bare --no-checkpoints --timesteps 1000 --num-envs 4 --num-evals 0 --episode-length 20 --batch-size 4
```

Result: passed to completion.

## Not Added Yet

These ideas were seen in other projects but are not fully implemented yet:

- Full frame stacking for policy history.
- Full critic frame stacking/history buffer.
- Terrain height scan observations.
- Full domain randomization curriculum.
- Full mocap imitation with root pose, joint velocities, foot world positions,
  foot contact timing, and/or behavioral cloning.

## Latest Fix - Reward/Contact Sanity

Motivation: v15/v16 BVH runs could receive high reward while still visibly
sliding. Unitree RL Mjlab and Unitree RL Gym both use explicit foot contact
signals for foot slip, gait/contact timing, and swing height; Gymnasium Humanoid
also exposes contact-force style observations/costs.

Applied changes:

- Per-step reward is now clipped to `[-5, 3]` instead of `[0, 3]`, so foot slip,
  drag, and low-height penalties can become real negative learning signals.
- Fall termination remains a larger `-25` penalty.
- Foot contact in MJX now uses MuJoCo contact geom pairs against the floor, not
  only `foot_z < threshold`.
- Capsule foot-floor placement now computes capsule lower Z from endpoints plus
  radius instead of treating every foot geom as a box.
- Privileged critic observation now includes Gymnasium Humanoid-style physical
  signals: scaled `cinert`, `cvel`, and `cfrc_ext`.
- BVH references now include finite-difference joint velocity targets.
- BVH reward now includes a small reference velocity tracking term, so the policy
  is not rewarded only for matching static leg poses.
- Eval logs now include `ref_vel` and `contact_force`.

Still not implemented:

- Full Unitree-style motion imitation with root/body pose, body velocity, and
  end-effector tracking.
- Behavioral cloning / motion prior.

## Recommended Next Runs

Short GPU smoke run:

```powershell
python train.py --device gpu --command-profile forward --timesteps 5000000 --num-envs 1024 --num-evals 5 --no-checkpoints
```

Main run:

```powershell
python train.py --device gpu --command-profile forward_slow --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 10000000 --num-envs 1024 --num-evals 5 --no-erfi --no-domain-randomization --no-checkpoints --run-tag bvh_target_obs_v16_balance_debug10
```

After the debug run visibly lifts feet, switch to full tier1:

```powershell
python train.py --device gpu --command-profile forward --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_forward_walk.txt --timesteps 60000000 --num-envs 1024 --num-evals 10 --no-erfi --no-domain-randomization --run-tag bvh_tier1_target_obs_v16
```

## Latest Fix - Anatomical Action Prior

Motivation: the V10 debug run still collapsed early. Reward improved, but
`episode_length` stayed low and `done_low` stayed high, so the next bottleneck
was not "more steps" but an over-free generated body.

Applied changes:

- New generated XML version: `trainfast_v17`.
- Added stricter leg joint limits in `biomechanics_model.py`.
- Reduced dangerous lateral/twist action freedom:
  - hip_y/hip_z are much smaller,
  - ankle_z is very small,
  - abdomen/pelvis remain stiff.
- Kept useful stride freedom:
  - hip_x can still swing,
  - knee_z can still flex,
  - ankle_y can still help foot clearance.
- Added Unitree-inspired variable posture prior:
  - strict while standing,
  - looser for stride joints while walking,
  - still strict for trunk and lateral/twist axes.
- Added `var_pose` to eval logging.

Reference source:

- Inspired by Unitree RL Mjlab / Unitree RL Gym design patterns: PD targets
  around a default pose, joint-specific action scale, posture regularization,
  and realistic joint limits.

Recommended first V11 run:

```powershell
python train.py --device gpu --command-profile forward_slow --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 20000000 --num-envs 1024 --num-evals 10 --no-erfi --no-domain-randomization --run-tag unitree_prior_v11_debug10
```

## Latest Fix - DeepMimic/LocoMuJoCo-Style BVH Root And Foot Signals

Motivation: after reviewing DRLoco, DeepMimic_mujoco, LocoMuJoCo, and GMR, the
biggest mismatch in our BVH setup was clear: our reward was still too close to
"match a few leg joint angles", while working mimic systems track root motion,
joint velocities, end-effectors/sites, COM/body targets, and often initialize
episodes from the reference trajectory.

Applied changes:

- BVH references now include approximate root height offsets from CMU `Hips.Y`.
- BVH references now include approximate forward velocity factors from CMU
  `Hips.Z`.
- The environment precomputes BVH qpos/qvel reference targets once at startup.
- The environment precomputes trunk-relative left/right foot position targets
  using MuJoCo forward kinematics.
- Added `ref_foot` reward term for matching reference foot positions relative to
  the trunk/root.
- Added `ref_root` reward term for matching reference root height and
  command-scaled forward velocity.
- Added optional `reference_phase_randomization` so BVH episodes can start from
  random reference frames.
- Added optional `reference_state_init` so reset can place the model directly in
  a BVH-derived qpos/qvel state.
- Fixed `reference_state_init` action smoothing by initializing `last_action`
  from the reset control target instead of zero.
- Added `reference_sanity.py` to quickly inspect BVH target ranges before a long
  PPO run.
- Eval now infers `reference_phase_randomization` and `reference_state_init`
  from a run config when loading a checkpoint.

Reference source:

- DRLoco: reference trajectory stepping, qpos/qvel imitation, COM-style reward,
  explicit foot contact checks, and early termination.
- DeepMimic_mujoco: pose + velocity + end-effector + root + COM reward structure.
- LocoMuJoCo: trajectory handlers and mimic reward over qpos, qvel, relative site
  positions/orientations/velocities.
- GMR: full BVH-to-robot retargeting direction with `root_pos`, `root_rot`,
  `dof_pos`, and body/link target output.

Local limitation:

- This is not full GMR retargeting yet.
- Root height and forward speed are approximated from BVH hips channels.
- Foot targets come from our retargeted qpos through MuJoCo FK, not from a full
  IK solve over the generated humanoid.
- `reference_state_init` is implemented but should be treated as an experimental
  second gate, not enabled together with every other change on the first run.

Verification:

```powershell
python -m py_compile biomechanics_env.py train.py evaluate.py config.py bvh_reference.py reference_sanity.py
```

Result: passed.

```powershell
python reference_sanity.py --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --xml-path generated_models/human_male_180cm_75kg_standard_trainfast_v16.xml
```

Observed:

- `clips=10`
- `max_frames=598`
- `root_height_offset span=0.2120`
- `root_velocity_factor mean=1.0000`
- left/right foot local target spans are non-zero and look physically plausible.

## Current Run Status - 2026-06-21

The active V10 forward resume run is not dead. It improved from early collapse to
`reward=960.2457` at step `53,739,520`, with `done_low=0.312` and
`episode_length=457.5625`.

Latest confirmed checkpoint:

```text
runs/biomechanics_noerfi_nodr_forward_ref_bvh_v10_18350_resume_forward_accurate_ppo_BiomechanicsHumanJoystickStandard_20260621_0254_80m_seed7_running/checkpoints/000053739520
```

Recommended next gate after this run finishes:

```powershell
python train.py --device gpu --command-profile forward --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --resume-from runs/biomechanics_noerfi_nodr_forward_ref_bvh_v10_18350_resume_forward_accurate_ppo_BiomechanicsHumanJoystickStandard_20260621_0254_80m_seed7_running/checkpoints/000053739520 --xml-path generated_models/human_male_180cm_75kg_standard_trainfast_v16.xml --legacy-action-prior --timesteps 8000000 --num-envs 1024 --num-evals 4 --num-eval-envs 16 --no-erfi --no-domain-randomization --run-tag v10_ref_root_foot_gate
```

Do not enable `--reference-state-init` on this first gate. If the root/foot gate
improves visible walking or at least does not regress, then run a separate short
gate with:

```powershell
--reference-phase-randomization --reference-state-init
```

## Reward Review - False 1000 Reward / No Visible Step

Motivation: the active V10 forward resume reached about `1000` reward, but visual
inspection in MuJoCo showed that it still could not produce a real step. That
means the reward was still exploitable: it rewarded velocity/upright/survival and
some gait metrics without requiring a real swing/stance foot cycle.

Finding:

- `gait_reward` could be high even when the swing foot was also touching the
  ground.
- `command_progress` and `tracking_lin_vel` could stay high during sliding.
- The log did not expose enough gait validity diagnostics to catch this quickly.

Applied changes:

- `gait_reward` now gives stance-contact bonus only for single-support:
  stance foot in contact and swing foot not in contact.
- Added `locomotion_quality`, based on single-support plus swing clearance.
- Velocity tracking reward is now gated by `locomotion_quality`, but keeps a
  small floor so early learning is not totally starved.
- Forward progress reward is fully gated by `locomotion_quality`.
- Added `double_support_drag` cost for moving along the command while both feet
  are in contact.
- Increased swing-foot drag and clearance-deficit penalties.
- Train logs now include:
  - `loco_q`
  - `gated_track`
  - `gated_prog`
  - `swing_ct`
  - `stance_ct`
  - `double_ct`
  - `dbl_drag`

Interpretation for future runs:

- A high reward is not trusted unless `loco_q` rises and `double_ct/swing_ct`
  do not stay saturated.
- If `tracking` is high but `gated_track/gated_prog` are low, the model is
  moving/sliding but not walking.
- If `gait` is high while `loco_q` is low, the gait reward is still too weak or
  the contact signal is wrong.

## Reward Rework - Separate Sine Zombie Walking From BVH Mimic

Motivation: after comparing our reward with DRLoco, DeepMimic_mujoco,
LocoMuJoCo, and Unitree tracking, the biggest conceptual mistake was that BVH
imitation was still mixed into the same velocity/upright/progress reward used
for procedural walking. Working mimic systems treat reference motion as the main
task, not as a small style bonus.

Applied changes:

- Split reward composition into three explicit sections:
  - `task_reward`: normal joystick velocity tracking without explicit imitation.
  - `zombie_sine_reward`: procedural sine / zombie-walking trajectory reward.
  - `bvh_mimic_reward`: BVH imitation-first reward.
- For `reference_gait == "bvh"`, the selected reward is now `bvh_mimic_reward`.
- For `reference_gait == "sine"`, the selected reward is now
  `zombie_sine_reward`.
- For `reference_gait == "none"`, the selected reward is `task_reward`.
- BVH mimic reward now follows the DeepMimic/LocoMuJoCo shape:
  - pose reference,
  - velocity reference,
  - foot/end-effector relative position reference,
  - root height/velocity reference,
  - contact/locomotion quality.
- Joystick velocity tracking remains in BVH mode, but only as a small use-case
  term. It is no longer allowed to dominate motion imitation.
- BVH reference pose/velocity/foot/root rewards are no longer disabled when the
  command is near zero; imitation remains active like in DeepMimic-style setups.
- Train logs now expose `task_rew`, `zombie_rew`, and `mimic_rew`.

Remaining difference from full reference repos:

- Still no full GMR-retargeted body dataset.
- Still no full body position/orientation/velocity reward for every relevant
  body.
- Still no RootPoseTrajTerminalStateHandler equivalent.
- `reference_state_init` exists and should be used for the next short BVH mimic
  probe, because DRLoco/LocoMuJoCo both rely on trajectory/reference-state init.

## Unitree-Style XML / Contact Guardrails

Motivation: Unitree-style humanoid tasks do not lock the feet to the floor.
They constrain the behavior space with joint limits, small PD target action
scales, soft joint-limit penalties, foot slip/contact metrics, and illegal
contact handling for bodies that should not touch the ground.

Applied changes:

- Bumped generated model version to `trainfast_v18`, leaving `v17` available for
  comparison.
- Kept the stricter `v17` leg ranges and Unitree-style per-joint action scales.
- Added a soft joint-limit penalty for the outer 10% of each controlled joint
  range, matching the `soft_dof_pos_limit = 0.9` pattern.
- Split contact collision groups:
  - terrain: contact type `1`, no affinity;
  - feet: contact type `2`, ground affinity;
  - illegal lower-body geoms: contact type `4`, ground affinity.
- Added named illegal-contact geoms on pelvis, thighs, and shanks.
- Added hip/thigh/shank floor-contact penalty.
- Added pelvis floor-contact termination.
- Added train/eval metrics:
  - `joint_limit`
  - `illegal_ct`
  - `done_illegal`

Expected effect:

- A high reward with folded legs, pelvis scraping, or knee/hip ground contact
  should now fail visibly in metrics.
- The real BVH mimic gate should use the default `trainfast_v18` XML and avoid
  `--legacy-action-prior`.

## BVH Bootstrap Reward Saturation Fix

Observation from `v18_unitree_bvh_mimic_gate`: the run stayed at short episodes
and did not learn useful movement. The raw `bvh_mimic_reward` was often around
`-1300` to `-2600` per eval episode, while the actual per-step reward was clipped
to the lower bound. This makes many bad actions look equally bad to PPO.

Applied changes:

- Reduced bootstrap anti-slide/contact penalties to Unitree-like magnitudes:
  - `foot_slip`: `1.0 -> 0.25`
  - `swing_drag`: `3.0 -> 0.5`
  - `clearance_deficit`: `2.0 -> 0.5`
  - `double_support_drag`: `1.5 -> 0.15`
- Reduced soft joint-limit and illegal-contact costs during bootstrap.
- Reduced low-height and fall penalties so early failed policies are still
  distinguishable instead of all saturating to the same clipped reward.
- Made BVH regularization bounded with `tanh` for large-cost terms, while
  keeping mimic/root/foot/stability rewards positive and readable.

Run guidance:

- Do not use `--reference-state-init` for the next partial-BVH gate. Current BVH
  targets are not full root/body retargeted states, so RSI starts from
  physically inconsistent poses.
- Re-enable RSI only after the standing-reset BVH gate survives and visibly
  steps, or after GMR/full-body retargeting is integrated.
