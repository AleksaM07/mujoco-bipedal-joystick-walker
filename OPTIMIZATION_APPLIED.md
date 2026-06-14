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
- True MuJoCo contact-force based foot contact reward in MJX.
- Terrain height scan observations.
- Full domain randomization curriculum.
- Reference motion imitation from mocap.

## Recommended Next Runs

Short GPU smoke run:

```powershell
python train.py --device gpu --command-profile forward --timesteps 5000000 --num-envs 1024 --num-evals 5 --no-checkpoints
```

Main run:

```powershell
python train.py --device gpu --command-profile forward --timesteps 50000000 --num-envs 1024 --num-evals 10
```

After forward walking starts to look stable, switch to full joystick:

```powershell
python train.py --device gpu --command-profile standard --timesteps 50000000 --num-envs 1024 --num-evals 10
```
