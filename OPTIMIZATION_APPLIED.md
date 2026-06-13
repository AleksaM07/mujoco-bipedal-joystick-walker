# Optimization Applied

Kratak pregled optimizacija i bitnih odluka za biomehanicki humanoid env.

## Model / XML

- XML se generise jednom i kesira u `generated_models/`.
- Aktivni training XML je `trainfast_v8`.
- `compiler angle` je `degree`, jer `mujoco-biomechanics` generator daje uglove u stepenima.
- Stari `trainfast_v3` je bio pogresan: stepene je tretirao kao radijane.
- Actuator `ctrlrange` se pravi iz anatomskog joint range-a.
- Vise ne koristimo globalni `+-3.14` za svaki actuator.
- Pasivni zglobovi imaju stiffness/springref, da model ne bude ragdoll.
- Kontrolisani zglobovi nogu imaju armature i frictionloss slicno Berkeley principu.
- Abdomen/pelvis u `trainfast_v6`: stiffness `350`, damping `8`, frictionloss `2`, armature `0.02`.
- Ramena/ruke/glava imaju slabiju pasivnu cvrstinu da ne vise potpuno mlitavo.
- `trainfast_v7` je dodao box-shaped sole contact za stopala.
- `trainfast_v8` zakljucava `abdomen_x/y/z` i `pelvis_x/y/z` preko MuJoCo equality constraints.
- Ovo pravi Berkeley-like rigid torso baseline umesto fleksibilnog biomehanickog spine/pelvis lanca.

## Controlled Joints

- `nq=40` i `nv=39`: full humanoid state postoji.
- `nu=12`: policy direktno kontrolise samo noge.
- Kontrolisani su:
  - `left_hip_x/y/z`
  - `left_knee_z`
  - `left_ankle_y/z`
  - `right_hip_x/y/z`
  - `right_knee_z`
  - `right_ankle_y/z`
- Abdomen/pelvis su trenutno zakljucani equality constraints-ima u `trainfast_v8`.
- Glava, ruke i sake ostaju pasivni.
- Ovo prati Berkeley princip: noge se kontrolisu, trup se ne lomi kroz action space.

## Physics

- `ctrl_dt = 0.02`, `sim_dt = 0.005`: 4 physics substep-a po policy koraku.
- Za trening kolidiraju samo teren i stopala.
- Trup, glava i ruke imaju masu/inerciju, ali ne ulaze u contact solver.
- Ovo smanjuje broj kontakt parova i ubrzava MJX step.

## Domain Randomization

- Ne generise se novi XML tokom treninga.
- Randomizacija menja MJX model arrays: velicina, mase, inercije i trenje.
- `site_pos` nije randomizovan jer ga trenutni observation/reward ne koristi.

## RFI / RAO / ERFI-50

- RFI je random torque injection u `qfrc_applied`.
- RAO je episodic torque offset u `qfrc_applied`.
- Trenutni limit je `2 Nm` za RFI i `2 Nm` za RAO.
- Perturbacije se primenjuju samo na 12 kontrolisanih nogu.
- ERFI-50 je implementiran po epizodi:
  - 50% epizoda koristi RFI.
  - 50% epizoda koristi RAO.
- ERFI je ukljucen za training env, ali iskljucen za eval env.

## Observation / Reward

- Observation koristi lokalne brzine tela, ne world-frame `qvel`.
- Ubacen je `projected_gravity`, da policy vidi nagib tela.
- Komanda u biomechanics observation-u je na indeksima `9:12`.
- Kod ovog modela anatomska vertikala toraksa je lokalna `Y` osa.
- Reward prati:
  - forward brzinu: lokalni `X`
  - lateral brzinu: lokalni `Z`
  - yaw brzinu: rotacija oko lokalne `Y`
- Terminal reward je `-100` kada human padne.
- Root height termination je podignut na `0.65 * init_height`, jer inspect pokazuje da policy kolabira nisko dok torso jos deluje uspravno.
- Height penalty pocinje vec ispod `0.8 * init_height`, da reward ne nagradjuje kratko spustanje pre pada.

## PPO / Training

- Default `num_envs = 512` radi boljeg GPU throughput-a.
- PPO block je `10240` env stepova:
  `batch_size=256 * unroll_length=10 * num_minibatches=4`.
- Progress log ide na 5%, 10%, 15%...
- Run folder pamti final reward i best reward za nove run-ove.
- `--bare` mode postoji za baseline:
  - ERFI iskljucen
  - domain randomization iskljucen
  - run folder dobija `biomechanics_bare_...`

## Berkeley vs Biomechanics

Sta Berkeley humanoid radi, a nas biomech env jos nema u istoj meri:

- Berkeley ima rucno pripremljen `home` pose za hod, ne anatomsku A-pose.
- Berkeley kontrolise samo 12 zglobova nogu; sada i mi radimo isto.
- Berkeley ima actuator opsege uskladjene sa joint range-ovima; sada i mi radimo isto.
- Berkeley koristi `sim_dt = 0.002`, odnosno oko 10 substepova po policy koraku.
- Nas biomechanics env sada podrazumevano koristi `sim_dt = 0.005`, odnosno 4 substep-a po policy koraku.
- Berkeley observation ima phase/gait signal (`sin/cos phase`), a nas jos nema.
- Berkeley reward ima vise komponenti:
  - tracking linear/yaw velocity
  - orientation cost
  - pose cost
  - action-rate cost
  - torque cost
  - feet air-time / feet phase reward
  - feet slip cost
- Nas reward je jos kraci i vise baseline/debug nego finalni locomotion reward.
- Berkeley ima noise u observation-u i push perturbacije, ali tek preko stabilnog env-a.
- Za nas sledeci ispravan korak je prvo `--bare`, pa tek onda vracanje ERFI/domain randomization.

## Compatibility

- Stari checkpointovi nisu kompatibilni posle prelaska na `trainfast_v4`.
- Razlog: `nu` je promenjen sa 18 na 12, a observation sa 96 na 90.
