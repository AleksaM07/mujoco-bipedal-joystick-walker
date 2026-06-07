# Optimization Applied

Kratak pregled optimizacija koje su stvarno bitne za biomehanicki humanoid env.

## Model / XML

- XML se generise jednom i kesira u `generated_models/`.
- Generator iz `mujoco-biomechanics` se importuje jednom po procesu.
- Training XML ima verziju u imenu: `trainfast_v3`, da ne koristi stari kes.
- Aktuatori su `position` servo aktuatori, ne direktni torque motori.
- Kontrolise se 18 locomotion zglobova, ne svi zglobovi tela.

## Controlled Joints

- `nq=40` i `nv=39`: full humanoid state i dalje postoji.
- `nu=18`: policy direktno kontrolise 18 aktuatora.
- Nismo izbacili bitne zglobove nogu.
- Kontrolisani su:
  - `abdomen_x/y/z`
  - `pelvis_x/y/z`
  - `left_hip_x/y/z`
  - `left_knee_z`
  - `left_ankle_y/z`
  - `right_hip_x/y/z`
  - `right_knee_z`
  - `right_ankle_y/z`
- Svaka noga ima 6 direktno kontrolisanih DoF-ova:
  - kuk: 3 DoF
  - koleno: 1 DoF
  - ankle/skočni zglob: 2 DoF
- Ukupno: 12 DoF za noge + 6 DoF za abdomen/pelvis.
- Glava, ramena, laktovi, rucni zglobovi i sake nisu direktno kontrolisani.
- Ti zglobovi nisu obrisani iz modela; ostaju pasivni sa dampingom.
- Ovo je locomotion-focused izbor: policy kontroliše noge i trup, bez eksplozije action space-a.

## Fizika

- `ctrl_dt = 0.02`, `sim_dt = 0.01`, znaci 2 physics substep-a po policy koraku.
- Ranije je bilo 4 substep-a, pa je svaki env step radio duplo vise integracija.
- Za trening kolidiraju samo teren i stopala.
- Trup, glava i ruke ostaju u modelu sa masom/inercijom, ali ne ulaze u contact solver.
- Ovo prati isti princip kao Berkeley prototip: manje kontakt parova, brzi MJX step.

## Domain Randomization

- Ne generise se novi XML tokom treninga.
- Randomizacija menja MJX model arrays: velicina, mase, inercije i trenje.
- `site_pos` je izbacen iz randomizacije jer ga trenutni observation/reward ne koristi.

## RFI / RAO / ERFI-50

- RFI je implementiran kao random torque injection u `qfrc_applied`.
- RAO je implementiran kao episodic torque offset u `qfrc_applied`.
- Trenutni limit je `2 Nm` za RFI i `2 Nm` za RAO.
- Perturbacije se primenjuju samo na 18 kontrolisanih locomotion DoF-ova.
- ERFI-50 je implementiran po epizodi:
  - 50% epizoda koristi RFI: novi random torque svaki physics step.
  - 50% epizoda koristi RAO: konstantan torque offset tokom epizode.
- Policy i dalje daje ciljne pozicije za position servo aktuatore.
- RFI/RAO se dodaju kao dodatni joint torque, a ne kao noise na action.
- ERFI je ukljucen za training env, ali iskljucen za eval env.
- Eval reward zato meri cistu politiku, bez random torque perturbacija.

## PPO / Training

- Biomechanics default je pomeren na `num_envs = 512` radi boljeg GPU throughput-a.
- Ako se pusti manje od 512 env-ova, trening log ispisuje upozorenje.
- PPO block je smanjen na `10240` env stepova:
  `batch_size=256 * unroll_length=10 * num_minibatches=4`.
- Progress log sada ispisuje `elapsed`, vreme od poslednjeg loga i ETA.

## Stabilnost

- Reward, observation i action imaju NaN guard.
- PPO koristi manji learning rate `1e-4`.
- PPO koristi `max_grad_norm = 1.0`.

## Sta jos moze ako je i dalje sporo

- Probati `--num-envs 1024` ako GPU memorija dozvoli.
- Probati `warp` backend tek kad lokalne verzije `warp`/`mujoco-mjx` budu kompatibilne.
- Ako kvalitet treninga padne zbog `sim_dt=0.01`, vratiti na `0.005`, ali to ce biti sporije.
