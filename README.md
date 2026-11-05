# MuJoCo humanoid locomotion

Treniramo humanoida u MuJoCo/MJX simulatoru pomoću PPO-a i BVH motion
reference. Projekat trenutno ima dve zaštićene linije rada:

- `main`: potvrđeni osnovni hod i postojeći joystick sistem;
- `experiment/arms-ik-v2`: prirodniji DeepMimic hod sa aktivnim rukama.

Završni DeepMimic rezultat prati `02_01.bvh` sa punom fleksijom lakta. Do njega
smo došli curriculum-om 25%, 50%, 75% i 100%, uz automatsku proveru stabilnog
hoda i stvarnog opsega oba lakta posle svakog nivoa.

## Najvažniji ulazi

| Putanja | Namena |
| --- | --- |
| `train.py` | Glavni PPO trening. |
| `evaluate.py` | Evaluacija, viewer i snimanje videa. |
| `biomechanics_env.py` | Fizika, observations, actions, rewards i termination. |
| `bvh_reference.py` | BVH učitavanje i retargeting. |
| `scripts/train_elbow_flexion_curriculum.py` | Trenutni gated elbow curriculum. |
| `scripts/evaluate_final_success.py` | Finalni zamrznuti prirodni hod. |
| `prezentacija/` | Zamrznuti uspesi, checkpoint-i i demonstracije. |
| `runs/` | Lokalni treninzi i checkpoint-i. |
| `videos/` | Lokalni snimci. |

Detaljnija mapa je u [docs/REPOSITORY.md](docs/REPOSITORY.md), a komande u
[docs/WORKFLOW.md](docs/WORKFLOW.md).

## Brza evaluacija

Prvi sačuvani uspeh na Windowsu:

```powershell
./.venv-win/Scripts/python.exe scripts/evaluate_first_success.py
```

Drugi uspeh sa aktivnim, ali krutim rukama:

```powershell
./.venv-win/Scripts/python.exe scripts/evaluate_second_success.py
```

Finalni prirodni hod sa pokretnim laktovima:

```powershell
./.venv-win/Scripts/python.exe scripts/evaluate_final_success.py
```

Provera trenutnog koda:

```powershell
./.venv-win/Scripts/python.exe -m unittest discover -s tests -v
```

## Pravila projekta

- Ne koristimo kod iz odbačenog `bipedal-humanoid-bullet-avoidance` projekta.
- Ne menjamo `main` dok ne zamrznemo završni DeepMimic rezultat.
- Ne proglašavamo uspeh samo na osnovu reward-a: tražimo stabilan rollout,
  fizičke metrike i video.
- `runs/`, `prezentacija/`, `videos/`, BVH/SMPL podatke i aktivne checkpoint-e
  ne brišemo tokom čišćenja.
