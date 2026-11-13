# MuJoCo humanoid locomotion

Na grani `navigation_walking_with_anatomical_human_robot` razvijamo fazu 2:
prirodan A→B, zaustavljanje, prelaze i push recovery. Sačuvani hod i zaseban
60 s stand postoje; -X recovery, walk→stand i A→B još nisu završeni.
[Ciljevi](docs/PROJECT_GOALS.md) i [poređenje pristupa](docs/FRAMEWORK_COMPARISON_2026-11-12.md)
određuju naredni razvoj. Zamrznuti hod označen je tagom `natural-walk-v1.0`.

Prvi [MimicKit test kompatibilnosti](docs/MIMICKIT_PREFLIGHT.md) je izvršen:
adapter anatomije i prenos walking reference prolaze proveru kinematike.
Native MuJoCo control-to-torque provera takođe prolazi. Slede iste provere
u novom simulatoru i mali komandovani task pre većeg treninga.

Treniramo humanoida u MuJoCo/MJX simulatoru pomoću PPO-a i BVH motion
reference. Projekat ima dve zaštićene linije rada:

- `natural_walking_with_anatomical_human_robot`: finalni DeepMimic hod;
- `joystick_walking_with_anatomical_human_robot`: potvrđeni joystick sistem.

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
| `scripts/evaluate_final_success.py` | Finalni zamrznuti prirodni hod. |
| `prezentacija/` | Zamrznuti uspesi, checkpoint-i i demonstracije. |
| `runs/` | Lokalni treninzi i checkpoint-i. |
| `videos/` | Lokalni snimci. |

Detaljnija mapa je u [docs/REPOSITORY.md](docs/REPOSITORY.md), a komande u
[docs/WORKFLOW.md](docs/WORKFLOW.md).

## Brza evaluacija

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
- Ne mešamo natural-walk i joystick razvoj bez eksplicitnog spajanja.
- Ne proglašavamo uspeh samo na osnovu reward-a: tražimo stabilan rollout,
  fizičke metrike i video.
- `runs/`, `prezentacija/`, `videos/`, BVH/SMPL podatke i aktivne checkpoint-e
  ne brišemo tokom čišćenja.
