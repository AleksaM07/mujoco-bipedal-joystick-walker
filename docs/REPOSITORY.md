# Mapa repository-ja

## Produkcioni runtime

| Putanja | Odgovornost |
| --- | --- |
| `train.py` | PPO konfiguracija, trening, checkpoint-i i logging. |
| `evaluate.py` | Učitavanje checkpoint-a, evaluacija, viewer i video. |
| `config.py` | `EnvConfig`, `TrainConfig` i podrazumevane vrednosti. |
| `biomechanics_env.py` | MuJoCo okruženje i RL ugovor. |
| `biomechanics_model.py` | Izgradnja humanoid modela. |
| `bvh_reference.py` | BVH parser, segmenti i host retargeting. |
| `reference_*.py` | IK, interpolacija, root/stance i kinematika. |
| `smpl_reference.py` | Postojeća SMPL podrška. |
| `phase1_backends.py` | JAX/Warp backend pomoćne funkcije. |

Ove module ne premeštamo dok traje curriculum. Refactor paketa bi promenio mnogo
import putanja bez poboljšanja ponašanja politike.

## Ulazi i provere

- `scripts/` sadrži samo korisničke entrypoint-e i benchmark skripte.
- `tools/` sadrži dijagnostiku koja štiti već pronađene reference/physics greške.
- `tests/` sadrži regresione testove za runtime i alate.

Jednokratne shell/PowerShell launchere iz završenih recovery, arms i joystick
eksperimenata uklonili smo. Njihove tačne verzije ostaju u Git istoriji.

## Podaci koje čuvamo

| Putanja | Razlog |
| --- | --- |
| `prezentacija/` | Uspešni modeli, manifesti i prezentacioni video. |
| `runs/` | Treninzi i checkpoint-i, uključujući aktivni curriculum. |
| `videos/` | Vizuelni dokazi i poređenja. |
| `BVH_walking_animation/` | Glavni BVH skup i trenutni `02_01.bvh`. |
| `CMU_SMPL+H-G/` | SMPL motion podaci za kasnije eksperimente. |
| `marina_bvh_parse/` | Još ga referencira kompatibilni one-clip preset. |
| `generated_models/` | XML verzije koje koriste sačuvane konfiguracije. |

Stare XML i motion putanje ne brišemo zato što ih postojeći run konfiguracioni
fajlovi i checkpoint evaluacija mogu zahtevati.

## Lokalni resursi

- `.venv` je aktivno WSL/GPU okruženje.
- `.venv-win` je Windows evaluaciono okruženje.
- `artifacts/repo_cleanup_20261102/branches.bundle` je privremeni recovery backup.
- `artifacts/workspace_archive_20261102/master_rad` sadrži materijale za master
  rad i ne sme se brisati kao običan cache.

Cache direktorijumi i privremeni klonovi nisu deo Git-a. Mogu se ponovo kreirati.

Trenutni kanonski DeepMimic rezultat je
`prezentacija/5th_success_final_natural_walk`; launcher
`scripts/evaluate_final_success.py` koristi samo njegove zamrznute asset-e.
