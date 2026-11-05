# Pokretanje i praćenje

Sve komande pokrećemo iz korena repository-ja.

## Windows evaluacija

```powershell
./.venv-win/Scripts/python.exe scripts/evaluate_first_success.py
./.venv-win/Scripts/python.exe scripts/evaluate_second_success.py
./.venv-win/Scripts/python.exe evaluate.py --help
```

Prvi JAX compile može trajati nekoliko minuta. Ne proglašavamo zastoj dok proces
aktivno koristi CPU ili GPU i nema greške u logu.

## Aktivni elbow curriculum

Campaign:

```text
artifacts/elbow_curriculum_20261105_4m
```

Status nivoa:

```powershell
Get-Content "artifacts\elbow_curriculum_20261105_4m\status.json"
```

Supervisor i audit log:

```powershell
Get-Content "logs\elbow_curriculum_20261105_4m.log" -Wait -Tail 50
```

Najnoviji detaljni PPO log:

```powershell
$run = Get-ChildItem runs -Directory |
  Where-Object Name -Like "*elbow_curriculum*" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Get-Content "$($run.FullName)\train.log" -Wait -Tail 50
```

Campaign automatski trenira `alpha=0.25, 0.50, 0.75, 1.00`. Posle svakog nivoa
traži četiri 500-step rollout-a bez fizičkog pada, dovoljan pose/arm score i
stvarno kretanje oba lakta. Neuspeli gate zaustavlja campaign.

## Ručni novi curriculum

Na Linux/WSL GPU-u:

```bash
PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -u scripts/train_elbow_flexion_curriculum.py \
  --steps-per-level 1000000 \
  --num-envs 64 \
  --learning-rate 0.00002
```

Ne pokrećemo drugi campaign dok prvi proces radi.

## Testovi

```powershell
./.venv-win/Scripts/python.exe -m unittest discover -s tests -v
```

Za brzu sintaksnu proveru promenjenih modula koristimo `python -m py_compile`.
