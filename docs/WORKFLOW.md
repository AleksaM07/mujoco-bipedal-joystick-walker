# Kako mi pokrećemo projekat

Mi sve naredne komande izvršavamo iz korena repository-ja. U Git Bash-u koristimo
jednolinijske komande ili `\` za nastavak reda; PowerShell backtick ne prenosimo
u Bash. Našu uspešnu politiku gledamo na Windows CPU-u, a GPU trening pokrećemo
u Linux/WSL okruženju.

## Naš zamrznut uspešan model

Mi koristimo wrapper koji eksplicitno bira arhivirani checkpoint, XML i BVH:

```bash
./.venv-win/Scripts/python.exe scripts/evaluate_first_success.py
```

Mi pokrećemo proveru bez prozora ili snimanje novog videa sledećim komandama:

```bash
./.venv-win/Scripts/python.exe scripts/evaluate_first_success.py --inspect --inspect-steps 500 --seed 11
./.venv-win/Scripts/python.exe scripts/evaluate_first_success.py --seed 55 --record-video videos/first_success_seed55.mp4 --record-steps 500
```

Mi prvi JAX compile možemo čekati nekoliko minuta. Uspeh procenjujemo po
rollout-u, logu i videu. Pri promeni seeda menjamo početnu referentnu fazu;
sa podrazumevanim determinističkim policy-jem ne uzorkujemo dodatni action šum.

Mi proveravamo arhivski sadržaj bez JAX-a i bez simulacije:

```bash
./.venv-win/Scripts/python.exe tools/verify_success_archive.py --require-tracked
```

Mi originalni `evaluate.py` i dalje koristimo za bilo koji drugi checkpoint:

```bash
./.venv-win/Scripts/python.exe evaluate.py --help
```

## Naš sledeći trening iz istog baseline-a

Mi trenutno imamo proverene launchere za pure tracker. U Windows PowerShell-u
možemo prvo samo ispisati komandu, bez pokretanja treninga:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_recovery_v6_local.ps1 -NumEnvs 128 -Timesteps 30000000 -ResumeFrom prezentacija/1st_success/final_model_000025681920 -PrintCommand
```

Mi uklanjanjem `-PrintCommand` pokrećemo pozadinski WSL trening. Podrazumevani
launcher nije vremenski ograničen; `-Probe` namerno pokreće ograničen benchmark.
Broj `-Timesteps` označava nove korake. Resume prenosi policy i observation
normalizer; optimizer i lokalni step counter kreću ponovo.

Mi sa čistog Linux-a ili iz WSL Bash-a isti preset možemo pokrenuti ovako:

```bash
RECOVERY_NUM_ENVS=128 RECOVERY_TIMESTEPS=30000000 RECOVERY_RESUME_FROM=prezentacija/1st_success/final_model_000025681920 bash scripts/train_recovery_v6_tracker.sh
```

Mi ovim komandama nastavljamo **isti pure tracker**. Joystick i ruke još nemaju
odobren novi trening preset; njihov redosled definišemo u [ROADMAP.md](ROADMAP.md).
Trening log i lokaciju novog run-a dobijamo iz launchera u `artifacts/`.

## Naša Python okruženja

Mi čuvamo `pyproject.toml` i `uv.lock` bez promene trening zavisnosti u ovom
organizacijskom prolazu. Na Linux-u naš projekat koristi CUDA JAX, MuJoCo 3.10.0
i postojeći lockfile. Na kompatibilnom Linux okruženju instaliramo ga pomoću:

```bash
python -m uv sync --frozen
.venv/bin/python -c "import jax; print(jax.devices())"
```

Mi na ovom Windows računaru već imamo radno `.venv-win` CPU okruženje. Ne
pokrećemo običan `uv run` u zajedničkom Windows/WSL checkout-u: pyproject još
zahteva Linux CUDA wheel-ove, a `.venv` pripada Linux-u. Prenosivu instalaciju
za potpuno nov Windows računar tretiramo kao zasebnu dependency popravku.

## Naše regresione provere

Mi koristimo postojeće unittest klase bez novih paketa:

```bash
./.venv-win/Scripts/python.exe -m unittest discover -s tests -v
```

Mi za uobičajeni rad ne pokrećemo sve dijagnostičke alate. Za kontakt/IK problem
biramo odgovarajući alat iz [indeksa](../tools/README.md), a za prezentaciju
koristimo zamrznuti model i gornji viewer.
