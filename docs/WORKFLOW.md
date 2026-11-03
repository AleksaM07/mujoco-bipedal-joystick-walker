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

## Naš prvi arms + joystick curriculum

Na grani `experiment/arms-joystick-v1` širimo provereni 18-DOF checkpoint na
26 akcija. Migracija zadržava stare lokomocione težine i normalizator, dok osam
novih shoulder/elbow izlaza počinje kao nulti residual nad BVH referencom. Vrat
za sada ostaje pasivno stabilizovan; nije deo 26 aktuatora.

Prvo samo ispisujemo komandu:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_arms_joystick_v1.ps1 -NumEnvs 64 -Timesteps 30000000 -PrintCommand
```

Bez `-PrintCommand` pokrećemo pozadinski 30M run bez timeout-a. Preset koristi
`standard_easy` komande u svim smerovima, mixed task+DeepMimic reward, aktivne
ruke, isti `02_01.bvh` ciklus i manji learning rate `5e-5` radi zaštite naučenog
hoda. PPO zadržava uspešne vrednosti `unroll=10`, `minibatches=4` i
`updates_per_batch=2`; ne preuzima novije globalne default-e.

## Izolacioni arms tracker bez joysticka

Pošto prvi zajednički arms+joystick run nije naučio stabilan hod, odvajamo dve
promene. `scripts/start_arms_tracker_v1.ps1` koristi isti 26-DOF warm-start i
aktivne shoulder/elbow aktuatore, ali `deepmimic_reward_mode=pure` prisiljava
joystick komandu na nulu i uklanja task reward. Tako merimo samo da li politika
može da zadrži hod dok uči BVH pokret ruku.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_arms_tracker_v1.ps1 -NumEnvs 64 -Timesteps 30000000
```

Ovo je 26 kontrolisanih DOF-ova. Stabilni model bez arm aktuatora ima 18 DOF-ova;
vrat nije policy akcija ni u jednoj od ove dve varijante.

## Joystick curriculum posle drugog uspeha

Sačuvao sam potvrđeni 26-DOF hod u `prezentacija/2nd_success_stiff_arms`.
Pokrećem ga preko `scripts/evaluate_second_success.py`. Novi 30M curriculum
pokrećem PowerShell launcherom `scripts/start_joystick_curriculum_v2.ps1`.
Koristim tri bloka sa proverom očuvanog hoda i odgovora na komande pre
povećanja težine. Detalje i ograničenja čuvam u
[JOYSTICK_CURRICULUM_V2.md](JOYSTICK_CURRICULUM_V2.md).

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
