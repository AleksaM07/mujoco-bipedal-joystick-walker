# Stand: kontrolisano poređenje transfera i učenja od početka

Ovaj eksperiment nastavlja `NEW_CHAT_HANDOFF_2026-11-05.md`. Prva varijanta
učitava walking actor i njegov observation normalizer, uz nov critic. Druga
počinje sa novim actor-om, critic-om i normalizer-om. Optimizer i brojač koraka
počinju iznova u oba slučaja. Nova opcija `TrainConfig.restore_value_fn=False`,
odnosno `train.py --no-restore-value-fn`, prosleđuje se direktno Brax-u.
Stare konfiguracije podrazumevano i dalje učitavaju critic.

Obe varijante koriste sačuvanu finalnu anatomiju i env konfiguraciju, sa
`locomotion_skill=stand` i generisanom neutralnom referencom. BVH se ne učitava.
Jednaki su seed 7, budžet 1M, learning rate `2e-5`, 64 trening okruženja,
četiri evaluaciona okruženja, 500 koraka, šest evaluacija, batch 128, unroll 10,
četiri minibatch-a i dva update-a. Backend je eksplicitno `mjx_jax`/`jax`.
ERFI i domain randomization su isključeni. Zadržani su arm reward 1.0 i svih
26 aktuatora. Brax zaokružuje budžet na cele PPO blokove; stvarni broj koraka
vidi se u imenima checkpoint-a.

Ovo poređenje kontroliše inicijalizaciju uz isti konzervativni learning rate.
Ne predstavlja pretragu optimalnog learning rate-a za scratch varijantu.

## Pokretanje u WSL-u

Iz korena repoa u PowerShell-u:

```powershell
wsl -e env XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/run_stand_ablation.py --dry-run
wsl -e env XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u scripts/run_stand_ablation.py
```

Runner pravi nov folder `runs/<UTC_timestamp>_stand_ablation_1000000_seed7`.
Trening i evaluacija svake varijante teku uzastopno u zasebnim procesima, tako
da oslobađaju GPU memoriju pre sledeće faze. `comparison.json` sadrži plan,
komande, rezultate i izabrani checkpoint; `transfer_train.log` i
`scratch_train.log` sadrže izlaze treninga. Svaki trening ima i sopstveni
`config.json`, `train.log`, XML manifest i `checkpoints/`.

Finalna natural-walk arhiva služi samo kao ulaz. Runner prihvata izlazni folder
isključivo ispod `runs/` i odbija ponovnu upotrebu postojećeg eksperimenta.

## Tumačenje loga i izbor checkpoint-a

Training logger deli Brax-ove prosečne episode sume prosečnom dužinom epizode.
Tako `stand_speed_mps` i `stand_drift_m` predstavljaju proseke preko svih
evaluiranih koraka, a `stand_eligible_pct` udeo koraka koji zadovoljavaju sve
stand uslove. Kraće epizode pravilno doprinose manjim brojem koraka.
`physical_termination_pct` predstavlja procenat evaluacionih epizoda sa
fizičkom terminacijom. Dwell i success sume se ne prikazuju kao završno stanje.

Runner naknadno proverava svaki checkpoint kroz nezavisni fizički rollout.
Rangiranje daje prednost kompletnom gate-u, broju punih epizoda, odsustvu
terminacije, završnom success-u, trajanju neuspelih epizoda, zatim manjem
maksimalnom drift-u i brzini. Reward je tek poslednji kriterijum.
`best_candidate` može zato označavati i najbolji **neuspešni** rezultat;
`stand_gate_passed` izričito kaže da li je naučeni stand prihvaćen.
Suffix `_s` na trening folderu označava završen PPO proces, ne fizički uspeh.

## Nezavisni gate

```powershell
wsl -e env XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python tools/evaluate_stand.py --checkpoint runs/<run>/checkpoints/<step> --device gpu --output runs/<run>/stand_gate.json
```

Za sve checkpoint-e jednog run-a postoje `--checkpoints-dir` i `--output-dir`.
Isti proces ponovo koristi kompajlirani physics/policy graf, dok težine ostaju
dinamičan ulaz. `--video-dir` snima fizičke rolloute; runner ga automatski
koristi za izabranog kandidata. `--skip-video` isključuje samo tu završnu fazu.

Gate koristi determinističku politiku i neomotano okruženje, bez auto-reseta.
Seed-evi 11, 22, 33 i 44 su obavezni. Svaki rollout traje tačno 500 koraka ili
do prvog fizičkog pada, nekonačne vrednosti ili drugog `done` događaja.
Izveštaj sadrži završni/max dwell, završni/max drift, max/prosečnu brzinu,
minimalnu visinu i uspravnost torza. Sva četiri roll-outa moraju završiti
500/500, bez terminacije, sa završnim `stand_success=1` i najmanje 50
neprekidnih prihvatljivih koraka. Neuspešan gate vraća exit code 1;
neispravan poziv ili konfiguracija vraćaju 2.

Neutralni reset trenutno daje iste fizičke početne uslove na sva četiri
seed-a. Ovaj gate proverava ponovljivost i održavanje balansa iz te poze.
Robustnost na perturbacije i walk→stand tranziciju zahteva naredne eksperimente.
Sačuvani Warp checkpoint-i se odbijaju jer ovaj gate ne migrira fiziku.

## Početni fizički nalaz

`analysis_outputs/skills/stand_walking_actor_four_seed_gate.json` ponavlja
walking-actor baseline: sva četiri seed-a padaju na 122/500 koraka, završni
dwell je 0, max dwell 1.04 s i max drift približno 1.559 m. To potvrđuje
neuspeh postojećeg actor-a na neutralnoj stand referenci pre PPO učenja.
Arhivirani walking checkpoint se učitava samo uz `--walking-baseline`.

## Provera implementacije

Pun unittest suite završio je sa 375/375 uspešnih testova. Naknadne fokusirane
provere pokrivaju novi runner, odbijanje tihe promene physics backend-a i
dosledan XML pri resume-u. Test prenosa koristi stvarni Brax initialization
kod i proverava očuvani actor/normalizer uz nov critic. Ruff provere novih
alata i izmenjenih testova, formatiranje i `git diff --check` prolaze.
