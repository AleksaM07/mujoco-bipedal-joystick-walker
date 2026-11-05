# Noćni stand red — 2026-11-05

Korisnik je odobrio **najviše četiri dodatna treninga od po 1M koraka**.
Ovo je lokalni, unapred definisan red eksperimenata. Ne pokreće novi AI chat
i ne donosi samostalne odluke o promeni koda.

Prvo čeka ceo postojeći eksperiment:
`runs/stand_balance_v1_20261105_2110Z_1m_seed7` — trening, fizičke provere svih
checkpoint-a i četiri videa izabranog kandidata. Čeka i izlazak parent procesa
da se oslobodi GPU. Proverava da sačuvana konfiguracija odgovara baseline-u.

Ako baseline ne prođe, red pokreće sledeće scratch eksperimente:

| Redosled | Learning rate | Traženi koraci |
| --- | ---: | ---: |
| 1 | `6e-5` | 1,000,000 |
| 2 | `1e-4` | 1,000,000 |
| 3 | `2e-4` | 1,000,000 |
| 4 | `3e-4` | 1,000,000 |

Svi koriste `balance_v1`, seed 7, 64 env-a, isti XML, reset, observation/action
layout i ostale PPO parametre. Menjaju se samo LR i naziv run-a. To je kontrolisana
pretraga learning rate-a, ne nastavak težina prethodnog kandidata. Prethodni LR
`2e-5` potiče iz walking fine-tuning-a i njegov izbor za scratch još nije potvrđen.
Brax zaokružuje svaki run na 1,024,000 stvarnih koraka.

Svaki run sam proverava sve checkpoint-e i snima video najboljeg kandidata.
Red staje posle prvog prolaza fizičkog gate-a, tehničke greške ili četvrtog run-a.
Gate ostaje 500/500 koraka na seed-ovima 11/22/33/44 i završni success sa dwell-om
od najmanje jedne sekunde. Isti početni uslovi ne dokazuju robustnost; video se
ujutru dodatno pregleda. Nijedan rezultat automatski ne pokreće A→B.

## Pokretanje i izlaz

Red je pokrenut iz commita `b85899d` 2026-11-05 u 22:25 UTC, kao skriven
pozadinski WSL proces. Potvrđeni su `queue.json` sa statusom `waiting`, Windows
launcher PID 15988 i WSL PID 7789. Launcher logovi su
`runs/stand_overnight_20261105_4x1m.stdout.log` i odgovarajući `.stderr.log`.
Proveriti postojeći proces i izveštaj pre ponovnog pozivanja komande.

Pripremljena komanda iz korena repoa u PowerShell-u:

```powershell
wsl -e env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_stand_overnight.py --after runs/stand_balance_v1_20261105_2110Z_1m_seed7 --after-pid 324 --output runs/stand_overnight_20261105_4x1m --max-runs 4
```

PID 324 važi za parent postojećeg run-a u ovoj sesiji. Proveriti ga pre ručnog
pokretanja. `--dry-run` prikazuje plan bez rezervacije reda ili korišćenja GPU-a.
Izlaz mora biti nov folder pod `runs/`. OS lock sprečava dva istovremena reda.
Ako parent nestane bez potpunog izveštaja ili ostane aktivan preko tri sata,
red se zaustavlja i beleži grešku.

Status i jutarnji pregled nalaze se u:

- `runs/stand_overnight_20261105_4x1m/queue.json`
- `runs/stand_overnight_20261105_4x1m/summary.md`
- `run_1.log` do `run_4.log`, uz pojedinačne foldere treninga/gate-ova/videa.

`budget_exhausted` znači da je potrošen odobreni broj run-ova bez potvrđenog
stand-a. `success` označava prolaz fizičkog gate-a, uz naknadni ljudski pregled.
Za mirno zaustavljanje pre sledećeg treninga napraviti prazan fajl:

```powershell
New-Item runs/stand_overnight_20261105_4x1m/STOP -ItemType File
```

Aktivni trening i njegova evaluacija tada se dovršavaju. Red ne briše rezultate,
ne menja grane, ne radi push i ne menja finalnu natural-walk arhivu. Računar mora
ostati uključen, bez sleep/hibernacije, sa otvorenim poklopcem i na napajanju.
Pri pripremi su AC/DC sleep timeout-i već bili nula; power podešavanja nisu menjana.

## Provera

Fokusirani testovi simuliraju uspeh, iscrpljen budžet, neuspešan subprocess,
nepotpun video, pogrešne success flag-ove, nedostajuće seed-ove, nestanak parent
procesa i STOP. Poseban test potvrđuje da LR menja samo rate i naziv run-a.
WSL dry-run i poređenje stvarnog baseline config-a prolaze bez GPU treninga.
