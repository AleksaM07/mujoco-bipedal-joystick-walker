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

Runner pravi nov folder `runs/<UTC_timestamp>_stand_<reward_mode>_1000000_seed7`.
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

## Rezultat prvog 1M + 1M poređenja

Oba treninga završila su 1,024,000 koraka. Provereno je svih pet sačuvanih
checkpoint-a po varijanti. Nijedan nije prošao fizički gate. Sačuvan celokupan
izveštaj je `analysis_outputs/skills/stand_ablation_1m_20261105.json`.

| Varijanta | PPO checkpoint | Fizički koraci | Max dwell | Završni dwell | Max drift | Prosečna brzina |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transfer, najbolji | 409,600 | 169/500 | 1.08 s | 0 s | 1.547 m | 0.501 m/s |
| Scratch, najbolji | 204,800 | 144/500 | 1.14 s | 0 s | 1.531 m | 0.550 m/s |
| Transfer, poslednji | 1,024,000 | 104/500 | 0.20 s | 0 s | 1.479 m | 0.837 m/s |
| Scratch, poslednji | 1,024,000 | 103/500 | 0.10 s | 0 s | 1.488 m | 0.793 m/s |

Rezultati važe za svaki od četiri seed-a, koji imaju iste fizičke početke.
Video `runs/stand_ablation_20261105_1615Z_1m_seed7/videos/000000409600/stand_seed_11.mp4`
pregledan je i potvrđuje pad. `video_exit=1` ovde znači neuspešan fizički gate;
sva četiri videa su uspešno zapisana. Mediapy writer nije radio u instaliranom
okruženju, pa je postojeći imageio fallback uspešno snimio video.

Transfer je dao nešto dužu najbolju epizodu, ali oba run-a degradiraju prema
kraju i nema uverljivog dokaza da je walking inicijalizacija dobar put do
naučenog stand-a. Čak i neutralni PD baseline traje 148 koraka. Sledeći test
zato koristi scratch i menja samo reward, uz isti konzervativni learning rate.

## Razlog i definicija testa `balance_v1`

Pure imitation reward namerno ignoriše apsolutni root XY. Uz arm weight 1.0,
polovina ukupne nagrade dolazi iz arm imitation grane. Visoka sličnost poza
može stoga ostati prisutna i dok se gubi ravnoteža. Ovo je merljiv nesklad sa
stand kriterijumima, ali ne dokazuje da je jedini uzrok slabog PPO rezultata.

Native MuJoCo PD trag je sačuvan u
`analysis_outputs/skills/stand_pd_balance_trace_20261105.json`. Na 75. koraku
(1.5 s) brzina je već 0.151 m/s, dok je torso-up još 0.997 i visina 1.490 m
naspram početnih 1.503 m. Na 100. koraku brzina raste na 0.523 m/s. To motiviše
gušći signal za korekcije pre nego što robot znatno izgubi visinu.

Opt-in `stand_reward_mode=balance_v1` koristi:

```text
error = (speed / 0.15)^2 + (anchor_drift / 0.25)^2
      + ((1 - torso_up) / 0.10)^2 + ((height_ratio - 1) / 0.15)^2
balance_score = exp(-error) * both_feet_contact
reward = imitation_reward * (0.2 + 0.8 * balance_score)
```

Fizički pad zadržava postojeći terminalni reward -25. Dwell i success ne daju
bonus. Reference, model, observacije, action layout, reset i fizički gate su
isti. Default ostaje `imitation` radi tačne reprodukcije prethodnih run-ova.
Trening i evaluator učitavaju verziju reward-a iz sačuvanog config-a.

Jedan kontrolisani scratch test pokreće se ovako:

```powershell
wsl -e env XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u scripts/run_stand_ablation.py --variants scratch --stand-reward-mode balance_v1
```

Budžet je opet 1M, seed 7, LR `2e-5`, 64 okruženja i `bare` režim. Nema
istovremene promene discount-a, learning rate-a ili početne poze. Reward između
verzija nije uporediv kao merilo uspeha; fizički gate ostaje uporediv.

Run je pokrenut 2026-11-05 oko 21:10 UTC iz čistog commita `dc29721`:
`runs/stand_balance_v1_20261105_2110Z_1m_seed7`. Sačuvani `config.json`
potvrđuje navedene parametre i `resume_from=null`. Pri beleženju ovog statusa
worker je pripremao okruženje. Rezultat još nije poznat; proveriti
`scratch_train.log` i `comparison.json` pre nastavka ili pokretanja drugog run-a.

## Provera implementacije

Pun unittest suite sa `balance_v1` izmenama završio je sa 387/387 uspešnih
testova. Provere pokrivaju novi runner, odbijanje tihe promene physics backend-a,
dosledan XML pri resume-u, očuvanu kaznu za pad i JIT/vmap nove nagrade.
Test prenosa koristi stvarni Brax initialization kod i proverava očuvani
actor/normalizer uz nov critic. Ruff provere novih alata i izmenjenih testova,
formatiranje i `git diff --check` prolaze. CLI dry-run potvrđuje scratch bez
resume checkpoint-a i da se u kontrolisanom testu menja samo reward režim.
