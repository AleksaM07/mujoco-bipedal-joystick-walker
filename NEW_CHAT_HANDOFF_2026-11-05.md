# Handoff za novi chat — skills, stand i budući A→B

Datum: 2026-11-05

## Najnovije — dijagnostika završena, pokrenut nastavak sa manjim LR

Provera 2026-11-09: oba slučaja u `runs/stand_control_diagnostic_20261106`
su završena. Nema aktivnog prethodnog treninga/evaluacije. Izvorni horizon60
checkpoint pada na 6.96 s, push-trained 409,600 na 6.94 s. Sve fizičke metrike
i svi snimljeni qpos/qvel frejmovi tačno odgovaraju prethodnim evaluacijama:
instrumentacija nije promenila ove putanje.

U istom intervalu 5.0–6.0 s nijedna ankle-z komanda, PD cilj ili snimljeni
moment ne dostiže svoju granicu. Maksimalne pozitivne sirove akcije levo/desno
su 0.559/0.153 za izvor i 0.325/0.078 za push-trained kandidat. To ne podržava
hipotezu zasićenja ankle komandi kao uzrok početnog gubitka balansa. Desno koleno
dostiže fizički limit ekstenzije; ne tvrditi da su svi limiti nebitni.
Dokazi: `analysis_outputs/skills/stand_control_diagnostic_20261106.json`.

Pokrenut je **jedan novi 1M nastavak**, folder
`runs/stand_pushx_continue_20261109_1m_lr1e-4`, iz push-trained 409,600
(3,481,600 koraka u lancu). Taj izvor prolazi 60 s i +X/±Y, ali ne -X.
LR je `1e-4` umesto `3e-4`; ista 60 s epizoda, `balance_v1`, seed 7, 64 env-a
i isti trening potisci do 5% težine. Actor/critic/normalizer se vraćaju;
optimizer je nov. Ovo proverava konzervativniji nastavak posle kasnije regresije,
ne dokazuje da je LR njen uzrok. Nema promene fizike ili akcionog raspona.
Stvarni WSL dry-run prolazi i potvrđuje izvor i konfiguraciju.

Dokazi i plan su sačuvani i pushovani u `4373829`. Run je pokrenut iz tog čistog
commita 2026-11-09 u 13:51:49 UTC (14:51 lokalno), skriveni Windows launcher
PID 31584 pri startu. Tačna komanda je u susednom `.launch.json`; logovi su
`.stdout.log`, `.stderr.log` i `resume_train.log` unutar foldera.
`comparison.json` potvrđuje `status=running`, `stage=training`,
`robustness_requested=true`. Stvarni `resume/*/config.json` u potpunosti se
poklapa sa WSL dry-run planom, uključujući LR, checkpoint i critic restore.
Pri proveri u 14:52 parent/worker imaju WSL PID 432/433; konstrukcija okruženja
je završena i worker radi reset dijagnostiku.
PID i status ponovo proveriti, jer još nema rezultata novog treninga.

Sledeća tačka: prvo proveriti postojeći `comparison.json`, `resume_train.log`
i proces, **bez pokretanja duplikata**. Runner automatski radi kratke gate-ove,
četiri videa i svih 25 dužih provera. Posle završetka pregledati potpuni
`robustness/report.json`, `robustness_candidates_passed` i snimak pre izbora
narednog koraka. Nijedan dosadašnji kandidat ne prolazi ceo robustness suite.
Noćni red se ne pokreće ponovo; walk→stand i A→B ostaju odloženi.
Natural-walk arhiva i joystick grana ostaju netaknute.

## Prethodni korak — push evaluacija završena, pokrenuta dijagnostika komandi

`runs/stand_pushx_20261106_1m_lr3e-4` je završio ceo pipeline i svih 25 provera.
**Nijedan checkpoint ne prolazi pun robustness test.** Prva dva prolaze 60 s
i +X/±Y, ali padaju pod -X; 614,400 dodatno pada pod +X, a 819,200 gubi nominalni
balans na 40.06 s. Poslednji checkpoint pada na 4.18 s u svim slučajevima, pre
početka sile. Ne nastavljati automatski poslednje težine. Sačuvani izvor
`stand_horizon60.../000001024000` ostaje prethodni stabilni kandidat.

Kompletni dokazi: `analysis_outputs/skills/stand_pushx_robustness_20261106.json`
i `docs/STAND_PUSHX_RESULTS.md`. Kinematička rekonstrukcija iz snimljenih poza
je u `analysis_outputs/skills/stand_pushx_kinematics_20261106.json`: posle -X
potiska težište nastavlja iza zadnjih kontakata stopala, uz ispravljanje kolena.
Uski ankle residual raspon je hipoteza za proveru, ne još dokazan uzrok.

Pokrenuta je kratka evaluacija bez treninga: samo postojeći -X probe na
izvornom horizon60 finalnom checkpoint-u i push-trained 409,600, uz
`--control-trace --probes push_x_neg`. Beleži akcije, smoothed akcije, stvarne
PD ciljeve, uglove i momente, kao i efektivne/ XML granice. Ne menja težine,
akcioni raspon ili fiziku. Podskup testova ne može se označiti kao pun prolaz.
Petnaest fokusiranih testova i Ruff prolaze.

Rezultati su u commitu `a7f9f91`, dijagnostički alat u `f34efc2`; oba su
pushovana. Dijagnostika je pokrenuta iz čistog `f34efc2` 2026-11-06 u 20:16 UTC
(21:16 lokalno), Windows launcher PID 28940 pri startu, u
`runs/stand_control_diagnostic_20261106`. Susedni `.launch.json` beleži tačne
argumente. Početni `report.json` potvrđivao je `status=running`, `full_suite=false`,
`requested_probes=[push_x_neg]` i `control_trace=true`.

Tadašnja sledeća tačka bila je sačekati oba slučaja i uporediti ciljeve/granice;
završeni rezultat i odluka opisani su u najnovijem odeljku iznad.
Natural-walk arhiva i joystick grana ostaju netaknute; A→B ostaje odložen.

## Prethodna provera — kratki gate gotov, duža evaluacija u toku

Provera 2026-11-06 oko 20:10 lokalno: `runs/stand_pushx_20261106_1m_lr3e-4`
završio je 1,024,000 novih trening koraka, svih pet kratkih gate-ova i četiri
videa. **Prva četiri checkpoint-a prolaze kratak gate, poslednji pada na
209/500 = 4.18 s** na sva četiri seed-a. Ne nastavljati automatski poslednji
checkpoint uprkos 60 s stohastičke PPO evaluacije.

Najbolji kratki gate je 614,400: max drift 0.990 cm, završni dwell 10 s.
Pregledani raspoređeni frejmovi i završni frejm videa potvrđuju stajanje do
kraja. Dokazi su u `analysis_outputs/skills/stand_pushx_short_gates_20261106.json`
i `docs/STAND_PUSHX_RESULTS.md`. Potpuna robustnost još nije poznata.

`comparison.json` je `status=running`, `stage=robustness`. Pri proveri je
završeno 5/25 slučajeva: checkpoint 204,800 prolazi 60 s i +X/±Y potiske,
ali pada pod -X na 6.98 s. Parent/evaluator WSL PID tada su bili 429/9115.
Proveriti sadašnji `robustness/report.json` i procese; **ne pokretati duplikat
ili novi trening dok ova evaluacija ne završi**. Sledeća tačka je pregled
svih 25 rezultata i izbor kandidata po celom fizičkom kriterijumu. Ako nijedan
ne prođe, prvo dijagnostikovati neuspeh pre novog kontrolisanog testa.
Natural-walk arhiva i joystick grana ostaju netaknute, A→B je i dalje odložen.

## Prethodni korak — 60 s stajanja prolazi, ostaje -X oporavak

Run `runs/stand_horizon60_20261106_1m_lr3e-4` je **završen**, uključujući
svih 25 robustness slučajeva i četiri kratka videa. Četiri checkpoint-a
prolaze svih 60 s stajanja. Poslednja dva (819,200 i 1,024,000) prolaze i
+X/±Y potiske; svi padaju pod -X potiskom. Finalni checkpoint pada na 6.96 s,
ali bez potiska drži 60 s uz max drift 2.436 cm. Nijedan nije potpuno robustan.

Dokazi: `analysis_outputs/skills/stand_horizon60_20261106.json` i
`docs/STAND_HORIZON60_RESULTS.md`. Dodatno su renderovani i pregledani
raspoređeni frejmovi iz `robustness/000001024000/hold_60s.mp4` i
`push_x_neg.mp4`; potvrđuju dugo stajanje i preostali pad.

Korisnik je tražio nastavak. Pokrenut je **jedan dodatni 1M run** iz finalnog
checkpoint-a (3,072,000 koraka u dosadašnjem lancu), ista 60 s epizoda,
`balance_v1`, LR `3e-4`, seed 7 i 64 env-a. Jedina nova trening opcija je
`--stand-push-max-fraction 0.05`: polovina epizoda u očekivanju ostaje bez
potiska; ostale imaju -X impulse jačine 0–5% težine, trajanja 0.2 s na svakih
10 s, sa fazom 2–8 s. Jačina/faza se ponovo izvlače na resetu. PPO evaluacija
ostaje bez potisaka, a kriterijumi nezavisne evaluacije ostaju nepromenjeni.

Aktivni folder: `runs/stand_pushx_20261106_1m_lr3e-4`.
Pun suite prolazi 412/412 testova, a WSL dry-run potvrđuje stvarni checkpoint
i konfiguraciju. Rezultati su sačuvani u `4cdf552`, implementacija u `8294630`;
oba commita su pushovana. Run je pokrenut iz čistog `8294630` 2026-11-06 u
17:10 UTC (18:10 lokalno), skriveni Windows/WSL launcher PID 28456 pri startu.
Tačna komanda je u susednom `.launch.json`; logovi u `.stdout.log`, `.stderr.log`
i `resume_train.log` unutar foldera. Rezultat ovog novog treninga još nije poznat.
Stvarni `resume/*/config.json` potvrđuje svih 1M koraka, 3,000-step epizodu,
LR `3e-4`, seed 7, 64 env-a, `stand_push_max_fraction=0.05`, critic restore
i tačan source finalni checkpoint. `comparison.json` beleži aktivan trening
i `robustness_requested=true`; pri poslednjoj proveri worker priprema okruženje.

Tačna sledeća tačka: prvo proveriti ovaj postojeći proces i `comparison.json`;
**ne pokretati duplikat**. Runner sam radi kratak gate, četiri videa i svih
25 dužih testova nakon treninga. Fizički uspeh zahteva kandidat u
`robustness_candidates_passed`, očuvano dugo stajanje i pregled snimka;
sam `status=complete` znači da je pipeline završen, ne da je stand robustan.
A→B i tranzicija iz hoda čekaju potvrđen oporavak. Noćni red ostaje završen.

## Prethodni korak — duži stand test nije prošao, 2026-11-06

Robustness run `runs/stand_robustness_20261106_selected_final` je **završen**:
svih 10 slučajeva je provereno, nijedan ne prolazi ceo fizički kriterijum.
Checkpoint 409,600 pada posle 19.42 s bez potiska, a 1,024,000 posle 19.34 s.
Kratak oporavak posle nekih potisaka ne znači uspeh: završni stand nije održan.
Finalni checkpoint pod +Y potiskom završava svih 20 s bez pada, ali izlazi
iz stand kriterijuma; ostali slučajevi završavaju fizičkim padom.

Prvih 500 koraka produženog testa izabranog checkpoint-a tačno reprodukuje
prethodni uspešan gate (max drift 0.020615 m, mean speed 0.005462 m/s).
Dokazi: `analysis_outputs/skills/stand_robustness_20261106.json` i
`docs/STAND_ROBUSTNESS.md`. Ne proglašavati robustan stand na osnovu 10 s videa.

Korisnik je ponovo odobrio nastavak. Pokrenut je **jedan novi 1M nastavak**
iz poslednjeg checkpoint-a jutarnjeg run-a, uz jedinu promenu treninga:
`episode_length=3000` umesto 500 (60 s umesto 10 s). Isti `balance_v1`,
LR `3e-4`, seed 7, 64 env-a, actor/critic/normalizer restore; optimizer je nov.
Bez novih potisaka tokom treninga. Runner dobija `--robustness-check` da
automatski proveri svih pet dužih slučajeva na svakom novom checkpoint-u,
posle standardnog gate-a i videa. Ciljni folder:

`runs/stand_horizon60_20261106_1m_lr3e-4`

Pokrenut 2026-11-06 u 13:14 UTC (14:14 lokalno), iz čistog commita `c37f924`.
Rezultati prethodne provere su u `3aa49dc`; oba commita su pushovana.
Skriveni Windows/WSL launcher imao je PID 26156 pri pokretanju. Tačna komanda,
vreme i source commit su u susednom `.launch.json`, logovi u `.stdout.log`
i `.stderr.log`; glavni trening log je `resume_train.log`. `comparison.json`
potvrđuje `status=running`, `stage=training` i `robustness_requested=true`.
Sačuvani stvarni `resume/*/config.json` potvrđuje epizode od 3,000 koraka,
1M budžet, LR `3e-4`, seed 7, 64 env-a, `balance_v1`, vraćanje critic-a i
tačan izvorni finalni checkpoint. Pri poslednjoj proveri parent/worker su
imali WSL PID 337/436; konstrukcija okruženja je završena, reset dijagnostika
je u toku. PID-eve ponovo proveriti; rezultat treninga još nije poznat.

Tačna sledeća tačka: prvo proveriti ovaj postojeći proces, `comparison.json`
i `resume_train.log`; **ne pokretati duplikat**. Runner sam radi trening,
kratak gate, video i svih 25 produženih testova. Posle završetka proveriti
`robustness/report.json` i `robustness_candidates_passed`, pregledati rezultate
i sačuvane poze/video pre narednog koraka. Prolazak kratkog gate-a sam nije
dovoljan. Noćni red ostaje završen. Walk→brake→stand i A→B čekaju.

## Prethodni korak — prvi uspešan stand gate i pokretanje duže provere

Korisnik je pregledao video i odobrio nastavak. Sva četiri videa su sada gotova,
run ima `status=complete`, `stand_gate_passed=true`, `video_complete=true`.
Pokrenuta je sledeća **fizička provera zamrznutih težina**, bez novog treninga:

`runs/stand_robustness_20261106_selected_final`

Protokol `docs/STAND_ROBUSTNESS.md`, implementacija `823bc51`: za checkpoint-e
409,600 i 1,024,000 proverava 60 s neprekinutog stajanja i četiri potiska od
5% težine modela, po 0.2 s, u pravcima ±world X/±world Y. Recovery mora imati
novu 1 s eligible sekvencu unutar 5 s od prestanka sile i uspešan kraj 20 s
epizode. Seed je 11; pravci su različiti kontrolisani fizički uslovi.

Pokrenuto 2026-11-06 u 08:36 UTC (09:36 lokalno), Windows launcher PID 27984
pri pokretanju. `report.json` se čuva posle svakog slučaja; `.stdout.log` i
`.stderr.log` su uz folder u `runs/`. CSV tragovi sadrže metrike i sile svakog
koraka, a NPZ sačuvane poze za kasniji video. Prvo proveriti ovaj proces i
izveštaj, ne pokretati duplikat. Rezultat je naknadno sačuvan u odeljku iznad.

Jutarnji 1M nastavak je završio trening i fizičku proveru. **Svih pet novih
checkpoint-a prolazi 500/500 na četiri obavezna seed-a**, bez pada, sa završnim
success=1 i dwell=10 s. Najbolji po fizičkom rangiranju je novi checkpoint
409,600: max drift 0.020615 m i mean speed 0.005462 m/s. Ukupno ima 1,433,600
koraka u lancu učenja. Zadržati i poslednji checkpoint: i on prolazi gate.

Dokazi: `analysis_outputs/skills/stand_first_pass_20261106.json` i
`docs/STAND_FIRST_PASS.md`. Pregledani su raspoređeni frejmovi i završni frejm
videa seed-a 11; potvrđuju mirno stajanje do kraja. Kompletan video pipeline
je završen pre pokretanja nove robustness provere.

Sledeći korak: duža provera održavanja stajanja izabranog i poslednjeg checkpoint-a,
pa kontrolisane perturbacije. Trenutni rezultat pokriva 10 s iz identičnog
neutralnog reseta; walk→brake→stand i A→B još nisu provereni. Ne pokretati novi
scratch/overnight trening niti menjati nagradu samo radi višeg reward-a.

## Prethodni rezultat — noćni red i pokretanje jutarnjeg nastavka

Noćni red je završen u 03:44 UTC (04:44 lokalno), tačno četiri dodatna run-a,
bez tehničkih grešaka. Status je `budget_exhausted`. LR `3e-4` dao je prvi
checkpoint sa 500/500 koraka bez pada, ali završni stand gate i dalje pada:
max dwell 9.26 s, završni dwell 0, završni drift 0.489 m. Sva četiri seed-a
daju isti neutralni fizički početak. Pregled snimka potvrđuje dugo stajanje
i izlazak iz balansa u poslednjoj sekundi, dok je robot na kraju još uspravan.

Rezultati svih 25 gate-ova: `analysis_outputs/skills/stand_overnight_20261106.json`.
Detalji i naredni korak: `docs/STAND_OVERNIGHT_RESULTS.md`.
Korisnik je jutros tražio da se nastavi. Pokrenut je **jedan 1M nastavak**
najboljeg stand checkpoint-a, isti `balance_v1` i LR `3e-4`, sa vraćanjem
actor/critic/normalizer težina. Implementacija je u commitu `56ad729`, rezultati
noćnog reda u `1670313`; oba su pushovana.

**Run sa završenim treningom/gate-om**, pokrenut 2026-11-06 u 07:01 UTC (08:01 lokalno):

`runs/stand_balance_v1_continue_20261106_1m_lr3e-4`

Startovan je iz čistog commita `56ad729` kao skriven Windows/WSL proces
(Windows launcher PID 16560 u trenutku pokretanja). Sačuvani config potvrđuje
tačan izvorni checkpoint `run_4_lr0.0003/.../checkpoints/000001024000`, seed 7,
LR `3e-4`, `balance_v1`, 1M traženih koraka i `restore_value_fn=true`.
Pri poslednjoj proveri završio je konstrukciju okruženja i radio reset
dijagnostiku. Log je `resume_train.log`; status je `comparison.json`.
Runner sam proverava sve nove checkpoint-e i snima najboljeg kandidata.

Sledeća tačka: proveriti ovaj postojeći proces i rezultat, pregledati gate/video
i uporediti sa izvornih 500/500 uz završni dwell 0. Ne pokretati duplikat.
Noćni red se ne pokreće ponovo. A→B ostaje odložen do potvrđenog stand-a.

## Noćni nastavak — istorija pokretanja

Korisnik je pred spavanje izričito odobrio **najviše četiri dodatna treninga
od po 1M koraka**. Implementacija je u commitu `b85899d`, a postupak u
`docs/STAND_OVERNIGHT.md`. Prošlo je 11 fokusiranih testova, Ruff i stvarna WSL
provera baseline config-a. Ovo je unapred definisan lokalni red eksperimenata,
ne novi AI agent koji samostalno menja kod.

Red je pokrenut 2026-11-05 u 22:25 UTC kao skriven Windows/WSL pozadinski proces:

`runs/stand_overnight_20261105_4x1m`

Pri proveri je imao Windows launcher PID 15988 i WSL PID 7789. `queue.json`
potvrđuje `status=waiting`, limit 4 i LR redom `6e-5`, `1e-4`, `2e-4`, `3e-4`.
Čeka završetak postojećeg balance_v1 run-a, uključujući video i izlazak parent
procesa 324. Taj trening je već završio 1,024,000 koraka; u trenutku pokretanja
reda radio je izabrani video rollout. PID-eve ponovo proveriti, ne pretpostavljati.

Tačka nastavka pri pokretanju: prvo pročitati `queue.json` i `summary.md` u ovom folderu
i proveriti procese. Ne pokretati duplikat niti dodatne run-ove preko odobrena četiri.
Red sam staje na fizičkom uspehu, tehničkoj grešci, STOP fajlu ili limitu. Ujutru
pregledati najbolje fizičke rezultate i video; A→B ostaje odložen do stand uspeha.
Zaustavljanje pre sledećeg run-a: napraviti prazan `STOP` u folderu noćnog reda.
Status čeka kraj aktivnog treninga/evaluacije; ne ubijati procese bez razloga.

## Aktuelni status — neuspešan prvi par, zatim `balance_v1`

Na skills grani dodati su fresh-critic restore (`--no-restore-value-fn`),
smislen stand logging i nezavisni checkpoint gate na četiri obavezna seed-a.
Detalji, tačne komande i ograničenja su u `docs/STAND_ABLATION.md`.
Runner `scripts/run_stand_ablation.py` koristi arhivirani env/PPO config za
transfer i scratch varijantu, po 1M koraka, bez ERFI/domain randomization-a.
Proverava svaki checkpoint fizički i čuva video izabranog kandidata.
Originalni tekst ispod opisuje stanje pre ovih izmena; status treninga i
rezultate treba čitati iz novog izveštaja eksperimenta.

Proverena implementacija je sačuvana u commitu `c5f3fc0`. Pun suite je prošao
375/375 testova, uz naknadne uspešne fokusirane provere novih izmena.
Ponovljen GPU baseline na četiri seed-a pada na 122/500 koraka.
Prvi transfer→scratch eksperiment pokrenut je u:

`runs/stand_ablation_20261105_1615Z_1m_seed7`

Prvo poređenje je **završeno i nije prošlo fizički gate**. Oba treninga imaju
1,024,000 stvarnih koraka (zaokruženje na PPO blokove). Najbolji transfer je
checkpoint 409,600: 169/500 fizičkih koraka. Najbolji scratch je checkpoint
204,800: 144/500. Sva četiri seed-a daju iste neutralne početke; oba kandidata
padaju i imaju završni dwell 0. Svi checkpoint-i su provereni, a video najboljeg
transfer kandidata potvrđuje pad. Ne nastavljati ka walk→brake→stand ili A→B.

Sačuvan izveštaj: `analysis_outputs/skills/stand_ablation_1m_20261105.json`.
Dijagnoza i naredni kontrolisani test opisani su u `docs/STAND_ABLATION.md`:
stand reward `balance_v1`, scratch, isti PPO parametri i budžet. Nova
implementacija prolazi pun suite od 387 testova i dodatnu proveru logger-a.
Rezultati su u commitu `15c777a`, implementacija u `dc29721`.
Korisnik je izričito odobrio push skills grane; oba commita su objavljena.
Natural-walk arhiva i joystick grana su netaknute.

**Završen prethodni eksperiment**, pokrenut 2026-11-05 oko 21:10 UTC:

`runs/stand_balance_v1_20261105_2110Z_1m_seed7`

Pokrenut je iz čistog commita `dc29721`. Sačuvani config potvrđuje scratch bez
resume-a, `balance_v1`, 1M traženih koraka, seed 7, LR `2e-5`, 64 env-a i bare.
Pri poslednjoj proveri pripremao je okruženje; nema još fizičkog rezultata.
Parent/worker su tada imali WSL PID 324/424 (PID proveriti, ne pretpostavljati).
Log je `scratch_train.log`, zbirni status `comparison.json`. Runner nakon
treninga sam proverava sve checkpoint-e na četiri seed-a i snima najbolji
kandidat. **Ne pokretati duplikat**: prvo proveriti procese, log i status.
Sledeća tačka rada je sačekati/obraditi taj rezultat, pregledati video i
uporediti fizički gate sa prethodnim scratch-om; reward između verzija nije
merilo napretka. Ne prelaziti na A→B bez uspešnog stand gate-a.

## Arhivirani početni handoff

Odeljci ispod, do završne kratke tačke nastavka, opisuju stanje pre prvog PPO
poređenja. Aktuelni rezultati i sledeći eksperiment su iznad i u
`docs/STAND_ABLATION.md`.

## Poruka koju treba poslati novom chatu

> Nastavi rad iz `NEW_CHAT_HANDOFF_2026-11-05.md`. Radi na grani
> `skills_walking_with_anatomical_human_robot`. Prvo pročitaj handoff i
> `docs/SKILLS_REVIEW.md`, proveri `git status` i poslednja tri commita. Nemoj
> menjati finalnu natural-walk arhivu niti joystick granu. Sledeći cilj je
> naučena stand veština, sa fizičkim gate-om od 500/500 koraka, a tek zatim
> walk→brake→stand i A→B. Pre PPO run-a reši ispravno prenošenje actor/critic
> težina i logging stand metrike; ne proglašavaj uspeh samo po reward-u.

## Glavni cilj projekta

Krajnja arhitektura treba da podrži odvojene veštine istog anatomskog humanoida:

1. prirodan hod — završen;
2. stabilno stajanje;
3. walk → kočenje → stajanje;
4. dolazak od tačke A do tačke B;
5. kasnije trčanje/dodatni locomotion stilovi;
6. kasnije ustajanje nakon pada;
7. tek na kraju izbegavanje prepreka/metaka.

Joystick nije trenutni razvojni prioritet. Sačuvan je kao zasebna grana za
master rad. Ne koristiti kod iz odbačenog projekta/grane
`bipedal-humanoid-bullet-avoidance`: korisnik je izričito rekao da ništa iz tog
A→B sistema, state machine-a, scenario manifesta ili goal testova ne valja.

## Trenutne grane

Postoje tri namerne linije rada:

| Grana | Namena | Trenutni lokalni commit |
| --- | --- | --- |
| `natural_walking_with_anatomical_human_robot` | Zamrznuti prirodan DeepMimic hod | `2a6156e` |
| `joystick_walking_with_anatomical_human_robot` | Sačuvani joystick sistem za master | `80edb67` |
| `skills_walking_with_anatomical_human_robot` | Aktivni stand → A→B razvoj | `f7f49fd` u trenutku ovog handoff-a |

Tag `natural-walk-v1.0` pokazuje na `2a6156e`.

Važno: remote trenutno prikazuje samo natural-walk i joystick grane. Skills
grana i tag su lokalni dok se eksplicitno ne pushuju. Radno stablo je bilo čisto
pre pravljenja ovog handoff dokumenta.

## Završeni natural-walk rezultat

Finalni rezultat je u:

`prezentacija/5th_success_final_natural_walk`

Tu se nalaze finalni checkpoint, XML, BVH, config, manifest, dokazi i video.
Launcher:

```powershell
./.venv-win/Scripts/python.exe scripts/evaluate_final_success.py
```

Dokazano stanje:

- 26 aktivnih aktuatora: trup, noge, ramena i laktovi;
- jedan očišćen CMU `02_01.bvh` walking ciklus;
- 500/500 koraka bez fizičkog pada na četiri fiksna seed-a;
- finalni eval reward `1446.7544`;
- DeepMimic pose približno `0.995`;
- arm-pose približno `0.994`;
- prirodan hod i aktivni laktovi;
- završni rezultat je dobijen elbow curriculum-om
  `0.25 → 0.50 → 0.75 → 1.00`, po 1M koraka po nivou.

Natural-walk arhivu ne menjati i ne trenirati preko njenog checkpoint foldera.
Novi run-ovi treba da pišu u `runs/`.

## Najvažniji zaključak o postojećoj walking politici

Finalna politika je **pure reference tracker**, a ne politika koja zaista prati
joystick/velocity komande:

- `_is_pure_reference_imitation()` isključuje komande;
- evaluator takođe postavlja komandu na nulu;
- rezidualna akcija se dodaje na BVH target;
- čak i nulta akcija nastavlja da reprodukuje hodajuću referencu;
- BVH sat napreduje nezavisno od komande;
- politika nema goal XY u observation-u.

Zbog toga komanda brzine nula ne znači „stani“. Zamrzavanje BVH frejma usred
koraka takođe nije ispravna stand veština. Za A→B su potrebni stand, tranzicija,
goal observation i task reward.

Finalni checkpoint koristi:

- actor observation: `state[366]`;
- critic observation: `privileged_state[809]`;
- action: `26`;
- kontrolni korak: `0.02 s`;
- simulacioni korak: `0.005 s`.

To omogućava tehničko učitavanje postojećih težina u stand env, ali ista
dimenzija nije dokaz da walking actor ume da stoji.

## Šta je urađeno na skills grani

### Commit `e780538`

`fix: complete arm metrics in reference-free physics steps`

Popravljen je pravi `KeyError` u non-BVH `step()` grani. Nedostajale su vrednosti:

- `arm`;
- `arm_pose`;
- `arm_velocity`;
- `arm_pose_error`;
- `arm_velocity_error`.

Reset ih je već imao, step nije. Popravka ne menja finalni BVH hod.

### Commit `fae61b1`

`feat: add an isolated stationary imitation skill and physical stand checks`

Dodato je:

- `EnvConfig.locomotion_skill`, default `"walk"` radi starih checkpoint-a;
- CLI: `train.py --locomotion-skill stand`;
- `skills/stand.py` sa `StandEnv`;
- `skills/stand_metrics.py` sa fizičkim gate-ovima;
- isti izbor env klase u treningu i evaluatoru;
- `tools/check_stand.py`;
- testovi za JIT/vmap, pravi MJX reset/step, observation/action layout, batch
  evaluator i saved-config kompatibilnost.

Stand referenca je dosledna neutralna poza istog anatomskog modela:

- jedan nepokretan target;
- nulte joint/root brzine;
- root, FK markeri, reward, future targets i residual action center svi koriste
  isti target;
- faza i referentni sat ostaju nula;
- jedan frejm ne završava epizodu;
- stand režim ne učitava BVH, iako zadržava `reference_gait="bvh"` kao postojeći
  interfejs za pure imitation kod; izvor se beleži kao
  `generated:neutral_stand`.

Stand success kriterijum:

- XY brzina `< 0.15 m/s`;
- XY drift od početne pozicije `< 0.25 m`;
- torso up `>= 0.9`;
- visina `>= 85%` početne;
- oba stopala zaista dodiruju pod;
- nema fizičkog pada ili NaN/Inf;
- svi uslovi neprekidno traju najmanje `1.0 s = 50` control koraka.

Svaki prekid resetuje dwell. Success ne završava epizodu i ne daje terminalni
bonus. To je namerno: robot mora da izdrži celu epizodu, a ne da na trenutak
ispuni kriterijum.

### Commit `f7f49fd`

`docs: record skills architecture review and failed stand transfer baselines`

Sačuvani su pregled i dva početna fizička izveštaja:

- `docs/SKILLS_REVIEW.md`;
- `analysis_outputs/skills/stand_preflight.json`;
- `analysis_outputs/skills/stand_walking_actor_probe.json`.

## Šta je fizički izmereno za stand

### Neutralni PD target bez politike

Komanda:

```powershell
./.venv-win/Scripts/python.exe tools/check_stand.py
```

Rezultat pod punom gravitacijom i sa free root-om:

- referenca je nepokretna;
- sve referentne brzine su nula;
- svi targeti su unutar actuator/control limita;
- oba stopala počinju sa malim preload kontaktom;
- robot pada posle `148/500` koraka, odnosno `2.96 s`;
- maksimalni neprekinuti dwell pre gubitka balansa: `1.48 s`;
- test nije prošao.

Zaključak: neutralna poza je koherentna, ali statički PD target nije feedback
kontroler balansa. PPO ima smisla, jer politika mora da nauči korekcije.

### Finalni walking actor na stand referenci

Komanda:

```powershell
./.venv-win/Scripts/python.exe tools/check_stand.py --mjx-steps 500 --policy --output analysis_outputs/skills/stand_walking_actor_probe.json
```

Rezultat:

- checkpoint se učitava: 366/809 observations i 26 actions su kompatibilni;
- pada posle `122/500` koraka, odnosno `2.44 s`;
- maksimalni dwell je oko `1.04 s`;
- završni dwell je `0`;
- završni drift je oko `1.56 m`;
- walking actor nije stand politika.

Ovo je jedan deterministički probe, ne statistička evaluacija.

## Testovi i trenutno poznata ograničenja

Pre poslednjih commit-a pun suite je pokrenut nad 361 testom. Jedini pad bio je
stari test koji je očekivao da svako `EnvConfig` polje ide direktno u base env;
`locomotion_skill` sada namerno bira env klasu u factory-ju. Test je prilagođen,
a svih 29 testova saved-config modula zatim je prošlo. Stand-focused testovi i
Ruff provera prolaze.

Ponoviti full suite posle sledećih izmena:

```powershell
./.venv-win/Scripts/python.exe -m unittest discover -s tests -v
```

Poznato ograničenje evaluation logger-a: Brax sabira svaku env metriku kroz
epizodu. Zato:

- `eval/episode_stand_success` bi bio broj uspešnih frejmova, ne boolean uspeh;
- `eval/episode_stand_dwell_seconds` bi bio zbir rastućeg tajmera, ne završni dwell;
- trenutni `TrainingProgressLogger` još ne prikazuje stand metrike.

Ne tumačiti te agregate kao victory. Potreban je zaseban završni rollout/gate
koji proverava 500/500 koraka, bez fizičke terminacije i finalni `stand_success`.

## Trenutno dostupno izvršavanje

Lokalni WSL vidi:

- `NVIDIA GeForce RTX 2060`;
- `6144 MiB` VRAM;
- JAX `0.11.0`;
- `CudaDevice(id=0)`.

Windows JAX služi za CPU dijagnostiku. WSL `.venv/bin/python` služi za GPU
trening. U trenutku provere nije postojao aktivan RL training proces.

Prethodni finalni 1M elbow run sa 64 env-a trajao je oko 1 h 8 min na ovom GPU-u,
sa približno 360–380 training steps/s posle compile-a. Stand run slične veličine
verovatno je istog reda, ali to tek treba izmeriti.

## Sledeći zadatak — pre dugog treninga

Ne pokretati odmah 30M. Prvo napraviti mali, merljiv stand eksperiment.

### 1. Ispravno preneti checkpoint

Brax `restore_checkpoint_path` trenutno vraća:

- actor;
- critic;
- observation normalizer;
- ne vraća optimizer state ni env step counter.

Za stand transfer critic iz walking zadatka nema pouzdano značenje i treba
omogućiti `restore_value_fn=False` kroz naš `TrainConfig`/CLI, zatim proslediti
ga u `ppo.train`. Actor i normalizer su tehnički kompatibilni, ali se semantics
reference menjaju, pa transfer mora biti eksperiment, ne pretpostavka.

Preporučen prvi par kratkih ablation run-ova sa istim seed-om i budžetom:

1. `walk actor + saved normalizer + fresh critic`;
2. potpuno `from scratch` stand politika.

Ne menjati finalni checkpoint. Oba run-a pišu u nove `runs/` foldere. Ako je
transfer jasno bolji, nastaviti njega; ako nije, koristiti scratch stand.

### 2. Popraviti stand reporting

Dodati u log samo agregate koji imaju jasno značenje:

- prosečna stand brzina po koraku;
- prosečan anchor drift po koraku;
- procenat eligible koraka;
- episode length i physical termination;
- finalni gate zasebnom rollout skriptom, ne preko sume dwell tajmera.

Najbolji checkpoint ne treba birati samo po imitation reward-u. Za stand je
prioritet:

1. `500/500` koraka;
2. `terminated=0`;
3. finalni `stand_success=1`;
4. mali drift i brzina;
5. tek onda reward/pose kvalitet.

### 3. Dodati deterministički checkpoint gate

Proširiti `tools/check_stand.py` ili napraviti mali poseban evaluator koji:

- prima `--checkpoint`;
- proverava bar seed-eve `11, 22, 33, 44`;
- za svaki radi tačno 500 koraka ili do physical termination;
- zapisuje finalni dwell, max drift, max/mean speed, minimal torso-up i height;
- prolazi samo ako sva četiri seed-a završe 500/500 i finalni success je 1;
- ne koristi zbir `episode_stand_success` kao boolean.

### 4. Pokrenuti prvo mali PPO poređenje

Razuman početni budžet je 1M po varijanti, ne 30M. Koristiti finalni XML i stand
mode, 64 env-a kao proverenu lokalnu konfiguraciju i bez ERFI/domain randomization
u prvom čistom testu. Learning rate za transfer treba ostati konzervativan
(`2e-5` kao finalni elbow fine-tuning); scratch može tražiti veći LR, ali prvo
treba napraviti fer kontrolisano poređenje.

Komandu ne kopirati napamet dok se ne doda fresh-critic opcija i proveri `--help`.
Osnova će biti približno:

```text
PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u train.py \
  --device gpu \
  --locomotion-skill stand \
  --timesteps 1000000 \
  --num-envs 64 \
  --num-eval-envs 4 \
  --batch-size 128 \
  --num-evals 6 \
  --episode-length 500 \
  --seed 7 \
  --recovery-physics-version consistent_v6 \
  --reference-loop-closure bounded \
  --reference-action-mode residual \
  --reference-replay-target-step 0 \
  --deepmimic-reward-mode pure \
  --arm-actuators \
  --xml-path prezentacija/5th_success_final_natural_walk/generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml \
  --bare \
  --learning-rate 0.00002 \
  --run-tag stand_transfer_1m
```

Transfer varijanta treba da doda finalni checkpoint i fresh critic opciju.
Stand mode ignoriše BVH fajl, ali parser trenutno može automatski popuniti default
BVH listu pre nego što factory napravi StandEnv. Proveriti da li to samo ostaje
neiskorišćen config podatak ili prilagoditi parser tako da stand ne popunjava BVH
putanju. Ne dozvoliti da duga BVH obrada slučajno uđe u stand konstruktor.

## Posle uspešnog stand-a

Tek kada stand prođe četiri seed-a:

1. napraviti walk→brake→stand curriculum;
2. resetovati tranziciju iz više faza hoda;
3. meriti vreme i put kočenja;
4. zadržati walking validation da se otkrije catastrophic forgetting;
5. zatim dodati lokalni goal XY i forward A→B reward;
6. uspeh A→B: udaljenost `< 0.25 m`, brzina `< 0.15 m/s`, dwell `>= 1 s`, bez pada;
7. tek potom ciljevi levo/desno, iza, finalni yaw i povezani ciljevi bez physics
   reseta.

Planirana arhitektura:

```text
lokalni XY cilj
    -> high-level goal controller
    -> željena brzina i heading
    -> naučena locomotion veština i tranzicije
    -> 26 residual motor akcija
```

DeepMimic/imitation reward čuva prirodnost. Poseban task reward meri dolazak,
progress, kočenje i dwell. Goal mora biti u policy observation-u; trenutno nije.

## Radovi i zaključci o multi-BVH/AMP/ASE

- DeepMimic potvrđuje kombinovanje imitation i task cilja i treniranje više
  veština/politika.
- AMP koristi skup nestrukturiranih motion klipova kao adversarial style prior,
  bez eksplicitnog biranja klipa. To je ozbiljna buduća promena algoritma.
- ASE uči latentni prostor ponovo upotrebljivih veština, ali traži znatno veći
  simulacioni budžet i nije gotov dodatak postojećem checkpoint-u.
- Više BVH klipova se ne „usrednjava“ automatski. Multi-clip može poboljšati
  raznovrsnost i robustnost, ali tek posle jasnog single-skill stand/A→B sistema.
- Natural-walk uspeh sa jednim `02_01.bvh` je dovoljan i završen. Ne vraćati se
  na beskonačno poliranje tog rezultata.

## Git pravila i zaštita rezultata

- Svaka razvojna faza treba da ima zaseban, detaljan commit.
- Pre svakog commit-a proveriti `git status` i diff.
- Ne brisati `runs/`, `prezentacija/`, `videos/`, BVH/SMPL podatke ili finalni
  checkpoint.
- Ne menjati natural-walk i joystick granu tokom skills razvoja.
- Ne pushovati ili brisati remote grane bez jasne namere; skills granu ipak treba
  pushovati kada sledeći provereni commit bude spreman, jer je trenutno lokalna.
- Ne proglašavati uspeh samo iz reward krive. Uvek tražiti fizički rollout i video.

## Fajlovi koje prvo treba otvoriti

1. `NEW_CHAT_HANDOFF_2026-11-05.md` — ovaj dokument;
2. `docs/SKILLS_REVIEW.md` — prethodni arhitektonski review;
3. `skills/stand.py` — stand environment;
4. `skills/stand_metrics.py` — victory kriterijum;
5. `tools/check_stand.py` — preflight i walking-actor probe;
6. `train.py` — checkpoint restore, logging i PPO poziv;
7. `prezentacija/5th_success_final_natural_walk/config.json` — finalna dobra
   env/PPO konfiguracija.

## Kratko: tačna početna tačka novog chata

Natural walking je završen i zaštićen. Fresh-critic transfer, smislen logging
i nezavisni gate su implementirani. Prvi 1M transfer i 1M scratch su završeni;
prvi par nije prošao stand gate. Noćna LR pretraga našla je bolji kandidat,
a jutarnji 1M nastavak zatim dao pet checkpoint-a koji svi prolaze 500/500,
završni success i dwell=10 s. Izabran je novi checkpoint 409,600 sa drift-om
oko 2.1 cm. Sledeća duža provera otkrila je pad oko 19.4 s, pa je odrađen
1M nastavak sa 60 s epizodama. On daje četiri uspešna 60 s checkpoint-a;
poslednja dva prolaze i tri pravca potiska, ali svi još padaju pod -X.
Pročitati najnoviji odeljak i `docs/STAND_HORIZON60_RESULTS.md`: sledeći
kontrolisani test dodaje samo trening potiske u tom smeru. Ako taj run već
postoji, prvo proveriti njegov proces/izveštaj, ne pokretati duplikat.
Walk→brake→stand i A→B ostaju odloženi.
