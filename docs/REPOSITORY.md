# Kako čitamo naš repository

Mi za svakodnevni rad prvo otvaramo README, zatim jedan launcher i konkretan
runtime modul. Sve ostalo koristimo po potrebi.

## Aktivni kod

| Putanja | Šta ovde radimo |
| --- | --- |
| `train.py` | Pokrećemo PPO, reset/eval wrappers, checkpoint-e i logovanje. |
| `evaluate.py` | Učitavamo sačuvanu konfiguraciju, policy, viewer i video snimanje. |
| `config.py` | Definišemo EnvConfig, TrainConfig i postojeće preset vrednosti. |
| `biomechanics_env.py` | Sastavljamo fiziku, observation/action ugovor, reward i termination. |
| `biomechanics_model.py` | Gradimo humanoid XML i modele. |
| `bvh_reference.py` | Učitavamo BVH, mapiramo zglobove i biramo ciklus. |
| `reference_interpolation.py` | Pripremamo kontinuirane krive i loop closure. |
| `reference_ik.py`, `reference_root.py`, `reference_kinematics.py` | Rešavamo IK, stance/root handoff i referentnu kinematiku. |
| `smpl_reference.py` | Čuvamo podršku za SMPL i istorijske konfiguracije. |
| `phase1_backends.py` | Biramo backend; naziv je istorijski, modul je i dalje aktivan. |
| `scripts/` | Koristimo kratke ulaze za gledanje, trening i benchmark. |
| `tools/` | Pokrećemo specifične provere; [indeks alata](../tools/README.md). |
| `tests/` | Čuvamo zaštitu od ponavljanja već pronađenih grešaka. |

Mi tokom ovog čišćenja zadržavamo import putanje runtime modula. Njihovo buduće
premeštanje u Python paket radićemo kao zaseban refactor sa testovima; time ćemo
moći jasno da odvojimo promenu organizacije od promene ponašanja modela.

## Podaci i rezultati

| Putanja | Kako je koristimo |
| --- | --- |
| `prezentacija/1st_success/` | Čuvamo zamrznut dokaz uspeha i njegov model. |
| `generated_models/` | Čuvamo XML varijante; uspeh koristi `v21_noarms`. |
| `BVH_walking_animation/` | Čuvamo izvorne BVH klipove i liste; uspeh koristi `02/02_01.bvh`. |
| `CMU_SMPL+H-G/` | Čuvamo drugi motion format za postojeću SMPL podršku. |
| `marina_bvh_parse/` | Čuvamo raniji motion skup; generički one-clip preset ga još referencira. |
| `runs/` | Čuvamo lokalne treninge i checkpoint-e; većinu Git namerno ignoriše. |
| `videos/` | Čuvamo lokalna snimanja; prva četiri uspeha verzionišemo u prezentaciji. |
| `analysis_outputs/` | Čuvamo rezultate ranijih merenja potrebne za dijagnostičke izveštaje. |
| `artifacts/` | Čuvamo lokalne logove, benchmark rezultate i oporavljivu arhivu workspace-a. |
| `docs/archive/` | Čuvamo istorijske izveštaje bez mešanja sa aktuelnim uputstvom. |
| `legacy/` | Čuvamo stari Berkeley runner i stare analize. |
| `references/`, `REFERENCES.md` | Pratimo poreklo ideja i implementacija. |

Mi ne premeštamo motion/XML skupove tokom ovog prolaza: njihove putanje već
postoje u sačuvanim run konfiguracijama. Privremene klonove, probe i render
foldere sklonili smo u `artifacts/workspace_archive_20261102/`; tamo ih možemo
vratiti. Za DeepMimic i MimicKit kopije proverili smo svaki fajl SHA-256 pre
uklanjanja izvora.

## Lokalno okruženje

Mi koristimo `.venv-win` za Windows evaluaciju, a Linux/WSL `.venv` za GPU trening.
Ostala virtualna okruženja, `.uv-cache` i JAX keševe tretiramo kao lokalne resurse.
Ne verzionišemo ih i ne brišemo ih radi kozmetičkog sređivanja stabla.

Mi imamo oko 2.42 GiB spakovane Git istorije pre ovog čišćenja. Premestanjem
današnjih fajlova ili brisanjem grana ne uklanjamo istorijske velike blobove.
Eventualnu migraciju velikih podataka i rewrite istorije planiraćemo odvojeno,
jer bi promenila stare commit identifikatore.
