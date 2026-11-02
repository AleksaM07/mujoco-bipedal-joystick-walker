# Humanoid koji je naučio da hoda

Mi treniramo humanoida u MuJoCo simulatoru pomoću PPO-a i BVH reference.
Na grani **`recovery/project-v2-imitation`** imamo prvi vizuelno potvrđen hod:
checkpoint **`000025681920`**, `consistent_v6`, jedan ciklus `02_01.bvh` i 18 akcija.
Trenutno učimo imitaciju sa modelom bez aktivnih ruku. Joystick, ustajanje i
izbegavanje projektila ostaju naši naredni zadaci.

## Odavde krećemo

| Kada želimo… | Otvaramo… |
| --- | --- |
| Da vidimo uspeh | [Četiri videa](prezentacija/1st_success/videos/) |
| Da razumemo šta smo popravili | [1ST_SUCCESS.txt](prezentacija/1st_success/1ST_SUCCESS.txt) i [preciziranja](prezentacija/1st_success/PROVERA_I_ISPRAVKE.txt) |
| Da pokažemo rezultat | [Graf 10 prekretnica](prezentacija/1st_success/10_NAJBITNIJIH_STVARI.png) |
| Da pokrenemo evaluaciju ili trening | [Komande i okruženja](docs/WORKFLOW.md) |
| Da razumemo fajlove | [Mapa repository-ja](docs/REPOSITORY.md) |
| Da nastavimo istraživanje | [Naš plan](docs/ROADMAP.md) |
| Da pronađemo staru dijagnostiku | [Istorijska arhiva](docs/archive/README.md) |
| Da vratimo staru granu | [Arhiva grana](docs/BRANCHES.md) |

## Uspešan model pokrećemo na Windowsu

Iz korena projekta, u Git Bash-u ili PowerShell-u, koristimo postojeće Windows
okruženje (jedan red):

```bash
./.venv-win/Scripts/python.exe scripts/evaluate_first_success.py
```

Ovom komandom učitavamo kopiju checkpoint-a, XML-a i BVH-a iz `prezentacija/`.
Za samostalnu proveru arhive bez simulacije koristimo:

```bash
./.venv-win/Scripts/python.exe tools/verify_success_archive.py --require-tracked
```

Detalje instalacije, snimanja, headless provere i WSL treninga držimo u
[WORKFLOW.md](docs/WORKFLOW.md). Linux `.venv` i Windows `.venv-win` koristimo
odvojeno. Našu trenutnu `uv` konfiguraciju sa CUDA zavisnostima koristimo za Linux;
još nemamo proverenu instalaciju iz tog lockfile-a na čistom Windows računaru.

## Šta smo izmerili

Na najboljoj trening evaluaciji dobili smo reward **1396.087**, prosek
**500/500 koraka** i sva četiri evaluaciona sveta do horizonta. Naknadno smo
snimili četiri deterministička rollout-a sa seedovima 11, 22, 33 i 44: svaki je
završio 500 koraka, odnosno 10 sekundi, bez prijavljenog pada.

Pod „93.1% kvaliteta” podrazumevamo normalizovani reward ovog zadatka. Prirodnost
hoda smo vizuelno potvrdili; tim procentom ne merimo biomehaničku tačnost.
Ovim testovima još ne dokazujemo proizvoljno dugu stabilnost, upravljanje do
ciljne tačke ili otpornost na guranje. Referentni XY pomak od oko 1.41 m/ciklus
ne poistovećujemo sa izmerenom putanjom politike.

Mi čuvamo rezultat u [prezentacija/1st_success/](prezentacija/1st_success/):
kompletan checkpoint, konfiguraciju, XML/BVH, log, videe i SHA-256 manifest.
Najbolji checkpoint razlikujemo od poslednjeg checkpoint-a završenog run-a.

## Kako organizujemo rad

Mi zadržavamo `train.py` i `evaluate.py` kao glavne ulaze. Runtime module držimo
u korenu radi postojećih import-a i kompatibilnosti. Svakodnevne komande držimo
u `scripts/`, dijagnostiku u `tools/`, a regresione testove u `tests/`.
Stare izveštaje čuvamo u `docs/archive/`; ne čitamo ih kao aktuelni status.

Mi testiramo iz postojećeg okruženja bez instaliranja dodatnih paketa:

```bash
./.venv-win/Scripts/python.exe -m unittest discover -s tests -v
```

Naše test klase su kompatibilne i sa pytest-om. U postojeći Windows venv još
nismo instalirali pytest. Reference i izvore pratimo u [REFERENCES.md](REFERENCES.md)
i [registry.yaml](references/registry.yaml).
