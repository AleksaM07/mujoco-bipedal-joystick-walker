# Kako mi koristimo grane i arhivske tagove

Posle čišćenja 2. novembra 2026. na glavnom GitHub repository-ju držimo jednu
kanonsku granu:

| Grana | Uloga |
| --- | --- |
| `main` | Naša proverena linija sa prvim uspešnim hodom i sređenom dokumentacijom. |

Stara `main` istorija bila je direktni predak recovery linije, pa smo `main`
bez prepisivanja istorije pomerili unapred za 102 commita. Privremeni naziv
`recovery/project-v2-imitation` zatim smo uklonili. Stari vrh `main` grane ostaje
sačuvan tagom `archive/20261102/main-before-first-walk`.

Za sledeću funkcionalnost pravimo novu kratkotrajnu granu iz `main`, umesto da
ponovo održavamo više nejasnih dugovečnih linija.

## Šta smo arhivirali

Mi smo stare nazive grana zamenili anotiranim tagovima koji čuvaju njihove tačne
vrhove:

| Arhivski tag | Nekadašnja grana | Commit |
| --- | --- | --- |
| `archive/20261102/main-before-first-walk` | stari vrh `main` | `33eaa1f` |
| `archive/20261102/main-modern-bvh` | `experiment/main-modern-bvh` | `a6ca23d` |
| `archive/20261102/v21-stable-baseline` | `experiment/v21-stable-baseline` | `cd6557c` |
| `archive/20261102/backup-before-test` | `backup/with-7f1294e-before-test` | `de5cc52` |
| `archive/20261102/project-v2-before-recovery` | `project_v2_refrences_only` | `8b4d442` |
| `archive/20261102/lets-call-it-dave` | lokalni `lets_call_it_dave` | `0cd3da8` |
| `archive/20261102/diagnosis-stash` | `stash@{0}: dijagnostika` | `ddf7a7c` |

Prvih pet arhivskih tagova i milestone tag `v0.1-first-walk` nalaze se i na
`origin` remote-u. Tagove za lokalni stari rad i dijagnostički stash čuvamo
lokalno i u bundle-u.

Mi vraćamo granu iz taga bez menjanja trenutnog checkout-a:

```bash
git branch restored/v21-stable-baseline archive/20261102/v21-stable-baseline
```

Mi uspešan milestone otvaramo pomoću taga:

```bash
git show v0.1-first-walk
```

## Lokalna sigurnosna kopija

Pre uklanjanja grana napravili smo i verifikovali:

`artifacts/repo_cleanup_20261102/branches.bundle`

Njegov SHA-256 je:

```text
7c9633e72e3722ec5688e30e5fa777469b345cbf836e5c49d9db420bee133310
```

Bundle je inkrementalan i oslanja se na zadržanu istoriju repository-ja. Mi ga
koristimo kao dodatni lokalni oporavak, dok su remote tagovi primarna dugoročna
arhiva starih eksperimentalnih vrhova.

## Drugi repository

Remote `newrepo` i njegov `newrepo/main` ostaju netaknuti. Tamo se nalazi ranija
linija `bipedal-humanoid-bullet-avoidance`. Mi ćemo je analizirati tek kada
stabilizujemo joystick, ruke i ustajanje; ne mešamo je sada u walking runtime.
