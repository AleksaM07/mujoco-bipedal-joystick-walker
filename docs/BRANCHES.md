# Kako mi koristimo grane i arhivske tagove

Posle čišćenja 2. novembra 2026. na glavnom GitHub repository-ju držimo dve
grane sa jasnom ulogom:

| Grana | Uloga |
| --- | --- |
| `main` | Naša ranija objedinjena osnova; ne sadrži novi DeepMimic uspeh. |
| `recovery/project-v2-imitation` | Naša aktivna linija sa prvim uspešnim hodom. |

Mi trenutno radimo na `recovery/project-v2-imitation`. Kada završimo sređivanje
i proverimo čist checkout, možemo zasebno odlučiti da li ova linija postaje novi
`main`. Samo čišćenje ne menja podrazumevanu granu automatski.

## Šta smo arhivirali

Mi smo stare nazive grana zamenili anotiranim tagovima koji čuvaju njihove tačne
vrhove:

| Arhivski tag | Nekadašnja grana | Commit |
| --- | --- | --- |
| `archive/20261102/main-modern-bvh` | `experiment/main-modern-bvh` | `a6ca23d` |
| `archive/20261102/v21-stable-baseline` | `experiment/v21-stable-baseline` | `cd6557c` |
| `archive/20261102/backup-before-test` | `backup/with-7f1294e-before-test` | `de5cc52` |
| `archive/20261102/project-v2-before-recovery` | `project_v2_refrences_only` | `8b4d442` |
| `archive/20261102/lets-call-it-dave` | lokalni `lets_call_it_dave` | `0cd3da8` |
| `archive/20261102/diagnosis-stash` | `stash@{0}: dijagnostika` | `ddf7a7c` |

Prva četiri arhivska taga i milestone tag `v0.1-first-walk` nalaze se i na
`origin` remote-u. Tagove za drugi repository i stari dijagnostički stash čuvamo
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
