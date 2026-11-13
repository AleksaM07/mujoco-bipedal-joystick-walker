# Grane

## Kanonske grane

| Grana | Uloga |
| --- | --- |
| `natural_walking_with_anatomical_human_robot` | Finalni DeepMimic/natural-walk rezultat. |
| `joystick_walking_with_anatomical_human_robot` | Kopija potvrđenog joystick `main` stanja. |
| `navigation_walking_with_anatomical_human_robot` | Aktivni razvoj faze 2: A→B, stop, prelazi i recovery. |

Grana `skills_walking_with_anatomical_human_robot` preimenovana je 2026-11-12
u `navigation_walking_with_anatomical_human_robot`, lokalno i na origin-u,
sa očuvanom istorijom. Stari naziv ostaje samo u istorijskim izveštajima.
Natural-walk i joystick grane se ne menjaju tokom ovog razvoja.

Lokalni `newrepo` remote za odbačeni bullet/A→B projekat je uklonjen. Spoljašnji
repository time nije obrisan.

## Oporavak

Stari milestone-i ostaju dostupni kroz Git tagove i istoriju. Dodatni lokalni
bundle je:

```text
artifacts/repo_cleanup_20261102/branches.bundle
```

Bundle zadržavamo do završne remote konsolidacije.
