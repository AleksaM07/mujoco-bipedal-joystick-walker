# Grane

## Kanonske grane

| Grana | Uloga |
| --- | --- |
| `natural_walking_with_anatomical_human_robot` | Finalni DeepMimic/natural-walk rezultat. |
| `joystick_walking_with_anatomical_human_robot` | Kopija potvrđenog joystick `main` stanja. |

Sve eksperimentalne i istorijske remote grane uklanjamo nakon što su ove dve
grane push-ovane. Potrebna istorija ostaje u Git commitima i recovery bundle-u.

Lokalni `newrepo` remote za odbačeni bullet/A→B projekat je uklonjen. Spoljašnji
repository time nije obrisan.

## Oporavak

Stari milestone-i ostaju dostupni kroz Git tagove i istoriju. Dodatni lokalni
bundle je:

```text
artifacts/repo_cleanup_20261102/branches.bundle
```

Bundle zadržavamo do završne remote konsolidacije.
