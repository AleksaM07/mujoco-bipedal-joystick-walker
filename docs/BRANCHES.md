# Grane

## Lokalno stanje posle čišćenja

| Grana | Uloga |
| --- | --- |
| `main` | Zaštićena potvrđena linija sa postojećim joystick sistemom. |
| `experiment/arms-ik-v2` | Aktivni DeepMimic/natural-walk rad sa rukama. |

Četiri lokalne eksperimentalne grane su uklonjene jer su svi njihovi commiti
direktni preci `experiment/arms-ik-v2`. Njihove remote kopije još postoje i ne
brišemo ih dok ne zamrznemo završni walking checkpoint.

Lokalni `newrepo` remote za odbačeni bullet/A→B projekat je uklonjen. Spoljašnji
repository time nije obrisan.

## Ciljno stanje

Na kraju želimo tačno tri dugoročne grane:

1. završnu DeepMimic/natural-walk granu;
2. joystick-only kopiju potvrđenog `main` stanja;
3. `main` sa odvojenim joystick i novim A→B modulima.

Treću strukturu ne pravimo dok trenutni walk nije zamrznut i novi A→B nije
testiran izolovano.

## Oporavak

Stari milestone-i ostaju dostupni kroz Git tagove i istoriju. Dodatni lokalni
bundle je:

```text
artifacts/repo_cleanup_20261102/branches.bundle
```

Bundle zadržavamo do završne remote konsolidacije.
