# Pokretanje i praćenje

Sve komande pokrećemo iz korena repository-ja.

## Windows evaluacija

```powershell
./.venv-win/Scripts/python.exe scripts/evaluate_final_success.py
./.venv-win/Scripts/python.exe evaluate.py --help
```

Prvi JAX compile može trajati nekoliko minuta. Ne proglašavamo zastoj dok proces
aktivno koristi CPU ili GPU i nema greške u logu.

## Završeni elbow curriculum

Završni curriculum dokazi su uz model u
`prezentacija/5th_success_final_natural_walk/evidence`. Prethodne međuarhive i
jednokratni launcheri uklonjeni su nakon odobrenja finalnog videa.

## Testovi

```powershell
./.venv-win/Scripts/python.exe -m unittest discover -s tests -v
```

Za brzu sintaksnu proveru promenjenih modula koristimo `python -m py_compile`.
