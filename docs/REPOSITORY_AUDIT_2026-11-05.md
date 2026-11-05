# Repository audit — 2026-11-05

## Zaključak

Produkciona walking linija je tehnički koherentna: runtime moduli imaju jasne
uloge, regresioni testovi pokrivaju pronađene reference/physics greške, a uspešni
checkpoint-i imaju odvojene arhive. Najveći problem nije bio aktivni kod nego
istorijski materijal pomešan sa trenutnim entrypoint-ima.

## Šta je ostalo

- `main` i aktivna `experiment/arms-ik-v2` grana;
- svi runtime moduli i regresioni testovi;
- `runs/`, `prezentacija/`, `videos/` i aktivni campaign;
- svi BVH/SMPL podaci i generated XML modeli;
- aktuelni natural-walk ablation izveštaji;
- materijali za master rad;
- Git bundle dok ne završimo remote konsolidaciju.

## Šta je uklonjeno

- četiri redundantna lokalna branch imena, čiji su commiti već u aktivnoj grani;
- lokalni remote zapis za odbačeni bullet repository;
- privremeni clone-ovi bullet/DeepMimic/MimicKit projekata i smoke/frame cache;
- stari phase izveštaji i recovery analysis output-i;
- neimportovani legacy Berkeley runner/notebook;
- završeni recovery, arms i joystick shell/PowerShell launcheri.

Git istorija čuva verzionisane obrisane fajlove. Privremeni clone-ovi nisu
jedinstveni podaci i mogu se ponovo preuzeti. Obrisani cache/frame fajlovi nisu
obnovljivi iz Git-a, ali su generisani artefakti.

## Šta namerno nije dirano

- Produkcioni Python kod nije premeštan niti funkcionalno menjan.
- `main` nije checkout-ovan, rebased ili pomeran.
- Remote eksperimentalne grane nisu obrisane.
- Veliki Git history rewrite nije rađen.
- Stari XML modeli nisu obrisani jer ih sačuvani runovi mogu referencirati.
- `marina_bvh_parse` nije obrisan jer ga kompatibilni preset još koristi.

## Sledeće odluke

1. Sačekati sva četiri elbow gate-a i izabrati završni checkpoint.
2. Snimiti video i zamrznuti peti uspeh ako 100% nivo prođe.
3. Tada push-ovati aktivnu granu i ukloniti zastarele remote eksperimente.
4. Napraviti joystick-only branch iz zaštićenog `main` stanja.
5. Novi A→B razvijati bez koda iz odbačenog bullet projekta.
