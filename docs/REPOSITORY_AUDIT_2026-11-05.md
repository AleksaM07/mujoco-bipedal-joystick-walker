# Repository audit — 2026-11-05

## Zaključak

Produkciona walking linija je tehnički koherentna: runtime moduli imaju jasne
uloge, regresioni testovi pokrivaju pronađene reference/physics greške, a uspešni
checkpoint-i imaju odvojene arhive. Najveći problem nije bio aktivni kod nego
istorijski materijal pomešan sa trenutnim entrypoint-ima.

## Šta je ostalo

- finalna natural-walk i zaštićena joystick grana;
- svi runtime moduli i regresioni testovi;
- `runs/`, finalna `prezentacija/` arhiva i `videos/`;
- svi BVH/SMPL podaci i generated XML modeli;
- materijali za master rad;
- Git bundle dok ne završimo remote konsolidaciju.

## Šta je uklonjeno

- četiri redundantna lokalna branch imena, čiji su commiti već u aktivnoj grani;
- lokalni remote zapis za odbačeni bullet repository;
- privremeni clone-ovi bullet/DeepMimic/MimicKit projekata i smoke/frame cache;
- stari phase izveštaji i recovery analysis output-i;
- neimportovani legacy Berkeley runner/notebook;
- završeni recovery, arms i joystick shell/PowerShell launcheri.
- međuarhive prvog, drugog, trećeg i četvrtog walking rezultata;
- jednokratni migration/curriculum launcheri koji su zavisili od njih.

Git istorija čuva verzionisane obrisane fajlove. Privremeni clone-ovi nisu
jedinstveni podaci i mogu se ponovo preuzeti. Obrisani cache/frame fajlovi nisu
obnovljivi iz Git-a, ali su generisani artefakti.

## Šta namerno nije dirano

- Produkcioni Python kod nije premeštan niti funkcionalno menjan.
- Finalni checkpoint, XML, BVH, evidence i videi nisu menjani.
- Veliki Git history rewrite nije rađen.
- Stari XML modeli nisu obrisani jer ih sačuvani runovi mogu referencirati.
- `marina_bvh_parse` nije obrisan jer ga kompatibilni preset još koristi.

## Rezultat elbow curriculum-a

Sva četiri nivoa (`0.25`, `0.50`, `0.75`, `1.00`) prošla su automatski gate.
Završni checkpoint završava 500/500 koraka na sva četiri fiksna seed-a bez
fizičkog pada. Elbow amplituda na nivou `1.00` dostiže približno 92–100% BVH
reference na većini proba; jedan desni rollout prelazi referencu. Video je ručno
odobren i rezultat je zamrznut kao finalni natural-walk success.

- finalni run: `runs/20261105_0358_elbow_curriculum_a100_elbow_curriculum_2_V1_steps-1m_reward-1446p7544_s`;
- finalni checkpoint: `checkpoints/000001024000` unutar tog run-a;
- campaign status: `artifacts/elbow_curriculum_20261105_4m/status.json`;
- finalna arhiva: `prezentacija/5th_success_final_natural_walk`;
- regresija: 356 testova prolazi.

## Sledeće odluke

1. Održavati samo natural-walk i joystick granu.
2. Novi A→B razvijati bez koda iz odbačenog bullet projekta.
