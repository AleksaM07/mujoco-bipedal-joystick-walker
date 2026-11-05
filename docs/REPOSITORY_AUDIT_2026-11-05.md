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

## Rezultat elbow curriculum-a

Sva četiri nivoa (`0.25`, `0.50`, `0.75`, `1.00`) prošla su automatski gate.
Završni checkpoint završava 500/500 koraka na sva četiri fiksna seed-a bez
fizičkog pada. Elbow amplituda na nivou `1.00` dostiže približno 92–100% BVH
reference na većini proba; jedan desni rollout prelazi referencu i zato rezultat
ostaje video-kandidat dok ga ručno ne odobrimo.

- finalni run: `runs/20261105_0358_elbow_curriculum_a100_elbow_curriculum_2_V1_steps-1m_reward-1446p7544_s`;
- finalni checkpoint: `checkpoints/000001024000` unutar tog run-a;
- campaign status: `artifacts/elbow_curriculum_20261105_4m/status.json`;
- video-kandidat: `videos/5th_success_candidate_elbow_curriculum_a100_seed11_quad.mp4`;
- regresija: 356 testova prolazi.

## Sledeće odluke

1. Ručno pogledati video i zamrznuti peti uspeh samo ako je pokret stvarno bolji.
2. Tada push-ovati aktivnu granu i ukloniti zastarele remote eksperimente.
3. Napraviti joystick-only branch iz zaštićenog `main` stanja.
4. Novi A→B razvijati bez koda iz odbačenog bullet projekta.
