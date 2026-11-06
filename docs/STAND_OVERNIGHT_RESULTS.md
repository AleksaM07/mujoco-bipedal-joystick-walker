# Stand: rezultat noćne LR pretrage — 2026-11-06

Noćni red završio je tačno četiri odobrena dodatna treninga, bez tehničke
greške. Završetak je zabeležen u 03:44 UTC, odnosno 04:44 po lokalnom vremenu.
Svaki run ima 1,024,000 stvarnih PPO koraka, pet fizički proverenih checkpoint-a
i četiri videa izabranog kandidata. Rezultati svih 25 checkpoint-a, uključujući
prethodni `balance_v1` baseline, sačuvani su u
`analysis_outputs/skills/stand_overnight_20261106.json`.

| LR | Najbolji checkpoint | Koraci bez pada / 500 | Max dwell | Završni dwell | Max drift | Prosečna brzina |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2e-5` (baseline) | 819,200 | 152 | 1.72 s | 0 s | 1.472 m | 0.495 m/s |
| `6e-5` | 1,024,000 | 165 | 1.90 s | 0 s | 1.479 m | 0.468 m/s |
| `1e-4` | 1,024,000 | 168 | 1.94 s | 0 s | 1.549 m | 0.475 m/s |
| `2e-4` | 1,024,000 | 276 | 4.20 s | 0 s | 1.497 m | 0.288 m/s |
| `3e-4` | 1,024,000 | **500** | **9.26 s** | **0 s** | **0.489 m** | **0.074 m/s** |

**Nijedan checkpoint još nije prošao završni stand gate.** Kod najboljeg
kandidata sva četiri deterministička rollout-a imaju 500/500 koraka, bez pada,
ali završni success je 0. Torso-up nikad ne pada ispod 0.9884, a minimalna visina
je 92.76% početne. Poslednji deo epizode ima ubrzavanje do 1.199 m/s i drift do
0.489 m, preko dozvoljenih 0.15 m/s i 0.25 m. Završni dwell je zato nula.

Pregledani su vremenski raspoređeni frejmovi i završni frejm snimka:
`runs/stand_overnight_20261105_4x1m/run_4_lr0.0003/videos/000001024000/stand_seed_11.mp4`.
Video potvrđuje dugo mirno stajanje i izlazak iz balansa pred kraj. Snimak
završava dok je robot još uspravan; ne tvrdi se šta bi se desilo posle 10 s.

Trening-evaluacija sa stohastičkom politikom takođe napreduje, ali i dalje pada:
poslednja prosečna dužina je 254.25/500 koraka, uz 100% fizičkih terminacija.
Deterministički rollout i stohastička trening-evaluacija nisu ista provera.
Četiri deterministička seed-a imaju isti neutralni fizički početak; ovo još
nije dokaz robustnosti na perturbacije ili tranzicije iz hoda.

## Sledeći kontrolisani korak

Posle jutarnjeg zahteva korisnika da se nastavi, odabran je jedan dodatni run:
**nastavak najboljeg stand checkpoint-a za 1M koraka, sa istim LR `3e-4`**.
Reward ostaje `balance_v1`; fizika, reset, PPO parametri i acceptance gate se
ne menjaju. Novi run piše u zaseban folder. Time se proverava efekat dodatnog
učenja tamo gde se na kraju prvog miliona pojavio fizički napredak.

Učitavaju se stand actor, critic i observation normalizer. Critic ovde opisuje
isti zadatak i reward. Brax kreće sa novim optimizer state-om i brojačem koraka;
ovo je nastavak težina, ne bit-identičan nastavak optimizer-a. Novi checkpoint
1,024,000 ima ukupno 2,048,000 koraka u lancu učenja.

Runner koristi `--variants resume --resume-from <stand_checkpoint>` uz
`--stand-reward-mode balance_v1 --learning-rate 0.0003`. Proverava izvorni XML
hash, potpuno poklapanje stand okruženja i kontrolisanih PPO parametara.
Default transfer/scratch ponašanje i noćni limit ostaju isti. Trinaest fokusiranih
testova za runner i noćni red prolazi, uz Ruff i WSL dry-run stvarnog checkpoint-a.

Po završetku proveriti svaki novi checkpoint i pregledati najbolji video.
Uspeh traži i 500/500 i završni success; tek potom sledi duža provera održavanja
stajanja i perturbacije. Walk→brake→stand i A→B još nisu započeti.
