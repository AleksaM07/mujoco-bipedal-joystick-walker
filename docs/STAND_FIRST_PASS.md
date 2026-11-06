# Prvi uspešan stand gate — 2026-11-06

Nastavak najboljeg noćnog checkpoint-a završio je svih 1,024,000 dodatnih PPO
koraka. **Svih pet sačuvanih checkpoint-a prolazi fizički stand gate** na
seed-ovima 11, 22, 33 i 44: 500/500 koraka bez pada, završni success 1 i svih
10 sekundi neprekinutog dwell-a. Rezultati su u
`analysis_outputs/skills/stand_first_pass_20261106.json`.

| Novi koraci | Max drift | Prosečna brzina | Završni dwell | Gate |
| ---: | ---: | ---: | ---: | --- |
| 204,800 | 4.287 cm | 0.01079 m/s | 10 s | prolazi |
| **409,600** | **2.062 cm** | **0.00546 m/s** | **10 s** | **izabran** |
| 614,400 | 6.776 cm | 0.01121 m/s | 10 s | prolazi |
| 819,200 | 4.349 cm | 0.00779 m/s | 10 s | prolazi |
| 1,024,000 | 2.470 cm | 0.00365 m/s | 10 s | prolazi |

Rangiranje posle fizičkog prolaza daje prednost manjem maksimalnom drift-u,
zato je izabran checkpoint 409,600, iako poslednji ima manju prosečnu brzinu.
Izabrani checkpoint ima 1,433,600 koraka u lancu učenja; završni ima 2,048,000.
Finalna stohastička PPO evaluacija takođe traje 500/500 koraka, sa reward-om
1452.7141. Reward nije korišćen kao zamena za fizički gate.

Run:
`runs/stand_balance_v1_continue_20261106_1m_lr3e-4`

Izabrani checkpoint je u podfolderu:
`resume/20261106_0802_stand_balance_v1_resume_1000000_lr0_0003_V1_steps-1m_reward-1452p7141_s/checkpoints/000000409600`.

Pregledani su frejmovi na razmacima od dve sekunde i završni frejm videa
`videos/000000409600/stand_seed_11.mp4`: robot ostaje uspravan i miruje i na
kraju epizode. Ponovljeni video rollout takođe prolazi gate, uz max drift
2.061 cm. Male numeričke razlike između dva procesa ne menjaju zaključak.
Sva četiri videa su sada završena i ponovljeni video gate prolazi na sva četiri
seed-a. Konačni `comparison.json` beleži `status=complete`, `stand_gate_passed=true`,
`video_complete=true` i `video_exit=0`. Potvrda je dodata sačuvanom izveštaju.
Korisnik je pregledao snimak, ocenio ga kao dobar i odobrio narednu proveru.

Ovo potvrđuje naučen stand **iz neutralnog reseta tokom 10 sekundi**. Svi seed-ovi
počinju iz iste fizičke poze. Duže održavanje stajanja, perturbacije i ulazak
iz hoda još nisu provereni. Sledeći korak je duži rollout izabranog i poslednjeg
checkpoint-a, pa kontrolisane perturbacije; tek zatim walk→brake→stand.
Natural-walk arhiva i joystick grana ostaju netaknute.
