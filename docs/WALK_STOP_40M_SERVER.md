# Walk → brake → stand: prvi 40M server test

## Zašto prethodni 40M ne nastavljamo

Server arhiva `20261116_2258_transition40m_a_stride...zip` ispravno je
raspakovana i provereno je svih 20 checkpoint-a od 2,007M do 40,141M. Reward se
popravio sa približno `-193` na `-129`, a low-height završeci su nestali, ali su
se pretvorili u tipping: završni checkpoint ima 75% tipped evaluacija i samo
25% motion-over završetaka. Najbolji completion odnos bio je zapravo na početku
treninga, ne na 40M. Zato Phase B nije pokrenuta.

Problem nije samo izbor `133_21`. Direktni PD tracking bez politike pada na pet
najboljih kinematičkih start/stop kandidata:

| BVH | Prvi fizički pad |
| --- | ---: |
| `137_29` | 0,68 s |
| `133_23` | 1,32 s |
| `104_29` | 1,44 s |
| `132_32` | 1,54 s |
| `104_28` | 1,68 s |

Poređenja i trace CSV fajlovi su u `videos/dynamic_screen_*`. To pokazuje da
veći budžet na istoj retargetovanoj CMU transition familiji ne rešava osnovni
dinamički mismatch.

Ovaj test više ne prati nestabilni CMU `133_21` transition take. Koristi
zamrznute ulaze finalnog prirodnog hoda (`02_01`, XML i walking checkpoint), ali
sve nove težine i izlaze piše u `runs/`.

`walk_stop` resetuje robota u različitim fazama potvrđenog walking ciklusa,
održava hod najmanje 2 s, dolazi do kalibrisanog double-support trenutka, tokom
0,8 s glatko menja walking target u neutralni stand target i zatim zahteva
fizički stabilno stajanje. Stand deo koristi brzinu, drift, uspravnost, visinu i
kontakt oba stopala; sama imitation nagrada nije dovoljna za prolaz.

Pre treninga je finalni walking actor provučen kroz isti fizički rollout. Hod i
blend završi do koraka 183, ali actor zatim pada na koraku 205, uz brzinu oko
2,55 m/s i drift 1,06 m. Izveštaj je u
`analysis_outputs/skills/walk_stop_walking_actor_baseline.json`. To je očekivani
negativni baseline koji 40M run mora da poboljša, a ne već rešen prelaz.

Pokretanje na serveru:

```bash
set -o pipefail
NUM_ENVS=1024 NUM_EVAL_ENVS=32 \
  bash scripts/run_walk_stop_40m.sh 2>&1 | tee walk_stop_40m_server.log
```

Launcher odbija duplikat ako je drugi `train.py` aktivan. Pokreće tačno 40M,
bez automatske druge faze. Sledeći trening je dozvoljen tek nakon fizičkog gate-a
checkpoint-a:

```bash
.venv/bin/python tools/evaluate_walk_stop.py \
  --checkpoint runs/<run>/checkpoints/<step> \
  --output runs/<run>/walk_stop_gate.json \
  --device gpu
```

Prolaz zahteva 500/500 koraka na sva četiri seed-a, stvarno završen prelaz,
bez fizičkog pada i najmanje jednu završnu sekundu brzine ispod 0,15 m/s i
drifta ispod 0,25 m uz oba stopala u kontaktu.
