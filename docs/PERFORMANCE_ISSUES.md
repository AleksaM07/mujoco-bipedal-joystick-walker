# Performance Issues

## Glavni Problem

Najskuplji resurs nije samo CPU/GPU vreme, nego eksperiment latency: los reward
ili los XML moze da potrosi ceo dan treninga bez korisnog hoda.

## Najbitniji Bottleneck-i

### 1. XML generisanje

Problem:

- XML build/validation moze da uspori start treninga.
- Cache mora da se koristi kad god je moguce.

Sta raditi:

- koristiti postojece `generated_models/*.xml`,
- menjati XML verziju samo kad je stvarno potrebno,
- ne regenerisati modele za svaki mali test.

### 2. Domain randomization

Problem:

- domain randomization povecava tezinu zadatka.
- ako basic gait nije stabilan, randomization samo zamagli problem.

Sta raditi:

- prvo trenirati bez ERFI/domain randomization,
- ukljuciti randomization tek posle stabilnog hoda.

### 3. Preveliki run-ovi

Problem:

- veliki PPO run moze dugo da trenira pogresan reward.

Sta raditi:

- prvo 2M-5M gate run,
- proveriti video i metrike,
- tek onda 50M+.

### 4. GPU memory / PPO shape

Ako ima OOM:

- smanjiti `num_envs`,
- smanjiti `batch_size`,
- paziti da `batch_size * num_minibatches` bude deljivo sa `num_envs`.

Primeri:

```powershell
--num-envs 768 --batch-size 384
--num-envs 512 --batch-size 256
```

## Metrike Za Stop/Continue

Prekini ili menjaj setup ako:

- reward raste, ali `loco_q` ne raste,
- `double_ct` ili `swing_ct` ostaju visoki,
- `done_low`, `done_tip` ili `done_illegal` ostaju visoki,
- `bvh_reg` dominira nad korisnim mimic signalom,
- video pokazuje klizanje uprkos dobrom reward-u.

Nastavi ako:

- `mimic_rew`, `ref_foot`, `ref_root` i `gated_prog` rastu zajedno,
- padovi opadaju,
- stopala se stvarno podizu,
- trajectory i torso-up izgledaju stabilnije.

## Preporuceni Workflow

1. Kratak gate run.
2. Pogledati video.
3. Proveriti `training_history.csv` i notebook metrike.
4. Ako ima smisla, produziti trening.
5. Tek na kraju ukljuciti teze varijante: ERFI, domain randomization, full BVH.

## Profiling

Ako treba bas profilisati:

```powershell
python -m cProfile -s cumtime train.py --debug-run
```

ili:

```powershell
py-spy record -o profile.svg -- python train.py --debug-run
```
