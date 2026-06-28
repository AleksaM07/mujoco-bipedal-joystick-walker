# Berkeley vs Generated Biomechanics Walking

## Short Version

Berkeley humanoid je dobar baseline zato sto je vec pripremljen locomotion
benchmark. Generated biomechanics human je tezi problem zato sto model, kontakti,
actuatori, joint limits, observations i reward nisu unapred uskladjeni za RL.

## Sta Trenutni Rezultat Dokazuje

- PPO pipeline radi.
- Generated human model moze da se trenira.
- Checkpoint-i mogu da se nastave i evaluiraju.
- Forward i joystick kretanje su nauceni do proof-of-concept nivoa.

## Sta Jos Ne Dokazuje

- prirodan ljudski hod,
- hod bez klizanja stopala,
- robustan full joystick u svim pravcima,
- profesor-ready animaciju bez ogranicenja.

## Zasto Berkeley Hoda Lakse

- model je vec stabilan za locomotion,
- kontakti su cistiji,
- actuatori i joint limits su bolje podeseni,
- reward i observation su vec prilagodjeni zadatku,
- fizicki problem je laksi od custom generated human modela.

## Glavni Problem Generated Modela

Velocity tracking sam ne garantuje lep hod. Ako reward ne kaznjava dovoljno
klizanje i los swing/stance ciklus, policy moze da nadje precicu:

```text
dobar reward != prirodan hod
```

## Najbolji Sledeci Pravac

Ne juriti samo veci reward ili jos 100M stepova.

Prioritet:

1. contact-aware foot slip,
2. swing/stance kvalitet,
3. root i foot reference,
4. BVH/mocap imitation,
5. jasna vizuelna evaluacija.

## Kratko Za Prezentaciju

Berkeley baseline pokazuje kako izgleda pripremljen humanoidni locomotion task.
Generated biomechanics model je tezi, ali je PPO pipeline uspeo da proizvede
kretanje. Ogranicenje je kvalitet hoda: za prirodniji gait treba jaci
contact-aware ili motion-imitation signal.
