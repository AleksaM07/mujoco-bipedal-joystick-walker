# Optimization Applied

## Cilj

Stari setup je bio pretezak za PPO: previse slobode, losi kontakti, slab reward
signal i premalo strukture za hod. Promene su uvedene da model bude trenabilniji
i da reward manje nagradjuje klizanje.

## Najbitnije Promene

### Model / XML

- 18 kontrolisanih actuatora.
- Abdomen i pelvis imaju kontrolisane, ali stiff zglobove.
- Dodati ili popravljeni foot sole kontakti.
- `trainfast_v17-v19` uvodi stroze joint ranges.
- `trainfast_v19` dodaje contact guardrails za pelvis/thigh/shank.
- Non-foot kontakti su filtrirani da stopala budu glavni kontakt sa podom.

### Actuatori

- Action scale vise nije isti za sve zglobove.
- Stride zglobovi imaju vise slobode.
- Hip roll/yaw, ankle roll, abdomen i pelvis su ograniceniji.
- Actuator `ctrlrange`, `kp` i `forcerange` su joint-specific.

### Observations

- Policy dobija cistiji `state`.
- Critic dobija siri `privileged_state`.
- Dodati gait phase signali.
- BVH target moze biti deo observation-a.

### Reward

- Velocity tracking nije dovoljan sam.
- Dodati su:
  - upright/base-height,
  - action/action-rate costs,
  - foot slip,
  - swing clearance,
  - stance contact,
  - locomotion quality,
  - BVH pose/velocity/root/foot reference.
- Reward je promenjen da klizanje i los kontakt mogu stvarno da budu kaznjeni.

### BVH / Imitation

- Dodata BVH reference gait podrska.
- Dodati root i foot reference signali.
- Dodata reference velocity komponenta.
- `reference_state_init` postoji, ali je eksperimentalan dok retargeting nije
  fizicki konzistentan.

## Najvaznija Lekcija

Policy ne treba samo da ide trazenom brzinom. Mora da izgleda kao da hoda:

```text
tracking + survival + contact quality + imitation > raw reward
```

## Trenutni Status

Pipeline je funkcionalan. `v19` XML je najkorisniji za dalje treninge jer ima
bolje joint/action priore i contact guardrails.

## Sta Ne Treba Raditi Prvo

- Ne pustati odmah ogromne 60M-100M run-ove bez kratkog gate testa.
- Ne gledati samo reward.
- Ne ukljucivati ERFI/domain randomization pre stabilnog osnovnog hoda.
- Ne koristiti full BVH listu dok mali debug set ne pokazuje napredak.

## Preporuka

Kratki gate run, vizuelna provera, pa tek onda duzi trening.
