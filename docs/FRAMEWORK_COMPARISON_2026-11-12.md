# Izbor pristupa za A→B, recovery i buduće veštine

Pregled zvaničnih repozitorijuma, dokumentacije i radova: **2026-11-12**.
Zadatak je definisan u [PROJECT_GOALS.md](PROJECT_GOALS.md).
Ovo proširuje [raniji pregled](SKILLS_REVIEW.md); nije benchmark izvršenih
spoljnih framework-a na našem humanoidu.

## Preporuka

**MimicKit je prvi kandidat za ograničen test kompatibilnosti i faze 2,
ali samo prisustvo ASE-a nije dovoljan razlog za konačan izbor.** Za prvi
A→B sistem treba proveriti jednostavniji AMP + task pristup pre velike
latentne hijerarhije. Trenutni repo koristi Brax PPO i sopstveni MuJoCo/JAX env;
nije već instaliran kompletan MimicKit/ASE sistem.

**ProtoMotions je najzanimljiviji kandidat za objedinjene generativne veštine
i parcijalne ciljeve. mjlab je ozbiljan kandidat ako je prioritet zadržavanje
MuJoCo modela i korišćenje konkretnih navigacionih/avoidance implementacija.**
To su procene prikladnosti za naš zadatak, ne izmereno rangiranje. Novi nalazi
LP-NavOA i PAC-MAN daju mjlab ekosistemu veću težinu nego poređenje samo sa
BeyondMimic tracker-om. PHC ostaje posebno važan izvor recovery postupka.

Ne postoji dokaz da bilo koji pregledani checkpoint neposredno rešava sve
tri faze na našem anatomskom modelu. Ako MimicKit zahteva previše prilagođavanja,
ostaju model, retargeting, evaluacioni protokoli, baseline i stečeno znanje;
ne vraćamo se na nulu.

## Obim pretrage i nivo dokaza

Pregled pokriva glavne praktične porodice: direktni task RL sa motion prior-om,
latentne veštine, goal-conditioned trackere, masked/generativne kontrolere,
behavioral foundation modele, zasebni get-up, perceptivnu navigaciju,
izbegavanje dolazećih objekata i model-predictive control.

Ne tvrdimo da je pregledan svaki objavljeni rad ili privatna implementacija.
Za glavne kandidate pregledani su i konkretni izvori navedeni ispod; za
prošireni spisak urađen je pregled zvaničnih opisa, dostupnosti koda i namene.
Rezultati autora važe za njihove modele i protokole. Javni kod, objavljene
težine, kompletni podaci i reprodukcija rada četiri su različita nivoa dostupnosti.
Linkovi na `main`/`master` su promenljivi; pri praktičnom testu pinovati commit
i zavisnosti, pa ponovo proveriti navedene nalaze.

## Glavni kandidati

| Kandidat | Relevantna sposobnost | Šta još treba za naš zadatak | Uloga |
| --- | --- | --- | --- |
| [MimicKit](https://github.com/xbpeng/MimicKit) | DeepMimic, AMP, ASE i drugi prior-i; dokumentovan AMP location task. | Anatomija/import, podaci, stop/dwell, prelazi i recovery moraju biti provereni; nema dokaza kompatibilnosti sa našim Brax težinama. | Prvi ograničeni praktični test, u skladu sa korisnikovom preferencijom. |
| [ProtoMotions / MaskedMimic / Metatron](https://github.com/NVlabs/ProtoMotions) | Tracking, steering, parcijalni ciljevi, scene i generativna kontrola; objavljena G1 Metatron politika sa recovery ponašanjem. | Naš model, retargeting, obuka i naši gate-ovi; G1/SOMA nisu naš XML. | Najjači kandidat po širini budućeg sistema veština; veći integracioni i trening zahtev. |
| [PHC](https://github.com/ZhengyiLuo/PHC) | Motion imitation sa namenskim fail-state recovery primitivom i naučenom kompozicijom. | Prilagođavanje modela/runtime-a i point-goal/stop zadatka; nije gotov perceptivni navigator. | Recovery uzor i rezervni tracker pravac. |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) → [mjlab](https://github.com/mujocolab/mjlab) | MuJoCo-Warp infrastruktura, velocity i motion tracking zadaci. | Task integracija, prirodnost, stop i recovery; više zasebnih projekata ne čini automatski jedan sistem. | Ozbiljna alternativa za MuJoCo razvoj i proširenja faze 3. |

### Šta tačno MimicKit daje

[ASE dokumentacija](https://github.com/xbpeng/MimicKit/blob/main/docs/README_ASE.md)
prikazuje treniranje i testiranje latentne politike. Sama ta stranica ne
potvrđuje kompletan get-up + high-level task pipeline iz originalnog ASE rada.
Pregledani [ASEEnv](https://github.com/xbpeng/MimicKit/blob/main/mimickit/envs/ase_env.py)
nasleđuje AMP i dodaje verovatnoću default reseta; to nije isto što i poseban
fall-state curriculum originalnog ASE-a.

Za fazu 2 konkretniji početak je
[AMP task primer](https://github.com/xbpeng/MimicKit/blob/main/docs/README_AMP.md):
`amp_location_humanoid_env.yaml` uz `amp_task_humanoid_agent.yaml` kombinuje
task i diskriminatorsku nagradu. U
[location kodu](https://github.com/xbpeng/MimicKit/blob/main/mimickit/envs/task_location_env.py)
postoje lokalni cilj, udaljenost, usmerena brzina i facing. To nije naš
nezavisni uslov `< 0.25 m`, `< 0.15 m/s`, `1 s`; taj gate mora da se doda/proveri.

Konkretan rizik importera: pregledani
[MJCF parser](https://github.com/xbpeng/MimicKit/blob/main/mimickit/anim/mjcf_char_model.py)
obrađuje 0, 1 ili 3 joint-a po telu, a ne 2; lokalnu rotaciju čita iz `quat`.
Naš arhivirani model ima po dva hinge joint-a u oba stopala i obe šake,
a `thorax` koristi `euler="90 0 0"`. Direktan import se zato ne sme pretpostaviti.
Potrebni su adapter i poređenje forward kinematics-a, osa, limita i action
mapiranja. Nije prihvatljivo ukloniti zglob samo da bi parser prošao.

Dokumentovani simulatori uključuju Isaac Gym, Isaac Lab i Newton;
MuJoCo XML format sam po sebi ne znači naš postojeći MJX/Brax trening backend.
Zamena framework-a je novi eksperiment.

### Šta preuzimamo iz originalnog ASE-a i PHC-a

[Originalni ASE](https://github.com/nv-tlabs/ASE) je označen kao deprecated i
upućuje na MimicKit, ali dokumentuje `HumanoidAMPGetup` i high-level
`HumanoidLocation`/heading zadatke. Njegov
[get-up kod](https://github.com/nv-tlabs/ASE/blob/main/ase/env/tasks/humanoid_amp_getup.py)
koristi pale početne poze i recovery period. `HumanoidPerturb` baca objekte
radi testa otpornosti; to nije dokaz da politika unapred opaža i izbegava udar.

[PHC rad, odeljak 3.2](https://arxiv.org/html/2305.06456)
uvodi poseban recovery primitive, trening iz neuspešnih stanja i prebacivanje
između praćenja i približavanja udaljenoj referenci; zatim uči kompoziciju
primitiva. To je dobar uzor za oporavak van uskog nominalnog stanja.
Nije dokaz da ručno mešanje našeg walking i stand action vektora daje isto
ponašanje, niti da PHC-ov recovery prolazi naš ograničeni anchor gate.

### Šta ProtoMotions/Metatron rešava, a šta ostaje nama

[Aktuelni README](https://github.com/NVlabs/ProtoMotions/blob/main/README.md)
beleži Metatron izdanje i veliku refaktorizaciju 2026-10-28, uz kontrolu
preko parcijalnih ciljeva i širi scene/recovery sistem. Starije konfiguracije
ne treba prenositi naslepo.

[Metatron trening recepti](https://github.com/NVlabs/ProtoMotions/blob/main/examples/experiments/metatron/README.md)
izričito su osnovni single-teacher recepti, ne potpuna multi-expert
co-distillation reprodukcija. Podaci za trening nisu upakovani; objavljena
Metatron politika je za G1, bez pretrained SOMA Metatron politike u tom paketu.
To ograničava očekivanje da je naš anatomski model podržan gotovim težinama.

Postoje konkretni
[recovery reset kod](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/envs/base_env/reset/recovery.py)
i [Metatron recovery primer](https://github.com/NVlabs/ProtoMotions/blob/main/deployment/native/examples/metatron_recovery.py).
Drugi koristi G1 goal prompting i predstavlja razvojni deployment primer.
Korisno je preuzeti princip i protokol; uspeh na našem modelu tek treba izmeriti.

### Zašto mjlab sada ima veću težinu

[Stari whole_body_tracking repo](https://github.com/HybridRobotics/whole_body_tracking)
arhiviran je 2026-10-16 i upućuje na mjlab.
[BeyondMimic rad/projekat](https://beyondmimic.github.io/) obuhvata i generativnu
kontrolu; dostupni tracking repo ne treba predstavljati kao ceo sistem rada.
[mjlab](https://github.com/mujocolab/mjlab) nudi direktan pristup MuJoCo
strukturama, GPU simulaciju i velocity/tracking zadatke.

Dva pronađena projekta proširuju ovaj pravac:

- [LP-NavOA](https://github.com/shenqiqishi/LP-NavOA-code): javni mjlab research
  fork za G1 lokalnu navigaciju i obilazak prepreka. Odvaja locomotion backbone
  od recurrent lokalnog planera koji menja heading komandu. To je konkretan
  uzor za A→B + prepreke; naš stop/dwell i anatomija ostaju zasebni zahtevi.
- [PAC-MAN](https://github.com/lzyang2000/perceptive_cbf_rl): javni mjlab/AMP
  trening, benchmark i deployment za G1 izbegavanje lopti, sa masked-depth
  opažanjem i CBF nagradama. Direktno relevantan za fazu 3C. Nije univerzalna
  garancija izbegavanja niti gotov checkpoint za naš model.

Ovi fork-ovi mogu koristiti različite mjlab verzije i međusobno nekompatibilne
observations/actions. Ne planirati njihovo spajanje kopiranjem foldera.

## Prošireni pregled drugih opcija

| Projekat / porodica | Šta smo proverili | Odluka za naš obim |
| --- | --- | --- |
| [CALM](https://github.com/NVlabs/CALM) | ASE naslednik sa uslovljenim latentnim veštinama, get-up task-om i style/heading kontrolom; Isaac Gym. | Relevantan ako nam zatreba kontrola stila i reusable latent; nije jednostavniji početak od AMP task-a. |
| [PULSE](https://github.com/ZhengyiLuo/PULSE) / [NCP](https://github.com/Tencent-RoboticsX/NCP) | Kontinualni odnosno diskretni motion latent-i za downstream zadatke; javne implementacije. | Pokrivaju alternative ASE-u; veći podatkovni/trening pipeline pre prvog A→B. |
| [MHC](https://github.com/osudrl/masked-humanoid-controller) | ECCV 2024 multimodal/partial-input humanoid controller, catch-up i promene pokreta; Isaac Gym kod. | Relevantan masked-control uzor; ProtoMotions ostaje glavni kandidat te porodice za praktičan test. |
| [Meta Motivo](https://github.com/facebookresearch/metamotivo) | Javni trening/evaluacija i šest pretrained HumEnv modela; reward/goal/tracking inference. | Važna alternativa klasičnom skill selector-u, ali drugo telo i nova obuka; [CC BY-NC 4.0](https://github.com/facebookresearch/metamotivo/blob/main/LICENSE) je dodatno ograničenje preuzimanja. |
| [HoST](https://github.com/InternRobotics/HoST) / [HumanUP](https://github.com/RunpeiDong/HumanUP) | Javni standing-up/get-up trening za robotske humanoide, uključujući G1; Isaac Gym. | Namenski izvori za fazu 3B. Ustajanje sa tla rešavaju direktnije od samog motion tracker-a; ne rešavaju kompletnu fazu 2. |
| [ASAP](https://github.com/LeCAR-Lab/ASAP) / [ExBody2](https://exbody2.github.io/) | Agile odnosno expressive whole-body tracking i sim-to-real postupci. | Relevantni za retargeting/tracker; transfer na fizički robot trenutno nije naš glavni problem. |
| [Holosoma](https://github.com/amazon-far/holosoma) | Aktuelni trening, retargeting i inference framework; velocity/tracking, IsaacSim/MJWarp, G1/T1. | Ozbiljna infrastrukturna alternativa; nema ovde potvrđenog kompletnog našeg task/recovery suite-a. |
| [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) | Aktuelni v1.1, MJX/MjWarp, JAX PPO/AMP/DeepMimic i biomehanički modeli. | Nije zastareo samo zato što je prvobitni rad stariji. Rezervna opcija za JAX/anatomiju, uz korisnikovu prednost MimicKit-u i novijim pravcima. |
| [MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc) | Model-predictive online optimizacija ponašanja u MuJoCo-u. | Pokriva i pristup bez naučene biblioteke latentnih veština; potencijalni baseline/teacher, bez dokaza naše prirodnosti i celog zadatka. |
| [PGMT](https://luyili.github.io/pgmt/) | Autorski prikaz perceptivnog general motion tracking-a i terrain/recovery prior-a. | Pratiti za fazu 3A; na pregledanoj stranici nije potvrđena kompletna javna implementacija. |
| [Humanoid Dodgeball](https://arxiv.org/abs/2610.40003), 2026-10-29 | Rad o threat-aware reward-u i smernim adversarial motion prior-ima za izbegavanje dolazećih objekata. | Direktno relevantan za 3C; javni training repo nije potvrđen ovim pregledom. |
| [Onboard Monocular Dodgeball](https://arxiv.org/abs/2611.01197), 2026-10-31 | Rad o proceni objekta iz slike, dodge/recovery i ponovnom nalaženju objekta. | Važan za buduće opažanje i ponovljene susrete; ne pretpostavljati dostupnost kompletnog koda. |

Ovo nije lista metoda koje treba sve implementirati. Za fazu 2 praktičan
izbor su MimicKit, ProtoMotions ili mjlab, uz PHC recovery analizu.
Ostali izvori popunjavaju određenu sposobnost ili daju alternativu ako se
pojavi konkretna prepreka.

## Da li su „skills” najbolji pristup?

Veštine su smislen način organizacije zadataka, ali nisu isto što i obavezni
ASE algoritam. Za fazu 2 treba prvo proveriti jednu komandovanu locomotion
politiku sa prior-om prirodnog kretanja. Njeni prelazi mogu biti naučeni unutar
iste politike. Latentna hijerarhija ima više smisla kada se ista biblioteka
pokreta koristi za mnogo zadataka; masked/generative politika kada su ciljevi
parcijalni i potrebna raznovrsna whole-body rešenja.

Za fazu 3 razumno je imati navigacioni sloj, detekciju pada i recovery režim,
te opažanje i izbor reakcije na objekat. Ti slojevi mogu deliti low-level
kontroler ili koristiti proverene specijaliste. To ostaje otvorena implementaciona
odluka. Treniranje bez odgovarajućih početnih stanja, kontakata i podataka neće
postati uspešno samo promenom naziva framework-a.

## Ograničeni praktični test pre velikih treninga

Ovo je sledeći predloženi eksperiment, nije pokrenut pregledom literature:

1. Pinovati verziju prvog kandidata i reprodukovati njegov mali zvanični demo
   u izdvojenom okruženju. Zabeležiti stvarni server GPU/VRAM, instalaciju,
   trajanje kompilacije i throughput; server specifikacije još nisu potvrđene.
2. Na radnoj kopiji anatomskog modela proveriti import, neutralnu pozu,
   forward kinematics, masu, ose/limite, kontakte i PD/action semantiku.
   Uporediti postojeći walking klip; odvojiti grešku retargetinga od RL greške.
3. Izabrati mali skup pokreta/stanja za hod, idle, start/stop i korektivni korak,
   ili proverljiv proceduralni/curriculum postupak. Jedan ciklus hoda i jedna
   statična poza nisu dokaz da podaci pokrivaju sve tranzicije.
4. Uporediti mali komandovani task sa baseline-om po fizičkim metrikama.
   Prvo prirodan hod, nulta brzina/stand i promena komande; zatim kompletan
   A→B stop/dwell i push suite. Izmeriti učenje i cenu, ne samo brzinu simulacije.
5. Ako prvi kandidat ima suštinsku blokadu importera ili nedostaje odgovarajući
   task/recovery postupak, sačuvati konkretan razlog i isti test preneti na
   najprikladniju alternativu. Ne trenirati tri kompletna sistema unapred.
6. Tek posle ovog testa odrediti veći budžet, seed-ove i server red. Nijedan
   framework nije ovim pregledom odabran kao već dokazan pobednik.

Za trenutni -X neuspeh dokazano je izostajanje dovoljne korekcije/koraka u
sačuvanim tragovima; konflikt sa statičnom imitacijom je hipoteza. Odsustvo
zasićenja momenata samo po sebi ne isključuje problem PD/mapiranja/geometrije.
Zato prethodna dva neuspešna recovery testa nisu razlog za još jedan identičan
dugi nastavak. Status završenog reda je u
[queue.json](../runs/stand_recovery_queue_20261112_2x2m_v2/queue.json).

## Preuzimanje koda i podataka

MimicKit i ProtoMotions objavljuju kod pod Apache-2.0, kao i mjlab;
Meta Motivo ima navedenu non-commercial licencu. Licence modela, težina i
motion podataka proveravaju se odvojeno od koda, posebno za AMASS/SMPL,
Reallusion i NVIDIA modele. Zadržati attribution i poreklo preuzetih delova.
Javno dostupan repozitorijum nije sam po sebi dozvola za svaku vrstu upotrebe.
