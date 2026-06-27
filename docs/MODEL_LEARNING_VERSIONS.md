# Model Learning Versions

Ovaj fajl je dnevnik glavnih iteracija ucenja. Poenta nije da svaka verzija ima
veci reward, nego da se zabelezi sta je stvarno nauceno i koji reward loophole je
otkriven.

## V0 - Berkeley/Barkley Baseline

**Ideja:** koristiti MuJoCo Playground Berkeley humanoid kao proof-of-concept za
joystick walking.

**Rezultat:** radi sa mnogo manje rucnog podesavanja.

**Zakljucak:** Berkeley nije dokaz da je custom generated human lak problem. To
je vec pripremljen locomotion benchmark sa dobrim modelom, kontaktima,
actuatorima, reward-om i PPO setupom.

## V1 - Generated Biomechanics Human, Initial Setup

**Ideja:** direktno trenirati generated human model.

**Rezultat:** mnogo treninga, malo korisnog hoda.

**Problem:** policy je imala premalo kontrole i previse losih lokalnih resenja:

- samo 12 leg actuatora,
- abdomen/pelvis prakticno zakljucani,
- reward je mogao da nagradi prezivljavanje/klizanje,
- bez gait phase signala,
- bez privileged critic signala.

**Zakljucak:** problem nije bio samo broj stepova. Task je bio lose postavljen za
PPO.

## V2 - 18 Actuators, Unlocked Stiff Trunk

**Ideja:** otkljucati abdomen i pelvis, ali ih drzati stiff kroz male action
scale-ove, jace PD limite i manji reset/random torque noise.

**Rezultat:** model postaje trenabilniji.

**Sta je dodato:**

- abdomen x/y/z actuators,
- pelvis x/y/z actuators,
- ukupno `nu=18`,
- privileged critic state,
- gait phase observation,
- action smoothing,
- veci PPO setup.

**Zakljucak:** ovo je bila potrebna infrastruktura. Bez toga generated human nije
imao dovoljno nacina da balansira.

## V3 - Forward Curriculum

**Ideja:** uciti prvo samo forward hod, bez punog joystick problema.

**Poznat dobar run:** forward run oko 90M stepova, reward oko `1400`.

**Rezultat:** policy ume da se krece napred i prezivi cele epizode.

**Problem:** forward-only policy ne zna pun joystick. Vizuelno i dalje moze da
izgleda zombi/klizavo.

**Zakljucak:** forward curriculum je koristan bootstrap, ali nije finalni cilj.

## V4 - Standard Joystick Fine-Tune

**Ideja:** nastaviti iz forward checkpoint-a na `command_profile=standard`.

**Rezultat:** reward oko `600+`, model reaguje na vise pravaca, ali kretanje i
dalje izgleda kao klizanje.

**Problem:** velocity tracking reward moze da se resi kontakt trikovima. Veci
reward nije automatski lepsi hod.

**Zakljucak:** pun joystick treba, ali ne sme biti jedini signal.

## V5 - Style V1 Reward

**Ideja:** pojacati stil hoda bez mocap-a:

- anti-slip/contact,
- swing foot clearance,
- torso/head posture,
- action smoothness,
- optional sine trajectory.

**Poznat run:** sine/reference trajectory run oko 60M, reward oko `715`.

**Rezultat:** malo bolje, ali nedovoljno. Hod i dalje nije profesor-ready.

**Problem:** proceduralna sine putanja daje ritam, ali ne garantuje dobar kontakt
i stvarne foot placements.

**Zakljucak:** sine reference je dobar debug alat, ali nije dovoljan kao finalni
style prior.

## V6 - BVH Reference Gait, Tier 1

**Ideja:** koristiti CMU/CGSpeed BVH walking clipove kao motion reference.

**Sta je dodato:**

- BVH parser/retargeting u `bvh_reference.py`,
- `--reference-gait bvh`,
- `--reference-gait-file` vise puta,
- `--reference-gait-list`,
- tier liste iz `BVH_walking_animation`:
  - tier1: vanilla forward walk,
  - tier2: lakse varijacije,
  - tier3: kompleksni/stilizovani hod,
  - uneven terrain/stairs posebno.

**Poznat run:**

```text
runs/biomechanics_noerfi_nodr_forward_ref_bvh_bvh_tier1_accurate_ppo_BiomechanicsHumanJoystickStandard_20260617_2118_60m_seed7_rew_1492p6453_best_1493p6858_s
```

**Rezultat:** reward oko `1493`, pune epizode, ali vizuelno i dalje kliza.

**Log signal:** pri kraju run-a:

- `episode_length=500`,
- `tracking ~= 490`,
- `ref_gait ~= 358`,
- `foot_slip ~= 386`,
- `swing_drag ~= 500`.

**Tumacenje:** policy je naucila da dobije visok reward, ali swing noga je cesto
ostajala u kontaktu, a stopala su klizila. Dakle BVH joint-angle tracking sam po
sebi nije dovoljan.

**Zakljucak:** potreban je jaci contact-aware anti-slip signal i kasnije bolji
mocap imitation koji prati stopala/root, ne samo joint uglove.

## V7 - Anti-Slip V2

**Ideja:** direktno zatvoriti reward loophole iz V6.

**Promena:**

- foot slip cost je sada kontakt-aware kvadrat brzine stopala,
- mala free-speed zona ignorise numericki jitter,
- slip penalty je znatno jaci,
- swing-foot drag penalty je znatno jaci,
- dodat je eksplicitni swing clearance deficit cost,
- swing clearance i stance contact reward su pojacani.

**Naknadna dijagnoza:** BVH tier1 model nije samo malo klizao; vizuelno skoro
uopste nije podizao noge. Zato anti-slip v2 sada posebno kaznjava situaciju gde
swing stopalo nije bar oko `8 cm` iznad stance stopala.

**Kriticni review nalaz:** ni ovo nije dovoljno ako policy ne vidi BVH target.
Prethodni BVH setup je nagradjivao pracenje random izabranog BVH clip/frame-a,
ali taj target nije bio u observation-u. To je skriven zadatak: policy ne zna
koji clip i frame treba da prati, pa uci prosecan klizavi kompromis.

## V8 - BVH Target-Conditioned Policy

**Ideja:** ukloniti skriveni BVH target iz reward-a.

**Promena:**

- Novi BVH trening dodaje trenutni reference target u policy observation.
- `state` observation za BVH target-conditioned run raste sa `98` na `116`.
- Command slice ostaje `9:12`, tako da joystick evaluator i dalje menja isti deo
  observation-a.
- Proceduralni gait/no-lift phase za BVH vise ne koristi nezavisni random sine
  clock, nego aktivni BVH phase.
- Swing noga za BVH reward se izvodi iz reference knee flexion-a, pa no-lift
  kazna vise ne moze da bude u konfliktu sa BVH frame-om.

**Ocekivanje:** ovo je prvi run u kome policy zaista zna koji BVH pose treba da
imitira. Stari BVH run-ovi nisu validan dokaz da motion imitation ne radi, jer
su bili delom non-Markov zadatak.

**Training note:** reward moze biti manji nego V6, ali ako hod izgleda bolje, to
je uspeh. Za ovaj projekat vizuelni gait kvalitet je vazniji od broja `1500`.

**Speed/debug note:** za prve provere koristi
`BVH_walking_animation/tier1_debug_10.txt`. To je mali subset od 10 vanilla walk
clipova, svi kraci od 5 sekundi, pa jedan 10s episode vidi ceo clip i loop.

**Foot contact experiment:** v16 XML uklanja dodati box/padding sole. Umesto toga
originalna generated foot capsule geometrija dobija ime `left_foot_sole` /
`right_foot_sole` i koristi se za kontakt/reward. Ako se pokaze gore, v15 box sole
moze da se vrati.

**Run result:** target-conditioned v16 debug run je poceo da dize noge, ali i
dalje pada. To je pomak: problem se prebacio sa no-lift/sliding na balans tokom
koraka.

**Next curriculum:** `forward_slow` sada prvo uci stabilan spor korak, pa tek
onda pun `forward`. Log iz resume run-a pokazao je da je `done_low` skoro uvek
glavni razlog pada, pa je sledeci rez balance-first:

- sporiji command `0.02-0.12 m/s`,
- 25% epizoda sa zero command radi stabilnog stajanja,
- manji velocity/progress reward za `forward_slow`,
- jaci base-height reward,
- jaci low-height cost,
- jaci fall penalty.

**Sledeci test:**

```bash
python train.py --device gpu --command-profile forward_slow --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 10000000 --num-envs 1024 --num-evals 5 --no-erfi --no-domain-randomization --no-checkpoints --run-tag bvh_target_obs_v16_balance_debug10
```

## Current Diagnosis

Najveci problem vise nije "ne ume da hoda". Ume da dobije reward i da prezivi.
Najveci problem je da reward jos uvek nije dovoljno vezan za fizicki uverljiv
kontakt stopala.

Kratko:

```text
velocity tracking + joint reference != natural walking
natural walking needs contact-aware foot constraints
```

## Next Research Direction

Najbolji sledeci pravac nije jos veci broj stepova, nego bolji imitation/contact
reward:

- contact-aware foot sliding penalty,
- stance foot should stay planted,
- swing foot should clear the ground,
- BVH/reference foot position tracking,
- root orientation/height tracking,
- optional behavioral cloning or motion prior later.

## V9 - Reward/Contact Sanity Fix

**Problem:** reward je mogao da bude visok dok video i dalje pokazuje klizanje.
Stari `clip(reward, 0, 3)` je gasio negativne per-step signale, pa su slip i
drag cesto samo smanjivali bonus umesto da stvarno kazne los kontakt.

**Promene:**

- Reward clip je promenjen na `[-5, 3]`.
- Fall penalty ostaje `-25`.
- `_foot_contact` sada koristi MuJoCo contact geom parove sa floor geom-om,
  umesto samo visinu stopala.
- `_minimum_geom_z` sada razlikuje box, sphere i capsule geometriju, pa v16
  capsule foot sole vise ne koristi box-corner pretpostavku.
- Dodan je `REFERENCES_USED.md` sa svim spoljnim referencama i lekcijama.

**Ocekivanje:** sledeci run moze imati manji reward nego stari sliding run, ali
treba da bude iskreniji. Ako reward opadne a video manje kliza, to je pobeda.

**Sud posle Unitree review-a:** nas trenutni BVH sistem je i dalje samo
lightweight pose reference. Unitree-style imitation trazi root/body pose,
orijentacije, brzine i end-effector tracking. To je sledeci ozbiljan korak ako
V9 ne resi klizanje dovoljno.

## V10 - Gymnasium + Unitree Signal Extraction

**Ideja:** izvuci korisne delove iz Gymnasium Humanoid i Unitree RL Mjlab bez
prepisivanja celog framework-a.

**Promene:**

- Privileged critic observation sada dobija Gymnasium-style fizicke signale:
  `cinert`, `cvel`, i `cfrc_ext`.
- Policy observation ostaje manji i cistiji; veliki physics signali idu samo
  critic-u.
- Dodata je mala Gymnasium-style contact-force kazna iz `cfrc_ext`.
- BVH loader sada pravi i `qvel_targets` iz retargetovanih poza.
- BVH reward sada ima `reference_velocity` pored `reference_gait`, sto je blize
  Unitree tracking ideji: ne prati se samo staticka poza nego i brzina pokreta.
- Train log sada prikazuje `ref_vel` i `contact_force`.

**Vazna kompatibilnost:**

- `state` policy obs ostaje `98` za obican run i `116` za BVH target-conditioned
  run.
- `privileged_state` critic obs raste, npr. sa `151` na `525` za obican run.
- Zato je ovo najbolje tretirati kao novi trening iz pocetka, ne kao resume
  starog checkpoint-a.

**Sta jos nedostaje za pravi Unitree-style imitation:**

- root/anchor position target,
- root/anchor orientation target,
- relative body positions,
- body orientations,
- body linear/angular velocities,
- explicit end-effector/foot position tracking.

## V11 - Unitree/Barkley-Style Anatomical Action Prior

**Problem iz run-a:** `gym_unitree_v10_debug10` se popravio sa negativnog reward-a
na oko `106`, ali `episode_length` je ostao oko `100` i `done_low` je prakticno
stalno aktivan. To znaci da policy dobija neke lokalne signale, ali telo i dalje
ima previse slobode da nadje nestabilne/glupe pokrete umesto stabilnog koraka.

**Ideja:** ne zakljucavati noge u animaciju, nego ograniciti prostor akcija kao
kod Unitree/Barkley-style modela:

- stride zglobovi smeju vise da rade: hip_x, knee_z, ankle_y,
- bocne/uvrtajuce ose smeju mnogo manje: hip_y, hip_z, ankle_z,
- abdomen/pelvis ostaju kontrolisani, ali mali i kruti,
- joint range u XML-u vise ne dozvoljava ekstremne generated-model polozaje.

**Promene:**

- Novi XML cache version: `trainfast_v17`.
- Dodan `LEG_ACTION_SCALE` u `biomechanics_env.py`.
- Action scale vise nije uniforman za noge:
  - hip_x `0.35`,
  - knee_z `0.55`,
  - ankle_y `0.24`,
  - hip_y `0.12`,
  - hip_z `0.14`,
  - ankle_z `0.08`.
- `--action-scale` i dalje radi kao globalni multiplier preko ovih odnosa.
- Dodan variable posture prior:
  - dok stoji, svi zglobovi imaju stroge tolerancije,
  - dok hoda, hip/knee/ankle pitch dobijaju veci prostor,
  - trunk i lateral/twist ose ostaju stroze.
- Train log sada prikazuje `var_pose`.
- U `biomechanics_model.py` dodan `LEG_JOINT_SPECS`, pa v17 XML ima strozije
  humanoidne joint range vrednosti.

**Provera:**

- `python -m py_compile biomechanics_model.py biomechanics_env.py train.py`
  prosao.
- Env je generisao i ucitao
  `generated_models/human_male_180cm_75kg_standard_trainfast_v17.xml`.
- JAX reset/step smoke test je prosao, ali je prvi compile bio spor.

**Vazna kompatibilnost:**

- Ovo tretirati kao trening iz pocetka.
- Ne bih resume-ovao stare v16/v10 checkpoint-eve, jer su dinamika, action prior
  i XML joint range promenjeni.

**Sledeci test koji ima smisla:**

```bash
python train.py --device gpu --command-profile forward_slow --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 20000000 --num-envs 1024 --num-evals 10 --no-erfi --no-domain-randomization --run-tag unitree_prior_v11_debug10
```

Ako do `5-10M` stepova `episode_length` ne krene jasno iznad `200`, problem je
jos uvek u stabilnosti/initial pose/contact dinamici, ne u broju trening koraka.

## V12 - DeepMimic/LocoMuJoCo/GMR Reference Review

**Problem:** BVH pipeline je krenuo u dobrom smeru, ali je i dalje bio preslab u
odnosu na ozbiljne imitation sisteme. Nas signal je bio blizu "prati nekoliko
leg joint uglova", dok DeepMimic/LocoMuJoCo-style pristup prati root, brzine,
end-effectors/sites i cesto resetuje epizodu blizu reference state-a.

**Repo-i pregledani:**

- DRLoco,
- DeepMimic_mujoco,
- LocoMuJoCo,
- GMR.

**Sta je nauceno:**

- Pravi imitation reward nije samo joint pose.
- Treba root target: visina, orijentacija i/ili brzina.
- Treba end-effector target: stopala u odnosu na root/trunk.
- Reference frame/phase mora biti eksplicitan.
- Reference-state initialization moze mnogo da pomogne, ali je rizican flag i
  treba ga testirati odvojeno.
- GMR je najbolji dugorocni BVH retargeting put, ali zahteva poseban integration
  sloj za nas model ili izbor podrzanog robot modela.

**Promene u nasem kodu:**

- BVH sada cuva approximate `root_height_offsets`.
- BVH sada cuva approximate `root_forward_velocity_factors`.
- Env precomputuje BVH qpos/qvel targete.
- Env precomputuje trunk-relative left/right foot targete preko MuJoCo FK.
- Reward sada ima `ref_foot`.
- Reward sada ima `ref_root`.
- Dodati su opcioni `reference_phase_randomization` i `reference_state_init`.
- `reference_state_init` je fixovan da ne pocne prvi step sa pogresnim
  `last_action`.
- Dodan je `reference_sanity.py`.

**Current run status:** V10 forward resume nije propao. Do sada je stigao do:

```text
step=53,739,520
reward=960.2457
episode_length=457.5625
done_low=0.312
checkpoint=000053739520
```

**Sledeci test:** ne pustati sve novo odjednom. Prvi gate treba da testira samo
novi root/foot reward signal iz kompatibilnog V10 checkpoint-a:

```bash
python train.py --device gpu --command-profile forward --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --resume-from runs/biomechanics_noerfi_nodr_forward_ref_bvh_v10_18350_resume_forward_accurate_ppo_BiomechanicsHumanJoystickStandard_20260621_0254_80m_seed7_running/checkpoints/000053739520 --xml-path generated_models/human_male_180cm_75kg_standard_trainfast_v16.xml --legacy-action-prior --timesteps 8000000 --num-envs 1024 --num-evals 4 --num-eval-envs 16 --no-erfi --no-domain-randomization --run-tag v10_ref_root_foot_gate
```

**Ne ukljucivati u prvom gate-u:**

- `--reference-state-init`,
- `--reference-phase-randomization`,
- full tier1 list,
- ERFI/domain randomization.

**Zakljucak:** ovo je najbolji sledeci korak ako ne krecemo iz pocetka. Ako gate
ne pomogne, sledeci veliki korak je GMR-style retargeting, a ne jos jedan
nasumicni 80M PPO run.
