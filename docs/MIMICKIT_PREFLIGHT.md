# MimicKit: provera anatomije i walking reference

Izvršeno 2026-11-12 na grani
`navigation_walking_with_anatomical_human_robot` (preimenovana skills grana).
Ovo je prvi implementirani korak iz [poređenja pristupa](FRAMEWORK_COMPARISON_2026-11-12.md),
za [fazu 2](PROJECT_GOALS.md). Kinematika i prenos reference prolaze;
kompatibilnost simulatora, kontrole i politika još nije dokazana.

Dopuna 2026-11-13: implementiran je i proveren prenos fizičkih control targets
u momente u native MuJoCo-u. To daje numerički oracle za sledeći engine test;
Newton/MimicKit simulator još nije prošao isti test.

## Implementacija

- [mimickit_adapter.py](../skills/mimickit_adapter.py) gradi referentni skelet
  preko stvarnog upstream `KinCharModel`, uz eksplicitno mapiranje MuJoCo
  koordinata. Serijske hinge zglobove predstavlja virtuelnim kinematičkim
  čvorovima; fizički XML, mase i aktuatori se ne menjaju. Čuva svih 33 hinge
  koordinata, uključujući sedam pasivnih; fizički model ima 16 tela i 26 aktuatora.
- [check_mimickit_model.py](../tools/check_mimickit_model.py) poredi položaj i
  orijentaciju svakog fizičkog tela sa nezavisnim native MuJoCo `mj_forward`.
  Proverava i nasumične root orijentacije, zglobne opsege i sačuvane replay poze.
  Izveštaj beleži originalni redosled aktuatora, limite i gear.
- [export_navigation_reference.py](../tools/export_navigation_reference.py)
  koristi postojeći environment, arhivirani config/XML/BVH i pripremljene
  reference. Time zadržava postojeći retargeting, prior, poravnanje poda i
  pripremu ciklusa. Izvozi referentne frejmove, ne ponašanje naučene politike.
- [convert_mimickit_motion.py](../tools/convert_mimickit_motion.py) zapisuje
  upstream motion format i učitava ga stvarnim `MotionLib`. Poredi uzorkovanje
  na vremenima izvornih frejmova i proverava konačnost brzina.

Adapter je namerno ograničen na jedan free root i hinge/fixed potomke.
Nepodržani slider/ball zglobovi i nenulti hinge anchors daju jasnu grešku.
Kompajlirani MuJoCo transforms obuhvataju `euler`, defaults i `ref` uglove.

Upstream je pinovan na
[`fbe1a36e878f036d6bf5521defb8eb3b8ab4170d`](https://github.com/xbpeng/MimicKit/tree/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d).
Alat odbija drugi commit ili izmenjen checkout. U ovom commitu originalni
MJCF importer pada sa `Series joints are not supported.` na našem XML-u.
Njegovo pretvaranje tri hinge zgloba u spherical nije zamena za očuvanje
naših koordinata. Zato je dodat odvojen kinematički adapter.

## Izmereni rezultati

CPU probe: Python 3.12, MuJoCo 3.10.0, Torch 2.9.1+cpu, NumPy 2.3.5.
Prag FK testa je `1e-5 m` i `1e-5 rad` po telu.

| Provera | Obim | Najveće odstupanje položaja | Rezultat |
| --- | --- | --- | --- |
| Native MuJoCo naspram MimicKit FK | Neutralna + 1.024 nasumične poze, seed 11 | 2,40 µm | Prolaz |
| Stvarni -X replay iz prethodnog recovery testa | 70 sačuvanih poza | 1,56 µm | Prolaz kinematike; sama epizoda je neuspešan recovery |
| Izvezeni walking klip | 134 izvorna frejma | 1,72 µm | Prolaz FK |
| Izvorni `MotionLib` | 133 vremena frejmova | 1,58 mm | Pad testa očuvanja frejmova |
| `MotionLib` sa eksplicitnom SLERP ispravkom | Ista 133 vremena i isti motion podaci | 3,11 µm | Prolaz |

Trajni izveštaji:

- [kinematika i actuator mapa](../analysis_outputs/skills/mimickit_kinematics_mj310_20261112.json);
- [originalna interpolacija](../analysis_outputs/skills/mimickit_motion_upstream_20261112.json);
- [ispravljena interpolacija](../analysis_outputs/skills/mimickit_motion_corrected_20261112.json).

Uzrok razlike je upstream `util.torch_util.slerp`: za veoma bliske rotacije
vraća konstantnu sredinu čak i kada je fraction nula. Lokalna funkcija
`interpolate_quaternions` koristi ponderisanu, normalizovanu interpolaciju
za mali ugao i shortest-arc SLERP inače. Konverter je uključuje isključivo
uz `--correct-small-angle-slerp`, u svom procesu; upstream checkout ostaje čist.
Obe varijante proizvode isti motion SHA256
`1fc8e4fdfd6339b8043baf7480ed48330aadd0bfedc1cb20de2278e0a908979d`.

Testovi adaptera uključuju rotirane lokalne ose, pasivan zglob, `ref` offsets,
fixed potomka, quaternion znak/skalu, nevažeća stanja i namerno pogrešan
transform koji oracle mora da prepozna. Ukupno prolazi 13 pytest slučajeva,
uz 12 postojećih queue/overnight regresionih testova posle promene naziva grane.

Lokalni paketi su u `runs/navigation_mimickit_preflight_20261112/`:
`reference_v2.npz`, `motion_upstream_v2/` i `motion_corrected/`.
Početna FK provera sa MuJoCo 3.8.1 sačuvana je kao
`kinematics_preliminary_mj381.json`; gornji kanonski rezultat je sa 3.10.0.
Izveštaji eksplicitno beleže da je exporter pokrenut iz radnog stabla tokom
implementacije (`working_tree_dirty=true`), parent commit i hash tadašnjeg
exportera. Naknadno su uklonjeni samo suvišni lint komentari u exporteru.

## Reprodukcija

Koristiti izdvojeni CPU environment; ne menjati Brax/JAX trening okruženje.
Prva instalacija, iz root-a repoa:

```powershell
python -m venv .venv-mimickit-probe
./.venv-mimickit-probe/Scripts/python.exe -m pip install -r requirements-mimickit-probe.txt
git clone --depth 1 https://github.com/xbpeng/MimicKit.git .tmp_mimickit_navigation
git -C .tmp_mimickit_navigation fetch --depth 1 origin fbe1a36e878f036d6bf5521defb8eb3b8ab4170d
git -C .tmp_mimickit_navigation checkout --detach fbe1a36e878f036d6bf5521defb8eb3b8ab4170d
```

Probe zavisnosti su pinovane odvojeno u
[requirements-mimickit-probe.txt](../requirements-mimickit-probe.txt).
To nije instalacija kompletnog GPU training stack-a. Sledeće komande koriste
nov izlazni folder; ponavljanje zahteva drugi naziv da sačuva prethodne dokaze.

```powershell
./.venv-mimickit-probe/Scripts/python.exe tools/check_mimickit_model.py --mimickit .tmp_mimickit_navigation --samples 1024 --output runs/navigation_mimickit_repro/kinematics.json
./.venv-win/Scripts/python.exe tools/export_navigation_reference.py --output runs/navigation_mimickit_repro/reference.npz
./.venv-mimickit-probe/Scripts/python.exe tools/convert_mimickit_motion.py --mimickit .tmp_mimickit_navigation --reference runs/navigation_mimickit_repro/reference.npz --correct-small-angle-slerp --output-dir runs/navigation_mimickit_repro/motion_corrected
./.venv-mimickit-probe/Scripts/python.exe -m pytest tests/mimickit -q
./.venv-win/Scripts/python.exe -m unittest tests.test_stand_recovery_queue tests.test_stand_overnight -q
```

Za isti replay dodati FK komandi
`--replay runs/stand_recovery_control_diagnostic_20261112_broad/000000808960/push_x_neg_replay.npz`
ako je lokalni paket prisutan. Za reprodukciju upstream greške pokrenuti
konverter bez `--correct-small-angle-slerp`, sa zasebnim izlaznim folderom;
očekivani exit code je 1 i čuva se neuspešan report. Pytest preskače testove
koji zahtevaju upstream kada checkout nije dostupan; preskok nije prolaz.

## Sledeće obavezne provere pre treninga

1. **Dinamika i kontrola.** Proveriti originalnih 26 position aktuatora,
   njihov gear/gain/bias, clipping i raspored, kao i sedam pasivnih zglobova,
   opruge, damping, limite i kontakte. Upstream Newton `_build_controls`
   uzima PD gains iz `dof_passive_stiffness/damping`, pa nulira te pasivne
   koeficijente. To nije dokaz ekvivalentnosti našem modelu koji odvojeno ima
   position actuators i pasivne sile. Prvo uporediti sile i kratke odzive pod
   istim ciljevima/stanjima, zatim rollout. Ovaj adapter ne rešava action prostor.
2. **Kontinuitet reference.** Proveriti vremena između frejmova, root pomeraj,
   prelaz ciklusa i brzine naspram sadašnje Hermite reference. Poslednji frejm
   wrap ciklusa izuzet je iz MotionLib poređenja jer pripada sledećem ciklusu.
   Konačne brzine nisu dokaz njihovog poklapanja. Pasivni uglovi u izvezenoj
   referenci ostaju u postojećoj početnoj pozi; to ne zamrzava pasivnu fiziku.
3. **Mali fizički zadatak.** Reprodukovati zvanični demo u izabranom engine-u,
   izmeriti throughput/memoriju, zatim naš kratki hod/stand/promenu komande.
   Proveriti podatke za idle/start/stop i korektivni korak. Tek onda odrediti
   veći trening budžet i uključiti nezavisni A→B stop/dwell i push suite.

Lokalno je potvrđen RTX 2060 sa 6.144 MiB kroz WSL. Korisnik je 2026-11-13
potvrdio da server još nije povezan; lokalne provere se nastavljaju. Nema
pokrenutog novog RL treninga niti server procesa. Stari recovery redovi su završeni i ne ponavljaju
se. Natural-walk arhiva, njena grana i joystick grana ostaju nepromenjene.

## Izvršena kontrolna provera — 2026-11-13

[mujoco_control.py](../skills/mujoco_control.py) prevodi originalnih 26 fizičkih
control targets u aktuatorske sile i generalizovane momente. Čuva gain/bias,
gear, control clipping pre force clipping-a i sabiranje više aktuatora na
istom zglobu. Formula prati
[MuJoCo actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model).
Pasivne sile ostaju u simulatoru; ne dodaju se drugi put i pasivni zglobovi ne
dobijaju motor. Ovo još ne prevodi normalizovane akcije postojeće Brax politike.

[check_navigation_control.py](../tools/check_navigation_control.py) izvršio je:

| Test | Rezultat |
| --- | --- |
| 1.025 nasumičnih stanja, targets unutar i izvan control limita | Maksimalna razlika prema native `actuator_force`/`qfrc_actuator` je 0 u ovom izvršavanju. |
| Saturacija sile | Obuhvaćeno 4.412 zasićenih aktuatorskih uzoraka. |
| Pasivni zglobovi | Svih sedam ostaju bez aktuatorskog momenta; njihove native pasivne sile dostižu 62,99 Nm u ovoj širokoj nasumičnoj raspodeli. To nije izmereni nominalni walking moment. |
| Tri početne poze, po 200 koraka / 1 s, kontakti prisutni u svih 600 koraka | Originalni position aktuatori i eksplicitni momenti ažurirani svakih 5 ms daju jednake `qpos` i `qvel` u ovom izvršavanju. |
| Kontrolni primer: moment se osvežava tek svakih 20 ms | Razlika najveće komponente `qpos` je 0,0259–0,0497. `qpos` meša metre, quaternion komponente i radijane; ove brojeve ne tumačiti kao centimetre. |

Trajni rezultat i kompletna gain/bias/passive mapa su u
[navigation_control_20261113.json](../analysis_outputs/skills/navigation_control_20261113.json).
Devet novih pytest slučajeva proverava non-unit i negativni gear, `ref`,
velocity bias, oba clipping-a, više motora na istom zglobu, pasivni zglob,
nevažeće targets i odbijanje nepodržane activation dinamike.

Za native rollout kopira se model u memoriji, uklanja samo aktuatorska
gain/bias sila i dodaje isti izračunati moment preko `qfrc_applied`. Originalni
kontakti, mase, ograničenja i pasivne sile ostaju isti. Probe zahteva nulti
aktuatorski velocity bias kao u našem XML-u: inače bi implicitna integracija
njegovog izvoda tražila dodatnu proveru. Ovo su kratke open-loop putanje;
nisu test stabilnog hoda, stajanja ili nove politike.

Praktična posledica: svih 26 aktuatora ima nenulti gain, ali **20 njihovih
zglobova ima nultu pasivnu stiffness**. Direktno korišćenje tih stiffness
vrednosti za motorski Kp ne čuva našu kontrolu. Pored toga, pinovani Newton
importer uključuje `convert_3d_hinge_to_ball_joints=True`. Pre sledećeg engine
testa moraju se eksplicitno očuvati serijske hinge koordinate, aktivni i
pasivni zglobovi, i ponoviti iste force/rollout provere. Potrebna je i provera
mapiranja 34 referentna čvora na 16 fizičkih tela; virtuelni čvorovi nisu tela.

Reprodukcija sa novim report nazivom:

```powershell
./.venv-mimickit-probe/Scripts/python.exe tools/check_navigation_control.py --reference runs/navigation_mimickit_preflight_20261112/reference_v2.npz --samples 1024 --steps 200 --output runs/navigation_control_repro/report.json
./.venv-mimickit-probe/Scripts/python.exe -m pytest tests/mimickit/test_control_bridge.py -q
```

Ako reference paket nije prisutan, prvo ga napraviti exporterom iz prethodnog
odeljka i proslediti novu putanju. Bez `--reference` izvršavaju se random force
provera i neutralna putanja; dve walking početne poze tada nisu proverene.

## Newton/MuJoCo-Warp fizički gate — 2026-11-13

Pinovani GPU probe koristi Newton `1.0.0`, `newton-actuators 0.1.0`, Warp
`1.12.0`, MuJoCo `3.5.0` i MuJoCo-Warp `3.5.0.2`. Newton 1.0 README navodi
Warp 1.12; noviji Warp 1.19 ne radi sa ovim izdanjem (`warp.config.verbose`
nedostaje), zato je environment odvojen i verzije su eksplicitne u
[requirements-mimickit-newton-probe.txt](../requirements-mimickit-newton-probe.txt).

[check_mimickit_newton_engine.py](../tools/check_mimickit_newton_engine.py)
uvozi neizmenjeni arhivirani XML sa sledećim obaveznim izborima:

- `convert_3d_hinge_to_ball_joints=False`, čime ostaju `nq=40`, `nv=39` i svih
  33 hinge koordinata;
- `ctrl_direct=True`, čime ostaje originalni red i fizika 26 MuJoCo position
  aktuatora;
- `enable_self_collisions=True`, jer je pod u istom MJCF-u kao robot, dok sam
  XML već isključuje neželjene parove. `False` bi uklonio i pod–đon kontakte;
- `impratio=1`, kao u izvornom modelu, umesto MimicKit hardkodovanih 10.

Newton 1.0 pogrešno tretira MuJoCo `damping` i `stiffness` pod
`compiler angle="degree"` kao per-degree vrednosti. Sirovi import zato odstupa
do 675,55 za damping i 3.940,70 za stiffness. Runtime adapter vraća kompajlirane
MuJoCo koeficijente pre pravljenja solvera i zadržava pozitivni izvorni
`jnt_solref=(0.02, 1)`. Arhivirani XML se ne menja.

Gate od 200 fizičkih koraka / 1 s prolazi na RTX 2060:

| Provera | Rezultat |
| --- | --- |
| Tela/zglobovi/koordinate/aktuatori | `17/34/40/39/26` u izvornom i izvedenom modelu. |
| Kontakti | 2–4 u oba engine-a, prisutni u svih 200 koraka. |
| CPU MuJoCo oracle | Maks. `qpos` greška `6,23e-7`, `qvel` `4,46e-5`. |
| RTX MuJoCo-Warp | Maks. `qpos` greška `2,81e-6`, `qvel` `1,99e-4`. |
| Fizički parametri | Damping, stiffness, solref i actuator gain/bias/gear/force range prolaze tolerancije. |

Kompletan rezultat je u
[mimickit_newton_engine_20261113.json](../analysis_outputs/skills/mimickit_newton_engine_20261113.json).
Solver izostavlja 14 collision-disabled vizuelnih geometrija (`ngeom=3` umesto
17), ali čuva tela, mase i inercije. Imena aktuatora se gube; njihov red i svi
fizički nizovi ostaju isti.

Izvršen je i veći, namerno stroži test. Najbolji sačuvani stand checkpoint
`409600` prošao je izvorni zatvoreni MJX rollout od 500 kontrolnih koraka / 10 s.
[export_stand_policy_trace.py](../tools/export_stand_policy_trace.py) sačuvao je
stvarno primenjene targets, a
[check_stand_policy_trace_newton.py](../tools/check_stand_policy_trace_newton.py)
ih je ponovio open-loop u native MuJoCo-u i Newtonu. Prvih 1 s ostaje blisko
(`qpos 5,52e-6`, `qvel 2,44e-4`), ali oba open-loop replay-a zatim padaju jer
više ne dobijaju feedback aktora. To nije Newton-only kvar: native MuJoCo se
od izvornog MJX traga raziđe do 2,90 u mešovitim `qpos` komponentama. Neuspešan,
ali informativan rezultat je u
[mimickit_newton_policy_trace_20261113.json](../analysis_outputs/skills/mimickit_newton_policy_trace_20261113.json).

Sledeći obavezni korak nije veći slepi trening. Potrebna je zatvorena petlja:
Newton stanje → postojeći observation builder i `last_action`/command stanje →
zamrznuti actor → postojeća action smoothing/residual mapa → četiri fizička
podkoraka. Tek kada neutralni stand i definisani -X push prođu u toj petlji,
ima smisla meriti multi-world throughput i pustiti trening na serveru.

Reprodukcija kratkog engine gate-a:

```powershell
python -m venv .venv-mimickit-newton
./.venv-mimickit-newton/Scripts/python.exe -m pip install -r requirements-mimickit-newton-probe.txt
$env:WARP_CACHE_PATH = (Join-Path $PWD '.warp-cache-newton')
./.venv-mimickit-newton/Scripts/python.exe tools/check_mimickit_newton_engine.py --steps 200 --output runs/navigation_mimickit_newton_repro/engine.json
```

## Zatvorena petlja i stand-recovery gate — 2026-11-13

[stand_policy_controller.py](../tools/stand_policy_controller.py) drži zamrznuti
JAX actor u odvojenom procesu, a
[evaluate_newton_stand_closed_loop.py](../tools/evaluate_newton_stand_closed_loop.py)
izvršava istu politiku nad native MuJoCo ili Newton/MuJoCo-Warp fizikom. Kontrola
ostaje na 20 ms, sa četiri fizička podkoraka od 5 ms i postojećim smoothingom,
action mapom i controller `info` stanjem.

Prenos samo `qpos/qvel` nije dovoljan. `mjx.step` i `mj_step` vraćaju konačne
koordinate, ali izvedena body polja ostaju vremenski poravnata sa stanjem pre
poslednje integracije. Ponovni `forward` je zato menjao actor observation već u
prvom koraku do `0,0604`, a u prvih deset koraka do `0,1257`. Continuous
reference dodatno čita `xquat` direktno. Konačni protokol prenosi `qpos`, `qvel`,
`xpos`, `xquat`, `xmat` i `cvel`; na direktnom MJX oracle-u reprodukuje actor
observation sa maksimalnom greškom `0` u proveri. Nezavisni 10-koračni trace
reprodukuje stvarne actuator targets sa maksimalnom greškom `5,96e-8`; nijedan
korak ne prelazi toleranciju `1e-6`.

Sa tim egzaktnim mostom dobijeni su sledeći rezultati za checkpoint `409600`:

| Gate | Rezultat |
| --- | --- |
| Native MuJoCo 3.10, neutralno 60 s | Pad na 37,46 s; najduži dwell 36,24 s. |
| Newton 1.0 / MuJoCo-Warp 3.5.0.2, neutralno 60 s | Pad na 37,44 s; najduži dwell 36,22 s. |
| Newton, −X potisak 5% BW tokom 0,2 s | Pad na 6,84 s; nema recovery-a. |
| Newton, +X potisak istog intenziteta | Prolaz 20 s; recovery za 1,0 s; drift 7,8 cm. |

Poklapanje native/Newton terminala unutar 20 ms pokazuje da GPU bridge nije
uzrok dugoročne nestabilnosti. Sačuvani checkpoint je uzak van svog trening MJX
backenda i izrazito smerno asimetričan. Zbog toga veliki navigation trening još
nije pokrenut. Sledeći trening mora da koristi ciljni backend ili physics
randomization i prvo da prođe neutralnih 60 s i bidirekcioni push suite. Sažet
trajni report je u
[mimickit_newton_closed_loop_20261113.json](../analysis_outputs/skills/mimickit_newton_closed_loop_20261113.json).
