# Berkeley vs generated biomechanics walking

Ovaj dokument je kratak teorijski rezime trenutnog stanja projekta: zasto je
Berkeley humanoid radio mnogo lakse, zasto generated biomechanics human teze
dolazi do lepog hoda, i sta je razuman sledeci korak bez beskonacnog
produbljivanja.

## Trenutni zakljucak

Trenutni generated human rezultat je dobar kao proof-of-concept:

- model se generise kao MuJoCo/MJX humanoid,
- PPO policy se trenira,
- checkpoint moze da se nastavi preko `--resume-from`,
- forward walking je naucen,
- standard joystick fine-tune daje kretanje u vise pravaca,
- model nije "mrtav" i ne krece od nule.

Ali trenutni rezultat nije jos dokaz prirodnog humanoidnog hoda. Video pokazuje
da policy ume da dobije reward, ali cesto koristi kontaktni trik: telo se krece
uz klizanje stopala i neprirodan nagib, umesto jasnog swing/stance ciklusa.

To je normalna RL pojava: ako reward ne kaze dovoljno jasno "ovo mora da lici na
hod", policy ce naci najjeftiniji fizicki nacin da dobije brzinu i stabilnost.

## Zasto je Berkeley radio toliko lako

Berkeley/Barkley rezultat nije bio samo "isti problem sa manje truda". To je bio
mnogo pripremljeniji problem.

### 1. Berkeley model je vec dizajniran za locomotion

Berkeley humanoid iz MuJoCo Playground-a je model koji je vec napravljen da bude
dobar RL benchmark:

- stabilne mase i inercije,
- razumni joint limits,
- dobro podeseni actuatori,
- kontaktne geometrije koje ne prave previse cudne lokalne optimume,
- reset poza koja je pogodna za hod,
- reward i observation dizajnirani za taj model.

Generated biomechanics human nije isto to. On je anatomski zanimljiviji, ali nije
automatski "learning friendly". Ima vise mesta gde policy moze da nadje los trik:
glava/pasivni zglobovi, stopala, padding, odnos mase i momenta, visina root-a,
kontakt sa podom, slabiji ili cudno skalirani actuatori.

### 2. Berkeley env vec ima dobro namesten task

Kod Berkeley prototipa si koristio env koji vec dolazi sa pripremljenim
locomotion setupom:

- joystick command je vec deo zadatka,
- PPO konfiguracija je vec tuned,
- observation prostor je smislen,
- action smoothing i PD targeti su vec u dobrom opsegu,
- reward je vec uskladjen sa tim telom,
- termination uslovi su vec provereni.

Kod generated human-a mi smo morali sve to da napravimo sami. To znaci da svaki
detalj moze da bude pogresan za 10 posto, a tih 10 posto se u RL-u brzo pretvori
u "naucio je cudan hod".

### 3. Berkeley je resavao laksi fizicki problem

Berkeley humanoid je vise roboticki benchmark. Takvi modeli cesto imaju:

- krace i cistije kontaktne lance,
- manje problematicne pasivne delove,
- jasnije actuatorske poluge,
- manje anatomskih izuzetaka,
- bolji odnos snage i mase.

Generated human je blizi stvarnom telu, ali to ne znaci da je laksi za RL. Cesto
je tezi bas zato sto fizika dozvoljava vise neprijatnih polu-resenja.

### 4. Reward-only hod prirodno pravi "zombie walk"

Ako reward kaze:

- budi ziv,
- idi trazenom brzinom,
- ne trosi previse akcije,
- drzi visinu,

onda policy ne mora da nauci lep ljudski hod. Mora samo da maksimizuje brojeve.
Ako moze da klizi, gura pelvis, savije torzo i ipak dobije tracking reward, on ce
to uraditi.

Berkeley model verovatno ima manje takvih rupa, pa isti tip reward-a izgleda
mnogo bolje.

## Sta trenutni generated model stvarno dokazuje

Trenutni rezultat dokazuje ovo:

- pipeline radi od generated modela do PPO treninga,
- 18-actuator setup je trenabilan,
- forward pretraining ima smisla,
- joystick fine-tune nije nemoguc,
- checkpoint transfer radi,
- reward signal je dovoljno informativan da policy nauci kretanje.

Ne dokazuje jos:

- prirodan ljudski gait,
- stabilan hod bez klizanja,
- kvalitetan lateralni/backward joystick,
- robustnost na perturbacije,
- profesor-ready animaciju.

Ovo je vazna razlika. Rezultat nije neuspeh, ali ga ne treba prodati kao finalni
realistican hod.

## Sta dalje

Postoje tri realna pravca.

## Opcija A: zaustaviti produbljivanje i spakovati proof-of-concept

Ovo je najrazumnije ako je cilj da projekat bude zavrsiv.

U tom slucaju bih prikazao:

- Berkeley legacy demo kao "cist, stabilan joystick walking baseline",
- generated human demo kao "custom generated biomechanics model naucen PPO-om",
- jasno ogranicenje: generated gait jos ima contact/style artefakte,
- sledeci rad: motion imitation ili contact-aware gait reward.

Ovo je dobar akademski narativ jer ne krije problem. Pokazuje da si uspeo da
napravis sistem, a ne tvrdi da si resio humanoid locomotion do kraja.

## Opcija B: proceduralni gait/style reward

Ovo je najbolji sledeci korak ako zelis da generated model izgleda bolje, ali ne
zelis odmah mocap.

Ideja: zadrzati joystick reward, ali dodati jace style signale:

- swing foot mora stvarno da se podigne,
- stance foot ne sme da kliza,
- jedna noga je uglavnom stance dok druga ide swing,
- torzo ne sme da pada napred,
- head/upper body treba da ostanu pasivno stabilni,
- action jerk/action rate treba da bude manji,
- pelvis/root ne sme da "vuce" telo dok stopala stoje zalepljena za pod.

Prednost:

- relativno brzo za implementaciju,
- ne treba dataset,
- dobro resava klizanje.

Mana:

- i dalje nije pravi ljudski hod,
- moze da izgleda "roboticki",
- treba pazljivo balansirati reward.

## Opcija C: reference gait ili mocap imitation

Ovo je najbolji pravac ako je cilj profesor-ready, prirodniji hod.

Postoje dve verzije.

### Proceduralni reference gait

Ovo znaci da ne koristis pravi mocap, nego zadatu fazu hoda:

- leva/desna noga imaju sinusoidalnu fazu,
- swing noga ima ciljnu visinu,
- stance noga ima kontakt,
- kuk/koleno/skocni zglob imaju ocekivani obrazac.

To je lakse od mocap-a i moze dosta da smanji zombie/sliding hod.

### Mocap motion imitation

Ovo znaci da imas referentni snimak ili dataset ljudskog hoda i policy dobija
nagradu da lici na taj pokret.

Tipican imitation reward:

- joint pose error,
- joint velocity error,
- end-effector/foot position error,
- root orientation,
- phase matching,
- pose prior/style term.

Prednost:

- najbolja sansa za prirodan hod,
- manje reward hack-ova,
- jasniji cilj: "hoda kao ovaj primer".

Mana:

- treba retargeting na tvoj skeleton,
- treba uskladiti mocap zglobove sa MuJoCo joint imenima,
- treba faza hoda i reset oko reference,
- ako je imitation prejak, policy nece dobro slusati joystick.

Najbolja verzija nije cist mocap i nije cist joystick. Najbolja verzija je
kombinacija:

```text
total_reward =
    joystick_tracking
  + alive/upright/base_height
  + contact/anti-slip
  + motion_style_or_reference_gait
  - action/action_rate costs
```

Drugim recima: joystick kaze gde treba da ide, reference/style kaze kako treba
da izgleda dok ide tamo.

## Moja preporuka

Ne bih sada jurio jos veci reward na istom setupu. Vec si video da veci reward ne
znaci nuzno lep hod.

Najbolji prakticni plan:

1. Sacuvati trenutni best generated checkpoint kao proof-of-concept.
2. Sacuvati Berkeley/Barkley legacy demo kao stabilan baseline.
3. Za finalnu prezentaciju jasno razdvojiti:
   - "baseline koji lepo hoda",
   - "generated model koji je uspesno treniran, ali jos ima gait artefakte".
4. Ako nastavljas razvoj, sledeci ozbiljan korak je hybrid style/imitation:
   - prvo proceduralni gait/style reward,
   - zatim eventualno mocap imitation ako imas vremena.

Ako je cilj da se projekat zavrsi i pokaze profesoru, ne bih odmah ulazio u pun
mocap pipeline. To je novi projekat u projektu.

Ako je cilj da generated human stvarno izgleda dobro, onda mocap/reference gait
ili barem jak proceduralni style reward postaje neizbezan. Reward-only velocity
tracking verovatno nece sam od sebe dati prirodan hod.

## Sta ne bih radio sada

Ne bih sada prvo:

- samo dodavao jos actuatora,
- samo otkljucavao jos kicme,
- samo pustao jos 100M stepova,
- samo pojacavao randomization/ERFI,
- samo jurio najveci reward.

To moze da popravi broj, ali ne mora da popravi hod.

Problem koji se vidi nije "nema dovoljno treninga". Problem je da reward jos ne
opisuje dovoljno dobro stil hoda koji zelimo.

## Kratka verzija za profesora

Berkeley humanoid je posluzio kao kontrolni baseline jer dolazi iz vec
pripremljenog locomotion benchmark-a. Custom generated biomechanics model je tezi
problem jer model, kontakt, actuator scaling, reward i observations nisu unapred
uskladjeni. Trening je uspeo da proizvede joystick locomotion, ali trenutni gait
pokazuje contact/style artefakte. Sledeci naucno opravdan korak je dodavanje
gait-style ili motion-imitation reward-a, jer velocity tracking sam po sebi ne
garantuje prirodan ljudski hod.
