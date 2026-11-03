# Od uspešnog hoda do komandi

Sačuvao sam drugi uspeh u `prezentacija/2nd_success_stiff_arms`: checkpoint
`000010035200`, konfiguraciju, XML, BVH, dva videa i SHA-256 manifest.
Seed 11 i seed 22 završavaju 500 koraka bez pada (1443.635 i 1445.914 reward).
Trening sam namerno zaustavio oko 17M koraka. Oznaka `unknown_f` u starom run-u
potiče od prekida; arhivirao sam vizuelno potvrđeni 10M checkpoint.

## Šta sam našao u prethodnom neuspehu

Pregledao sam `20261102_2355_arms_joystick_v1_30m_20261102_235444_259`.
Poslednji eval završio je prosečno posle 57.5/500 koraka, sa reward -143.508.
Naziv launchera je govorio curriculum, ali je od početka koristio
`standard_easy`: napred, nazad, bočno, yaw i stajanje. Istovremeno sam menjao
broj aktuatora i prelazio sa čistog imitation reward-a na staru task nagradu.

U toj nagradi sam našao kvadratnu overspeed kaznu deljenu brzinom komande,
sa donjom granicom 0.05. Na nultoj komandi i brzini 0.4 m/s samo taj deo može
biti 48 pre završnog clipping-a. To je vrlo loš početak za već hodajuću politiku.
Uz to sam istovremeno nagrađivao brzinu originalne reference i drugu,
zadatu brzinu. Ovi nalazi objašnjavaju konkretne konflikte, ali ih ne tumačim
kao eksperimentalni dokaz jednog jedinog uzroka neuspeha.

Našao sam i problem novih ulaza: u pure treningu su komande stalno nula,
njihov normalizer std je oko 0.03162, a odgovarajuće ulazne težine nisu naučene.
Komanda 0.65 zato postaje normalizovani signal oko 20.55. U posebnom warm-start
checkpointu postavio sam command std na 1 i command ulazne težine na nulu.
Ostale težine/statistike čuvam. Testiram da početne akcije ostaju iste čak
i kada uključim nenulte komande.

## Kako sada treniram

Polazim od uspešnih 26 DOF, istog XML-a, reference, fizike i residual akcija.
Zadržavam imitation signal za pozu, joint brzine, visinu/orijentaciju i stopala.
Brzinu root-a iz BVH-a izostavljam iz ove nagrade, a dodajem bounded praćenje
komande: `3 * ((1-w) * imitation_without_root_velocity + w * tracking)`.
Tracking je `0.75 * linear + 0.25 * yaw`; fizički pad i dalje dobija staru
terminalnu kaznu. Originalni `pure` i `mixed` režimi ostaju reproduktivni.

Izmerio sam približno 1.1 m/s na sačuvanoj politici. Početak centriram oko
stvarno naučenog hoda, umesto da zahtevam naglo usporavanje na 0.14 ili 0.5 m/s.
Preflight pripremljenog checkpointa sa novim reward-om i komandom 0.5 m/s
završio mi je 500/500 za oba seeda; brzine su bile 1.0919 i 1.1144 m/s,
pose score oko 0.9963. Očekivano je da pre učenja još ignoriše komandu:
time sam proverio da početno zadržava hod. Rezultat čuvam u
`analysis_outputs/joystick_v2_transfer_preflight.json`.

| Nivo | Brzina napred | Yaw | Udeo tracking nagrade |
|---|---|---|---|
| 0 | 1.05–1.15 m/s | 0 | 20% |
| 1 | 1.00–1.20 m/s | ±0.10 rad/s | 30% |
| 2 | 0.95–1.25 m/s | ±0.25 rad/s | 40% |

Budžet delim na tri bloka od približno 10M koraka. Zadržavam 64 env-a,
PPO 10/4/2, batch 128 i learning rate 5e-5. Između blokova prenosim težine
i normalizator; optimizer i lokalni brojač ponovo kreću. PPO može malo
prekoračiti nominalni budžet da završi ceo batch.

Pre povećanja težine merim parove komandi sa istim seed-ovima 11 i 22:
tražim 500/500 bez pada, pose ≥0.95, male greške prosečne brzine i stvarnu
razliku odgovora na komande. Dobar reward uz ignorisanje joysticka ne prolazi.
Ako provera ne prođe, sledećih 10M ostajem na istoj težini. Ako prosečno
trajanje padne ispod 450 koraka, za sledeći blok vraćam prethodni checkpoint.

Još ne tvrdim da imam pun joystick: stop, reverse, bočni hod i velika skretanja
nisu uključeni. Klip/phase clock ostaju isti; target observation još sadrži
originalni root XY. Pratim da li to ograničava skretanje. Ako ova faza uspe,
sledeća može konzistentno menjati referencu/brzinu faze ili koristiti više klipova.

## Zašto su ruke krute

Otkrio sam da retargeter traži `LeftHand/RightHand`, a klip ima
`LeftWrist/RightWrist`. Svih osam arm targeta zato je bilo nula. XML ima aktivne
aktuatore i upotrebljive opsege. Dodao sam opt-in `arm_reference_mode=cmu_wrist`
sa testovima na stvarnom BVH-u i XML-u. To je heuristički zamah ramena i blag
pregib lakta, a ne potpuni upper-body IK. U ovom joystick eksperimentu zadržavam
`legacy` ruke da bih prvo proverio prenos već uspešnog hoda na komande.

## Pokretanje i praćenje

Iz Windows PowerShell-a pokrećem:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_joystick_curriculum_v2.ps1
```

Launcher mi ispisuje `.stdout.log` i `status.json`. U logu vidim trenutni blok,
eval i napredak, a u statusu odluke o težini i checkpoint za nastavak.
Proces nema timeout. Za prikaz sačuvanog uspeha na Windowsu koristim:

```bash
./.venv-win/Scripts/python.exe scripts/evaluate_second_success.py
```
