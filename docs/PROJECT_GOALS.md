# Ciljevi projekta: tri faze

Dogovoreni obim: 2026-11-12. Ovaj dokument definiše šta želimo da postignemo;
[ROADMAP.md](ROADMAP.md) određuje redosled rada, a
[poređenje pristupa](FRAMEWORK_COMPARISON_2026-11-12.md) obrazlaže izbor tehnologije.
Predloženi evaluacioni protokol ispod odvojen je od već potvrđenih rezultata.

**Cilj je isti anatomski humanoid koji se prirodno kreće do zadatog cilja,
zaustavlja, održava ravnotežu i nastavlja kretanje.** Projekat treba da omogući
kasnije učenje izbegavanja prepreka, ustajanja i izbegavanja projektila.
Prva završena funkcionalna celina jeste faza 2; faza 3 je ambiciozno proširenje.

## Faza 1 — dokazani hod i postojeće polazne sposobnosti

Status: **prirodan hod završen i zamrznut**. Postoji i zaseban stand rezultat.

| Rezultat | Šta je dokazano | Granica tog dokaza |
| --- | --- | --- |
| Prirodan hod po referenci | Isti anatomski model, 26 aktuatora, očišćen CMU `02_01.bvh`, aktivne ruke i laktovi; 500/500 koraka na četiri fiksna evaluaciona seed-a. | To je praćenje jednog walking ciklusa, bez proizvoljnog XY cilja i naučenog zaustavljanja. |
| Mirno stajanje | Sačuvani zasebni stand kandidati održavaju 60 s iz neutralnog početka. | Ne dokazuje stajanje posle hoda niti robusnost iz raznovrsnih početnih poza. |
| Oporavak od potiska | Postoje kandidati koji prolaze +X i ±Y u postojećem protokolu. | -X i dalje pada; ceo robustness suite nije prošao. |
| Reprodukcija | Sačuvani checkpoint-i, konfiguracije, model, referenca, izveštaji i snimci. | Više seed-ova sa identičnim fizičkim resetom nije više različitih početnih stanja. |

Dokazi: [finalni hod](../prezentacija/5th_success_final_natural_walk/FINAL_SUCCESS.txt),
[60 s stand](STAND_HORIZON60_RESULTS.md),
[postojeći push rezultati](STAND_PUSHX_RESULTS.md).
Red sa dva dodatna 2M recovery testa završen je 2026-11-12 bez potpunog prolaza;
[queue.json](../runs/stand_recovery_queue_20261112_2x2m_v2/queue.json) ima
`status=budget_exhausted`. To nije novi uspeh faze 1.

Arhiva `prezentacija/5th_success_final_natural_walk` ostaje nepromenjena.
Njene težine su baseline i mogući izvor transfera; kompatibilnost sa novim
framework-om nije dokazana. Uspešan hod se ne proglašava neuspešnim zato što
novi zadaci traže dodatnu obuku.

## Faza 2 — prva završena celina: prirodan A→B sa stabilnim prelazima

Status: **cilj razvoja; još nije implementirana kao celina**.
Prva verzija radi na ravnom tlu, u prostoru bez prepreka.

| Zahtev | Uslov uspeha |
| --- | --- |
| Prirodan hod | Bez pada, izraženog klizanja stopala i očigledno neprirodnih korekcija. Uporediti ravne deonice pri uporedivoj brzini sa sačuvanim hodom; meriti klizanje stopala tokom kontakta, držanje i nagle promene pokreta, uz video. |
| A→B | Dobija cilj u prostoru i sam prilagođava smer i kretanje. Evaluacija obuhvata ciljeve ispred, bočno i iza, kao i više udaljenosti i početnih orijentacija. |
| Zaustavljanje | Udaljenost root-a od cilja u XY je `< 0.25 m`, XY brzina `< 0.15 m/s`, uz stabilno uspravno stanje; svi uslovi neprekidno najmanje `1 s`. Prolazak kroz cilj bez zaustavljanja nije uspeh. |
| Prelazi | `stand → walk → brake → stand → walk`, iz različitih faza hoda, bez resetovanja fizike, teleportovanja root-a ili poništavanja brzina. |
| Stajanje | Održava 60 s nominalnog stajanja po postojećem nezavisnom fizičkom kriterijumu. |
| Push recovery | Pri definisanim potiscima zadržava ravnotežu ili napravi korektivni korak i vrati se u stabilno stajanje. Privremeno podizanje stopala tokom oporavka je dozvoljeno. |
| Ponovljivost | Isti zamrznuti kandidat prolazi unapred definisane različite ciljeve, početna stanja, faze hoda i trenutke potiska. Čuvaju se i neuspesi. |

Konačni stand uslovi uključuju torso-up `>= 0.9`, visinu `>= 85%` nominalne
uspravne visine, oba kontakta stopala i konačne numeričke vrednosti, kao u
[SKILLS_REVIEW.md](SKILLS_REVIEW.md). A→B meri rastojanje od cilja B;
postojeći stand test meri drift od početnog anchor-a. To su različite metrike.
Završna yaw orijentacija može se dodati zasebno; nije uslov koji je korisnik
zadao za prvu verziju.

### Nezavisna evaluacija

Postojeći [stand robustness v1](STAND_ROBUSTNESS.md) ostaje obavezan regresioni
test: 60 s bez potiska i četiri 20 s epizode sa silom od 5% težine modela,
trajanja 0.2 s, u ±world X/±world Y. Potisak deluje na toraks od 5.0 do 5.2 s.
Posle otpuštanja traži se nov 1 s stabilan dwell unutar 5 s i stabilan kraj
epizode bez pada. Zadržava se postojeći anchor kriterijum; širi korektivni
koraci, ako postanu zaseban zadatak, dobijaju imenovan novi protokol.

Predlog dodatnog protokola, koji treba zaključati pre poređenja kandidata:

- A→B: osam pravaca cilja i udaljenosti 1, 3 i 5 m, sa najmanje tri početne
  orijentacije. Zabeležiti da li se humanoid okrene ka cilju ili hoda unazad.
- Prelazi: zaustavljanje u najmanje četiri faze ciklusa i pri više brzina
  unutar deklarisanog radnog opsega; zatim novi cilj iz postignutog stajanja.
- Potisci: više trenutaka impulsa i stvarno različite početne poze/orijentacije,
  uz očuvanje originalnih pet regresionih slučajeva.
- Jedna povezana putanja sa više dolazaka i ponovnih polazaka, bez reseta
  između ciljeva. Odvojeno prikazati mirno stajanje i oporavak tokom kretanja.
- Čuvati success rate, broj pokušaja i padova, vreme dolaska/oporavka,
  prekoračenje cilja, završnu brzinu/dwell, drift i klizanje stopala.
  Nasumičnu evaluaciju odvojiti od trening raspodele i izbora checkpoint-a.

Za konačno prihvatanje svi slučajevi zaključanog obaveznog suite-a moraju
proći. Širi stres-test izvan deklarisanog opsega prikazuje granice sposobnosti.
Broj trening seed-ova i širi statistički budžet definišu se nakon merenja na
serveru; ne predstavljati četiri identična reseta kao statistiku robusnosti.

Rezultat faze 2 je reproduktivan demo i izveštaj za jednu celinu. Odvojeni
walking i stand snimci, visoka PPO nagrada ili uspeh različitih checkpoint-a
na različitim delovima testa nisu dovoljan dokaz.

## Faza 3 — ambiciozno proširenje

Status: **željene buduće sposobnosti; njihova implementacija nije uslov faze 2**.
Predloženi redosled je 3A → 3B → 3C; svaka ima zasebnu evaluaciju.

| Podfaza | Cilj | Dokaz koji ćemo tražiti |
| --- | --- | --- |
| 3A — prepreke | Stigne do B obilazeći prepreke; kasnije i pokretne prepreke. | Dolazak sa istim stop/dwell kriterijumom, bez nedozvoljenog kontakta celog tela, na neviđenim rasporedima; broj sudara, padova i uspešnih dolazaka. Početi jednostavnim geometrijama i poznatim stanjem scene. |
| 3B — ustajanje | `fallen → get_up → stand → walk`, iz više poza na leđima, stomaku i boku, zatim nastavak zadatka. | Bez teleportacije ili ručne pomoći; stvarni kontakti sa tlom, stabilno završno stajanje i nastavak kretanja. Posebno meriti uspeh i vreme za svaku grupu početnih poza. |
| 3C — projektili | U simulaciji izbegne dolazeću loptu/objekat korakom, saginjanjem ili drugom fizički ostvarivom reakcijom, pa nastavi zadatak. | Objekat bi pogodio neaktivnog humanoida; uspeh zahteva izostanak udara, očuvanu stabilnost i povratak zadatku. Varirati pravac, brzinu, veličinu, vreme do udara i redosled više objekata. |

Potisak, ustajanje i izbegavanje projektila nisu ista veština. `HumanoidPerturb`
demonstracija otpornosti na udar nije dokaz anticipativnog izbegavanja.
U fazi 3C prvi korak je poznato stanje objekta u simulaciji; kasnija procena iz
senzora/slike uvodi zaseban problem opažanja. Ne obećavamo izbegavanje
proizvoljno brzog objekta niti uspeh u fizički nerešivoj situaciji.

## Šta arhitektura mora da omogući već pri izboru za fazu 2

Ovo su zahtevi za proširivost, ne nalog da sada gradimo sve module:

1. **Odvojeni cilj i motorna kontrola.** Lokalni cilj, željena brzina/smer ili
   parcijalni body-goal moraju stvarno uticati na naučenu politiku. Koji je
   tačan interfejs zavisi od izabranog pristupa; ASE latent nije obavezan.
2. **Kontinuitet fizike.** Promena cilja, režima ili reference čuva stanje
   simulacije. Naučeni prelazi se proveravaju; sirovo mešanje izlaza politika
   nije dokaz stabilnog prelaza.
3. **Proširive opservacije.** Sada cilj i stanje tela; kasnije prepreke i
   dolazeći objekti, uz odvajanje stvarno dostupnih od privilegovanih podataka.
4. **Poseban trening i kriterijum za recovery.** Ustajanje mora dozvoliti
   korisne kontakte ruku/torza i početak iz palih poza; pad ne sme automatski
   završiti get-up epizodu. Numerički neuspeh i dalje završava epizodu.
5. **Verzionisani model i mapiranje.** Sačuvati masu, kinematiku, granice,
   aktuatorski raspored, PD/action mapiranje i koordinatne konvencije.
   Novi framework dobija proveru fizičke i kinematičke kompatibilnosti.
6. **Nezavisne metrike i regresije.** Promena reward-a ne menja definiciju
   uspeha. Svaka nova veština mora da sačuva proveru hoda i faze 2.

Konkretan fizički preduslov faze 3: u
[zamrznutom XML-u](../prezentacija/5th_success_final_natural_walk/generated_models/human_male_180cm_75kg_standard_trainfast_v21_arms.xml)
14 geometrija tela ima `contype=0, conaffinity=0`, dok su kontakti uključeni
na dva đona. Za ustajanje i pouzdanu detekciju udara/prepreka treba proveriti
kompajlirani kontaktni model i napraviti zasebnu verziju sa odgovarajućim
geometrijama i filtrima kontakta. To je promena fizike koja zahteva regresiju;
ne sme se tiho uneti u finalnu natural-walk arhivu.

## Granice i sačuvani rezultati

- Razvoj je na `navigation_walking_with_anatomical_human_robot` (preimenovana
  skills grana); joystick grana ostaje zaštićena. Eventualna
  integracija joystick komandi dolazi kasnije, u zasebnoj integracionoj grani.
- Odbačeni `bipedal-humanoid-bullet-avoidance` kod se ne koristi.
- Koristiti postojeće javne metode i implementacije gde odgovaraju cilju;
  ne pisati sve od nule niti birati framework samo po nazivu algoritma.
- Nove politike, modeli i dokazi pišu se u nove lokacije. Za svaki prihvaćeni
  rezultat sačuvati source commit, config, model/data hash, seed-ove,
  checkpoint, pun izveštaj, reprezentativne snimke i neuspešne slučajeve.
