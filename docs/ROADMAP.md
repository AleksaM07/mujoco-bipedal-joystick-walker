# Naš plan posle prvog hoda

Mi prvi uspešan model čuvamo kao polaznu tačku za poređenje svih narednih faza.
Za svaku fazu pravimo odvojen commit, merimo rezultat i snimamo najbolji checkpoint.

| Faza | Šta ćemo uraditi | Kako ćemo znati da smo uspeli |
| --- | --- | --- |
| 1. Arhiva i jasnoća | Čuvamo težine, assete, videe, log i preglednu dokumentaciju. | Prolazimo SHA-256 proveru i možemo da pustimo arhivirani model. |
| 2. Kontrola brzine i smera | Usklađujemo referencu i komande; postepeno učimo joystick. | Merimo grešku komande i padove pri promenama brzine/yaw-a kroz više seedova. |
| 3. Aktivne ruke i ostali zglobovi | Definišemo novi model i action/observation dimenzije; proveravamo retargeting. | Snimamo prirodno koordinisan hod i merimo kontakte/limite i stabilnost. |
| 4. Ustajanje | Uvodimo padove kao reset stanja i učimo povratak u stabilan stav. | Merimo uspešnost, vreme ustajanja i prelaz nazad na hod. |
| 5. Izbegavanje projektila u simulaciji | Dodajemo opažanje dolazećih objekata i progresivno teže scenarije. | Merimo izbegavanje, sudare i oporavak bez gubitka kontrole kretanja. |

Mi pri joystick radu prvo moramo rešiti činjenicu da pure režim potiskuje komande.
Samo uključivanje komandnog vektora ne garantuje da je politika naučila da ga
prati. Transformaciju reference ili hijerarhijsku kontrolu uporedićemo uz isti
zaleđeni walking baseline.

Mi ruke ne uključujemo prostim učitavanjem istog 18-action checkpoint-a u veći
model. Pri promeni dimenzija biće nam potreban eksplicitan transfer/destilacija
ili novi trening. Prvo ćemo proveriti fizički model i dostupne reference za ruke.

Mi za ustajanje koristimo posebne kriterijume uspeha i reset distribuciju.
Još nismo dokazali robustan hod pod udarima, višeminutnu stabilnost ili navigaciju
do proizvoljne tačke. Te rezultate ćemo meriti kao nove sposobnosti.
