# Roadmap

Ažurirano 2026-11-12. Merljivi ciljevi i granice faza su u
[PROJECT_GOALS.md](PROJECT_GOALS.md). Izbor pristupa je obrazložen u
[FRAMEWORK_COMPARISON_2026-11-12.md](FRAMEWORK_COMPARISON_2026-11-12.md).

Aktivna grana je `navigation_walking_with_anatomical_human_robot`.
[MimicKit preflight](MIMICKIT_PREFLIGHT.md) je implementiran i izvršen:
kinematika i walking frejmovi prolaze; slede simulator/PD/action paritet,
kontinuitet reference i mali fizički task pre većeg treninga.

1. **Faza 1 — sačuvani rezultati.** Prirodan referentni hod je završen i
   zamrznut. Postoji i zaseban 60 s stand. -X recovery nije rešen; ograničeni
   red sa dva dodatna 2M testa je završen bez punog robustness prolaza.
2. **Faza 2 — prirodan A→B sa zaustavljanjem i stabilnim prelazima.** Najpre
   ograničen test kompatibilnosti i komandovane politike u izabranom pristupu,
   sa MimicKit-om kao prvim kandidatom. Proveriti model, podatke, kontrolu i
   mali task pre većeg treninga. Zatim naučiti polazak, hod, kočenje i stand,
   dodati prostorni cilj i proveriti povezane dolaske bez fizičkog reseta.
   Obavezni su očuvana prirodnost, 60 s stand, definisani potisci i raznovrsna
   nezavisna evaluacija. Postojeće implementacije koristiti kada odgovaraju.
3. **Faza 3 — buduća proširenja.** Predloženi redosled: obilazak prepreka,
   ustajanje posle pada, zatim izbegavanje dolazećih objekata. Svaki zadatak
   dobija poseban curriculum i fizički test. Pre ustajanja/udara pripremiti
   verzionisani kontaktni model celog tela i regresije prethodnih sposobnosti.

Joystick integracija je odvojena kasnija opcija, na novoj integracionoj grani
tek posle zasebnih testova oba moda. Zaštićena joystick grana i finalna
natural-walk arhiva se ne menjaju. Odbačeni bullet/A→B projekat se ne koristi.

Za svaki eksperiment sačuvati konfiguraciju, source commit, model/data poreklo,
checkpoint, pune fizičke rezultate i reprezentativne snimke, uključujući
neuspehe. Novi framework nije automatski bolji rezultat, a viši reward nije
zamena za prolaz dogovorenih kriterijuma.
