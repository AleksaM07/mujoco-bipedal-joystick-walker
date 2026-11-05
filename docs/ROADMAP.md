# Roadmap

Radimo sposobnosti redom; ne menjamo više velikih stvari u istom eksperimentu.

1. **Završni prirodni hod — završeno 2026-11-05**
   - elbow curriculum 25/50/75/100% prošao je sve gate-ove;
   - finalni checkpoint, config, XML, BVH, dokazi i video su zamrznuti;
   - dodatni seedovi ostaju opciona robusnost, ne uslov za početak A→B.
2. **Čist A→B zadatak**
   - napraviti novi sistem od nule;
   - koristiti relativni cilj u lokalnom frame-u, željenu brzinu i završnu
     orijentaciju;
   - ne preuzimati ništa iz odbačenog bullet/A→B projekta.
3. **Integracija**
   - sačuvati joystick-only granu iz `main`;
   - napraviti čist interfejs za izbor joystick ili A→B moda;
   - spojiti ih u `main` tek kada oba imaju zasebne testove.
4. **Više locomotion skillova**
   - proceniti više BVH klipova i AMP/MimicKit pristup;
   - odvojiti walk, run i stand metrike i prelaze.
5. **Ustajanje**
   - posebna reset distribucija i success kriterijum;
   - prelaz `fallen → stand → walk`.
6. **Avoidance**
   - tek nakon stabilne locomotion arhitekture;
   - prepreke i projektili ostaju novi modul, ne nasleđe starog projekta.

Za svaku fazu čuvamo checkpoint, konfiguraciju, najmanje četiri deterministička
rollout-a i kratak rezultat neuspelih ablations.
