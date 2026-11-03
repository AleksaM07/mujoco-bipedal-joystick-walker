# Kako smo ubrzali trening

Na poslednjem završenom 30M run-u izmerili smo približno **850–900 environment
koraka u sekundi**. Zbog toga 30 miliona koraka završavamo za oko deset sati,
umesto da procena bude više dana. Ubrzanje nije došlo iz jedne velike promene,
već iz nekoliko manjih.

## Šta direktno ubrzava izvršavanje

1. **JIT i paralelni environment-i.** MuJoCo/MJX dinamiku, reset i reward
   izvršavamo kroz JAX programe nad 64 environment-a istovremeno. Python ne
   obrađuje svaki robot posebno.

2. **JAX compilation cache.** Kompajlirane programe čuvamo u Linux cache-u:
   `${HOME}/.cache/mujoco-bipedal-jax`. Ponovljeni kompatibilni run zato može da
   preskoči deo skupog XLA kompajliranja. Nova konfiguracija, poput prelaska sa
   mixed na pure reward, ipak zahteva novi compile.

3. **Jeftiniji reset.** Raniji reset je mogao više puta da pozove pun
   `mjx.forward` dok proverava kandidate. Sada za kandidate koristimo samo
   potrebnu kinematiku, izaberemo stanje, pa puni forward radimo jednom. Testovi
   proveravaju da konačno stanje ostaje fizički isto.

4. **Razuman broj environment-a.** Koristimo 64 environment-a i batch 128.
   Veći broj nije automatski brži ako izazove GPU/WSL OOM ili preveliku
   kompilaciju. Stabilan run koji ne restartujemo je praktično mnogo brži.

5. **Kontrolisana količina logovanja.** Training metrike šaljemo približno na
   svakih 50.000 koraka, a evaluaciju radimo 31 put tokom 30M run-a. Tako imamo
   dovoljno dijagnostike bez čestog zaustavljanja GPU rada zbog host callback-a.

## Šta uglavnom sprečava gubitak vremena

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` sprečava da JAX odmah zauzme skoro svu
  GPU memoriju. To nije sirovo ubrzanje, ali smanjuje OOM rizik.
- `bare` režim isključuje ERFI i domain randomization dok izolujemo problem.
  Time je eksperiment jednostavniji i nešto jeftiniji.
- Launcher koristi lock, jedinstven run folder i nema timeout. Ne pokrećemo dva
  skupa treninga slučajno i ne gubimo višesatni run zbog veštačkog ograničenja.

## Šta ubrzava učenje, a ne simulaciju

Ne krećemo od slučajne politike. Proširili smo najbolji 18-DOF walking
checkpoint na 26 akcija: stare težine i normalizator ostaju, a nove arm akcije
počinju kao neutralni residual. Warm-start ne povećava broj koraka u sekundi, ali
može drastično smanjiti broj koraka potreban da ponovo dobijemo hod.

Najvažnije je da brzinu uvek merimo posle prvog JIT-a. Vreme kompajliranja i
stvarni training SPS predstavljaju dve različite stvari.
