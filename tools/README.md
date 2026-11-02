# Kako mi biramo dijagnostički alat

Mi svakodnevno koristimo `scripts/evaluate_first_success.py`, `train.py` i
`evaluate.py`. Ove alate pokrećemo kada imamo konkretno dijagnostičko pitanje,
iz korena projekta preko `python -m tools.ime_alata --help`.

| Kada proveravamo… | Koristimo… |
| --- | --- |
| Integritet prvog uspeha | `verify_success_archive.py` |
| 18→26 DOF warm-start za ruke | `migrate_checkpoint_to_arms.py` |
| Vreme, kinematiku i continuous reference | `check_recovery_reference.py`, `check_reference_curves.py` |
| Kontakt, kolizije i dinamiku | `check_reference_contacts.py`, `check_reference_collisions.py`, `audit_reference_dynamics.py` |
| IK i faze retargetinga | `audit_ik_fit.py`, `check_retarget_stages.py` |
| Support anchor i root | `check_root_lock_transitions.py`, `check_root_lock_ablation.py` |
| Reward | `check_reward_discrimination.py`, `debug_reward_landscape.py` |
| Replay bez RL | `debug_reference_playback.py`, `debug_reference_pd_tracking.py`, `debug_reference_static_hold.py` |
| Prikaz sirovih i pripremljenih referenci | `render_raw_bvh_videos.py`, `render_bvh_reference_videos.py` |
| Istorijske A/B eksperimente | `check_knee_offset_ablation.py`, `check_root_height_ablation.py`, `check_stability_prior_attribution.py` |
| Poređenje grana/runova | `probe_branch_invariants.py`, `summarize_branch_evidence.py`, `analyze_playback_trace.py` |
| Step/control eksperimente | `run_step_probe_matrix.py`, `run_control_paradigm_probe.py` |
| Pripremu podataka | `filter_cmu_walking.py`, `make_strict_bvh_list.py` |

Mi `recovery_dynamics.py` i `recovery_replay.py` koristimo kao zajedničke module
alata. Dijagnostičke ablation opcije ne proglašavamo produkcionim presetima.
Naročito čuvamo dokaze za knee-offset i smoothing kandidate koje nismo prihvatili.

Mi filter podataka pokrećemo bez `--delete` kada želimo samo izveštaj. Pisanje
manifesta liste BVH klipova pokrećemo eksplicitno; to nije deo evaluacije modela.
