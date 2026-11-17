# BRAKE_TRANSITION v1 — 40M results

Run:

`20261117_1804_brake_transition_40m_20261117_180446_V1_steps-40m_reward-233p7205_s`

## Verdict

The run found a useful braking gradient but did not solve the transition. It is
not accepted as `BRAKE_TRANSITION`.

| Metric | Initial actor | Best handoff checkpoint (36.1M) | Final checkpoint (40.1M) |
|---|---:|---:|---:|
| Mean episode length / 200 | 92.6 | 112.3 | 126.9 |
| Mean horizontal speed | 1.474 m/s | 1.099 m/s | 1.106 m/s |
| Mean capture distance | 0.400 m | 0.290 m | 0.284 m |
| Mean support distance | 0.168 m | 0.152 m | 0.140 m |
| Capture-ready fraction | 0.0% | 3.0% | 1.0% |
| Handoff-ready fraction | 0.0% | 1.9% | 0.3% |
| Physical termination | 100% | 96.9% | 100% |

The raw PPO reward preferred 40.1M, while the physical handoff metric preferred
36.1M. Neither passes.

## Connected supervisor test

The 36.1M actor was tested in one continuous physics rollout:

```text
FINAL_WALKING -> BRAKE_TRANSITION -> FINAL_STANDING
```

`FINAL_WALKING` remained stable for 4.0 seconds. After a 0.2-second motor-target
blend, the bridge survived another 1.2 seconds and fell before `FINAL_STANDING`
was activated. At failure:

- horizontal speed: 2.542 m/s;
- minimum torso-up: 0.473;
- minimum height ratio: 0.283;
- maximum contact-foot slip: 3.844 m/s.

Evidence:

- report: `runs/brake_handoff_40m_diagnostic_seed11/report.json`;
- video: `videos/final_skill_transitions/brake_handoff_40m_diagnostic_seed11_quad.mp4`.

## Root causes and v2 corrections

1. Training reset from ideal BVH states, not physical states produced by the
   frozen walking policy. At the connected handoff, the closest old reset had
   total physical mismatch 0.793 and root-velocity RMSE 0.755.
2. A fallen terminal state received imitation reward instead of a negative
   terminal signal.
3. The original gate divided an already-normalized physical-termination
   fraction by episode length, making 96.9% termination appear as 0.86%.

V2 exports 256 real `FINAL_WALKING` states into
`skills/assets/final_walking_state_bank.npz`. Of these, 137 are inside the
configured 0.2–0.8 second lead window. Against the same connected handoff, the
best state-bank mismatch is 0.046 and root-velocity RMSE is 0.058 — about 13×
smaller than v1. V2 also adds a negative physical-terminal reward and corrects
checkpoint gating.
