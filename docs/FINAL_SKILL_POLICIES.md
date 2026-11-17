# Frozen locomotion policies

The phase-2 supervisor uses two explicitly named, immutable policy artifacts:

| Label | Role | Checkpoint | Acceptance evidence |
|---|---|---|---|
| `FINAL_WALKING` | Natural cyclic forward walking | `prezentacija/5th_success_final_natural_walk/model_000001024000` | `prezentacija/5th_success_final_natural_walk/FINAL_SUCCESS.txt` |
| `FINAL_STANDING` | Nominal 60-second standing | `runs/stand_horizon60_20261106_1m_lr3e-4/resume/20261106_1414_stand_balance_v1_resume_1000000_lr0_0003_V1_steps-1m_reward-8554p6201_s/checkpoints/000001024000` | `docs/STAND_HORIZON60_RESULTS.md` |

These names are aliases, not copies. The archived natural walk, its checkpoint,
and the joystick branch are not modified. Both checkpoints use the same robot
XML hash (`494cb6707a59...`), 366 policy observations and 26 actions, but each
actor retains its own learned observation normalizer and reference semantics.

The intended stop pipeline is:

```text
FINAL_WALKING -> BRAKE_TRANSITION -> FINAL_STANDING
```

`BRAKE_TRANSITION` is deliberately not marked final. It is a specialized bridge
that starts near a stop request, removes walking momentum, enters the standing
capture region, and hands the unchanged physical state to `FINAL_STANDING`.
Normal walking is never retrained inside that bridge.

Before accepting a bridge checkpoint, the connected rollout must show, without
resetting or editing `qpos`/`qvel` at a switch:

- no physical fall;
- speed below 0.15 m/s;
- root drift below 0.25 m from the stand handoff anchor;
- torso-up at least 0.90 and height ratio at least 0.85;
- both feet in contact for the final one-second dwell;
- successful `FINAL_STANDING` continuation after the handoff.

The canonical definitions and fail-closed metadata checks live in
`skills/policy_registry.py`. `tools/evaluate_skill_transitions.py` is the first
connected supervisor probe; it loads both frozen actors independently.
