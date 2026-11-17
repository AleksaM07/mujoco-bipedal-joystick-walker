# BRAKE_TRANSITION 40M experiment

## Why this is a separate skill

The two accepted frozen policies solve different state distributions:

- `FINAL_WALKING` is stable around a roughly 1.1 m/s cyclic gait;
- `FINAL_STANDING` is stable only after momentum has already been removed.

A connected seed-11 probe preserved one physical MuJoCo state and blended the
two actors' motor targets for 0.5 seconds. `FINAL_WALKING` covered 4.24 m in the
first 4 seconds without falling. At the end of the direct blend speed was still
0.918 m/s. `FINAL_STANDING` then fell after 0.88 seconds, at 2.52 m/s. The report
is in `runs/final_walk_to_final_stand_probe_20261117_seed11/report.json`; the
baseline video is
`videos/final_skill_transitions/direct_walk_to_stand_seed11_quad.mp4`.

This establishes the missing contract:

```text
FINAL_WALKING -> BRAKE_TRANSITION -> FINAL_STANDING
```

`BRAKE_TRANSITION` is initialized from `FINAL_WALKING`, but is trained only on
the short braking window. It does not replace or fine-tune either final policy.

## Training task

The bridge uses the exact final-walk assets:

- BVH `02_01.bvh`, selected processed clip 1 (source frames 79--213);
- archived final-walk XML hash `494cb6707a59...`;
- observation sizes 366/809 and 26 actions;
- each reset starts 0.2--0.8 seconds before the calibrated double-support stop;
- the episode is 200 control steps, rather than spending most samples
  relearning ordinary walking.

The dense additive reward tracks speed decay, capture-point distance, support
distance, torso orientation, height, angular speed, contact, smooth action and
foot slip. It therefore remains informative before the strict stand region is
reached. A handoff becomes eligible after 0.20 seconds inside the capture gate.

## Server command

```bash
set -o pipefail
NUM_ENVS=1024 NUM_EVAL_ENVS=32 \
  bash scripts/run_brake_transition_40m.sh 2>&1 | tee brake_transition_40m_server.log
```

The launcher refuses another active `train.py`, validates both final-policy
artifacts, trains exactly 40M environment steps, and evaluates 21 checkpoints.
It selects by capture/handoff readiness rather than raw PPO reward. If the
training gate passes, it automatically runs a no-reset connected rollout of all
three policies and writes its quad video under `videos/final_skill_transitions/`.

To validate a checkout without starting training:

```bash
PRECHECK_ONLY=1 bash scripts/run_brake_transition_40m.sh
```
