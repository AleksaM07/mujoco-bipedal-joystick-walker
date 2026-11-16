# Start/walk/stop server curriculum (40M minimum, 75M total)

The server job is intentionally staged instead of repeating the failed direct
full-clip fine-tune:

1. 40M steps on a finite middle stride from `133_21`;
2. 15M steps on overlapping stand-to-walk and walk-to-stand clips;
3. 20M steps on the complete 6.54 s start/walk/stop take.

The stride phase uses the mixed task/DeepMimic reward. The earlier 5M pure
imitation probe preserved a near-perfect pose score while physical completion
declined, so simply extending that objective would train the wrong behavior.

Each phase starts a fresh critic while restoring the actor and observation
normalizer. The next phase starts only if at least 75% of evaluation episodes
reach `done_motion` and no more than 25% end through low/tipped/invalid states.
The natural-walk archive is read-only. Its archived XML is passed explicitly,
so a server's current `generated_models/` copy cannot silently replace the
checkpoint-compatible model.

From an updated checkout of
`navigation_walking_with_anatomical_human_robot`, run:

```bash
NUM_ENVS=256 NUM_EVAL_ENVS=16 \
  bash scripts/run_transition_curriculum_40m.sh
```

For a larger GPU, increase `NUM_ENVS` only after checking memory, for example:

```bash
NUM_ENVS=1024 NUM_EVAL_ENVS=32 \
  bash scripts/run_transition_curriculum_40m.sh
```

The script automatically uses `BATCH_SIZE=256` with `NUM_ENVS=1024` and four
minibatches. `BATCH_SIZE` and `NUM_MINIBATCHES` can be overridden, but the
script rejects combinations that violate Brax's divisibility requirement
before creating a training run.

The script refuses to start if another `train.py` is active or if any phase
would reuse an existing run tag. Override `PYTHON_BIN`, `NATURAL_CHECKPOINT`,
`NATURAL_XML`, or `TRANSITION_BVH` through environment variables when the
server layout is different.
