# Lower-rate stand recovery continuation — 2026-11-09

The run `runs/stand_pushx_continue_20261109_1m_lr1e-4` completed 1,024,000 new
training steps, all twenty short acceptance rollouts, four videos and all 25
long robustness cases. **No checkpoint passes the complete suite.** Every new
checkpoint still falls after the -X push, and all five also lose the 60-second
nominal hold. Do not continue from the latest weights.

## Experiment and provenance

Training started from clean commit `4373829` at 13:51:49 UTC on 2026-11-09
and exited with code 0 at 14:52:10 UTC. The source is push-trained checkpoint
409,600 from `stand_pushx_20261106_1m_lr3e-4`, which already passed the 60 s
hold and +X/±Y probes but failed -X at 6.94 s.

Learning rate is `1e-4`, down from `3e-4`. The continuation retains
`balance_v1`, 3,000-step training episodes, seed 7, 64 environments, the same
training-only pushes up to 5% of body weight, model, physics, action mapping
and independent acceptance thresholds. Actor, critic and observation
normalizer were restored; the optimizer was fresh. All 58 environment and
25 training fields in the saved configuration match the launch plan.

The source lineage contains 3,481,600 training steps. This run's final
checkpoint therefore contains 4,505,600 steps in its lineage. The experiment
tests a conservative continuation; it is not a matched causal demonstration
that learning rate caused the previous late regression.

## Completed short gates

Evidence: `analysis_outputs/skills/stand_pushx_continue_short_gates_20261109.json`.
The gate finished at 15:10:50 UTC (16:10 local). Every checkpoint completes
500/500 steps on seeds 11, 22, 33 and 44, without a physical termination,
with final success and continuous final dwell of 10 s.

| New checkpoint | Max drift | Mean speed | Final dwell | Gate |
| ---: | ---: | ---: | ---: | --- |
| 204,800 | 1.894 cm | 0.004793 m/s | 10 s | pass |
| 409,600 | 1.326 cm | 0.004212 m/s | 10 s | pass |
| 614,400 | 1.576 cm | 0.004801 m/s | 10 s | pass |
| 819,200 | 2.642 cm | 0.004779 m/s | 10 s | pass |
| 1,024,000 | 2.658 cm | 0.004458 m/s | 10 s | pass |

All physical summary metrics agree across the four seeds within each
checkpoint. Neutral resets are physically identical, so these establish
repeatability, not varied initial-state robustness. The short-gate selection
is 409,600, with 3,891,200 training steps in its lineage. Selection for longer
recovery must be made from the full per-checkpoint robustness results.

Reviewed decoded frames 0, 166, 333 and 499 of
`videos/000000409600/stand_seed_11.mp4`, including the final pose at the end
of the 500-step rollout. All sampled views show upright stationary stance.
The video contains 500 frames at 50 fps; the review contact sheet is saved
under the run's `review/short_gate_409600_seed11_frames.jpg`. This is a review
of spaced frames supported by every-step physical metrics, not a long-hold
or push-recovery result.

## Completed robustness validation

The runner finished with `status=complete`, `video_complete=true` and
`robustness_complete=true`. Its `robustness_candidates_passed` list is empty.
Pipeline completion therefore records a valid failed experiment, not robust
standing.

| New checkpoint | Hold | +X | -X | +Y | -Y |
| ---: | --- | --- | --- | --- | --- |
| 204,800 | fall, 45.08 s | fall, 9.32 s | fall, 6.82 s | pass | pass |
| 409,600 | fall, 37.58 s | pass | fall, 6.84 s | pass | pass |
| 614,400 | fall, 32.84 s | pass | fall, 6.78 s | pass | pass |
| 819,200 | fall, 35.06 s | fall, 7.78 s | fall, 6.82 s | pass | pass |
| 1,024,000 | fall, 56.96 s | fall, 7.76 s | fall, 6.84 s | pass | pass |

The lower learning rate avoided the previous immediate late-checkpoint collapse,
but did not improve the remaining direction: -X failure time stays near the
source's 6.94 s. It also did not retain the source's confirmed 60-second hold.
The source checkpoint remains the safer starting point because it passes the
nominal hold and +X/±Y probes.

The saved -X replay for new checkpoint 409,600 was rendered offline without a
new physics rollout. Reviewed steps 0, 250, 260, 275, 300 and terminal step 342.
The robot remains upright through the force release at 5.2 s, continues moving
backward without a sufficient corrective step, leans behind its support by
5.5–6.0 s and falls at 6.84 s. The video and contact sheet are in the run's
`review/robustness_409600_push_x_neg.*` files. This agrees with the prior
kinematic diagnosis rather than revealing a different failure mode.

The training distribution explains the next controlled hypothesis. Half of
episodes are nominal; pushed episodes draw force uniformly from zero to 5% of
body weight. The expected training force is therefore only 1.25% of body
weight, while acceptance always applies the 5% boundary. The next experiment
must directly exercise the hard end of that distribution while retaining a
nominal subset and using the earlier stable source. This is a distribution
hypothesis, not a claim that more training alone will solve recovery.

The previous [control diagnosis](STAND_PUSHX_RESULTS.md) does not support
ankle command saturation as the cause of initial backward balance loss.
The [recorded project comparison](SKILLS_REVIEW.md) remains the architectural
reference for subsequent work. Walk-to-stand and A-to-B are still pending.
