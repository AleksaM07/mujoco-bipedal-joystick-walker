# Hard negative-X stand curriculum — 2026-11-09

The lower-rate continuation did not improve -X recovery and lost the confirmed
60-second hold. The next controlled test returns to the earlier stable source:
`stand_pushx_20261106_1m_lr3e-4` checkpoint 409,600. That policy passes the
60-second hold and +X/±Y probes and fails only -X at 6.94 s.

The previous training distribution makes half of episodes nominal and samples
active force uniformly from zero to 5% of body weight. Its expected force over
all episodes is 1.25%, while the fixed acceptance probe always uses 5%. This
test moves the active distribution to 4–5% in 75% of episodes. The remaining
25% preserve nominal standing. Pulse direction, thorax application point,
0.2-second duration, 10-second repetition, randomized 2–8 s phase, model,
physics, reward and acceptance suite stay unchanged.

The requested budget is 2M steps, with 60-second episodes, 64 environments,
seed 7 and learning rate `5e-5`. Actor, critic and observation normalizer are
restored; the optimizer starts fresh. The smaller learning rate and stable
source are intended to reduce loss of nominal stance while the harder
distribution tests whether insufficient exposure to the acceptance boundary
caused the unchanged -X failure. This is one combined curriculum test, not a
causal proof about learning rate.

Output: `runs/stand_pushx_hard_20261109_2m_lr5e-5`.

The runner must perform short gates for every saved checkpoint, render the
selected short-gate policy, and run the 60-second hold plus ±X/±Y probes for
every checkpoint. A candidate is accepted only if all five long cases pass and
the saved recovery motion is visually reviewed. No further training is queued.
