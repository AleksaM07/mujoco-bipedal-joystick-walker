# Stand recovery queue

The bounded overnight queue waits for the active `balance_v1` hard-push run to
finish its training, checkpoint gates, videos and all 25 robustness probes. It
does not start a duplicate. If any checkpoint passes the full five-case suite,
the queue stops immediately.

If the active run does not pass, the queue runs two 2M-step continuations from
the same stable pre-hard-push checkpoint, with LR `5e-5` and 60-second episodes:

1. `recovery_v2` with broad 0-5% body-weight pushes in 50% of episodes.
2. `recovery_v2` with hard 4-5% pushes in 75% of episodes.

`recovery_v2` preserves the dense speed, drift, posture and height feedback but
does not zero it while one foot is temporarily lifted. The separate acceptance
gate still requires both feet in contact and a one-second dwell. This tests the
specific hypothesis that `balance_v1` suppresses a learned corrective recovery
step after a backward impulse.

The queue writes `queue.json` and `summary.md` after every phase. Creating a
file named `STOP` in the queue directory prevents the next experiment from
starting; it never interrupts a training process in the middle of a run.
