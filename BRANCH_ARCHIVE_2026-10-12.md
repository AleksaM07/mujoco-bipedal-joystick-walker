# Local experimental branch cleanup

On 2026-10-12 the user requested removal of these three branches after retaining
their insights. Only these **local branch refs** were deleted:

| Branch | Preserved tip |
| --- | --- |
| `experiment/main-modern-bvh` | `a6ca23d1e8a5cb0ca1ea8710631a83a11dcbaf20` |
| `experiment/v21-stable-baseline` | `cd6557c4678cb6e3f884b5f6531e25b803d922b4` |
| `backup/with-7f1294e-before-test` | `de5cc52863dcbd1209231e01c88392849294794a` |

Their findings and provenance remain in `PROJECT_DIAGNOSIS_2026-10-11.md` and
`PROJECT_STATUS_MAIN_MODERN_BVH.txt`. Recovery phases 1-3 incorporate the verified
timing, measurement and evaluator lessons with regression tests; they do not
blindly merge old curricula, reward experiments or unverified default settings.
Unused code can still be recovered from the archived tips.

Before deleting any ref, all three were archived to the local, Git-ignored file
`artifacts/branch_archive_20261012.bundle`. `git bundle verify` succeeded and
`git bundle list-heads` matched all three full hashes above.

This is an **incremental bundle**, requiring the retained main history through
`33eaa1fb76f2ffb0fb8a821deb9cad27f3989426`. It is not a standalone backup of the
entire repository. SHA256:

```text
16a6da555b3f46fc102b09100905b3053f8032322d2939dabd5d8074297bf86e
```

Restore any branch in this repository, for example:

```bash
git bundle verify artifacts/branch_archive_20261012.bundle
git fetch artifacts/branch_archive_20261012.bundle refs/heads/experiment/v21-stable-baseline:refs/heads/experiment/v21-stable-baseline
```

Substitute the exact archived name in both sides of the refspec to restore
either other branch. This recreates the branch without checking it out.

Remote branches were **not deleted** and no push was performed. `main`,
`project_v2_refrences_only`, `lets_call_it_dave`, the active
`recovery/project-v2-imitation`, the diagnostic stash and all checkpoints remain.
