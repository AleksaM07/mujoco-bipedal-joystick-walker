# Project References And Provenance

This file is the human-readable overview.  Precise per-implementation provenance
is tracked in `references/registry.yaml`; runtime code should only carry short
`REF` and `TYPE` comments that point back to that registry.

## DEEPMIMIC2018

**DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based
Character Skills**

Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne, 2018.

URL: https://xbpeng.github.io/projects/DeepMimic/index.html

Used for:

- the imitation-reward decomposition direction: pose/orientation, velocity,
  end-effectors, root, and center-of-mass terms;
- the general reference-motion tracking direction.

Usage:

- adapted, not a full reproduction;
- concrete constants currently used by this project are traced to the
  DeepMimic reference implementation entry `DEEPMIMIC2018-CODE-IMITATION-REWARD`
  in `references/registry.yaml`;
- paper section and equation numbers have not been verified for the current
  implementation.

Limitations:

- this project uses a MuJoCo humanoid and BVH-derived retargeted joint targets,
  while the original implementation uses its own character/motion format;
- the current BVH path is an FK-feature adaptation, not full MimicKit/GMR
  retargeting.

## DEEPMIMIC2018-CODE

Repository: https://github.com/xbpeng/DeepMimic

Local reference checkout commit: `1f915c52fcd4b95b5f5f15b759ae91bd81e9a801`

Used for:

- implementation-level imitation reward weights and exponential reward shape
  in `biomechanics_env.py`.

Usage:

- adapted from Bullet/DeepMimic character code to MuJoCo/MJX data structures.

## ALEKSAM07-BULLET-WARP

Repository: https://github.com/AleksaM07/bipedal-humanoid-bullet-avoidance

Local reference checkout commit: `0cd3da8135a7380dfb09c81ff29cba774cb5fdcf`

Used for:

- MJX-Warp backend validation and conversion helpers;
- Warp contact-buffer/world-id handling;
- Warp capacity defaults.

Usage:

- code-derived implementation adapted into this repository's simpler training
  and environment structure.

Limitations:

- this is a project/reference-code source, not a scientific paper.

## PROJECT-LOCAL

Used for:

- model-calibrated thresholds tied to this MuJoCo humanoid XML;
- engineering defaults such as XML hash guards, logging metrics, and software
  package pins;
- experimentally selected default parallelism (`12288`) based on prior local
  GPU runs.

Limitations:

- where no paper or external implementation has been verified, the registry
  marks the item as `MODEL_CALIBRATED`, `EXPERIMENTALLY_SELECTED`, or
  `ENGINEERING_DEFAULT` instead of inventing a citation.
