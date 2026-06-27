# BVH Walking Tiers

Generated from `cmu-mocap-index-text.txt` descriptions.

Recommended curriculum:

- Start with `tier1_forward_walk.txt` only.
- After stable walking, resume with tier1 + tier2.
- Keep tier3 and uneven terrain for later robustness experiments.
- Do not switch tiers automatically inside the env; use separate runs.

## tier1_forward_walk.txt

Count: 102

- `BVH_walking_animation/cmuconvert-max-01-09/02/02_01.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/02/02_02.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/05/05_01.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/06/06_01.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_01.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_02.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_03.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_06.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_07.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_08.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_09.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_10.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_11.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_01.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_02.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_03.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_06.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_08.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_09.bvh` - walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_10.bvh` - walk
- ... 82 more

## tier2_walk_variations.txt

Count: 166

- `BVH_walking_animation/cmuconvert-max-01-09/07/07_04.bvh` - slow walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_05.bvh` - slow walk
- `BVH_walking_animation/cmuconvert-max-01-09/07/07_12.bvh` - brisk walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_04.bvh` - slow walk
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_05.bvh` - walk/stride
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_07.bvh` - walk, exaggerated stride
- `BVH_walking_animation/cmuconvert-max-01-09/08/08_11.bvh` - slow walk/stride
- `BVH_walking_animation/cmuconvert-max-01-09/09/09_12.bvh` - navigate - walk forward, backward, sideways
- `BVH_walking_animation/cmuconvert-max-113-128/120/120_19.bvh` - Walk slow
- `BVH_walking_animation/cmuconvert-max-113-128/127/127_04.bvh` - Walk to Run
- `BVH_walking_animation/cmuconvert-max-131-135/131/131_01.bvh` - Start Walk Stop
- `BVH_walking_animation/cmuconvert-max-131-135/131/131_02.bvh` - Start Walk Stop
- `BVH_walking_animation/cmuconvert-max-131-135/131/131_09.bvh` - Start Walk Left
- `BVH_walking_animation/cmuconvert-max-131-135/131/131_10.bvh` - Start Walk Left
- `BVH_walking_animation/cmuconvert-max-131-135/131/131_11.bvh` - Start Walk Left
- `BVH_walking_animation/cmuconvert-max-131-135/131/131_12.bvh` - Start Walk Right
- `BVH_walking_animation/cmuconvert-max-131-135/131/131_13.bvh` - Start Walk Right
- `BVH_walking_animation/cmuconvert-max-131-135/131/131_14.bvh` - Start Walk Right
- `BVH_walking_animation/cmuconvert-max-131-135/132/132_17.bvh` - Walk Fast
- `BVH_walking_animation/cmuconvert-max-131-135/132/132_18.bvh` - Walk Fast
- ... 146 more

## tier3_style_or_complex_walks.txt

Count: 260

- `BVH_walking_animation/cmuconvert-max-102-111/102/102_16.bvh` - WalkEvasiveLeft
- `BVH_walking_animation/cmuconvert-max-102-111/102/102_17.bvh` - WalkEvasiveRight
- `BVH_walking_animation/cmuconvert-max-102-111/103/103_07.bvh` - Casual Walk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_02.bvh` - neutral Male walk, exact footfalls
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_03.bvh` - Frankenstein male walk, exact footfalls
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_11.bvh` - HappyGoLuckWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_12.bvh` - ExcitedWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_13.bvh` - StumbleWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_14.bvh` - HappyStartWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_16.bvh` - SpasticWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_19.bvh` - CasualWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_22.bvh` - RegularWalk8
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_27.bvh` - RegularWalkRightAngleTurns
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_28.bvh` - SternWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_29.bvh` - SternWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_30.bvh` - SternWalk8
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_32.bvh` - SternWalkRightAngleTurns
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_35.bvh` - SlowWalk
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_36.bvh` - SlowWalk8
- `BVH_walking_animation/cmuconvert-max-102-111/104/104_39.bvh` - SlowWalkRightAngleTurns
- ... 240 more

## uneven_terrain_walks.txt

Count: 52

- `BVH_walking_animation/cmuconvert-max-01-09/03/03_01.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-01-09/03/03_02.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-01-09/03/03_03.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-01-09/03/03_04.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-102-111/111/111_27.bvh` - Walking up stairs
- `BVH_walking_animation/cmuconvert-max-113-128/113/113_19.bvh` - Walking up and down stairs
- `BVH_walking_animation/cmuconvert-max-113-128/113/113_20.bvh` - Walk up and down stairs
- `BVH_walking_animation/cmuconvert-max-113-128/114/114_07.bvh` - Walking up and down stairs
- `BVH_walking_animation/cmuconvert-max-113-128/114/114_08.bvh` - Walking up and down stairs
- `BVH_walking_animation/cmuconvert-max-113-128/114/114_09.bvh` - Walking up and down stairs
- `BVH_walking_animation/cmuconvert-max-141-144/143/143_17.bvh` - Walk Up Stairs And Over
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_01.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_04.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_05.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_06.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_07.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_08.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_10.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_11.bvh` - walk on uneven terrain
- `BVH_walking_animation/cmuconvert-max-35-39/36/36_12.bvh` - walk on uneven terrain
- ... 32 more
