# Performance Issues Found

## Critical Issues (Highest Impact)

### 1. **XML Generation & Validation Happens Every Training Run**
**Location:** [biomechanics_model.py](biomechanics_model.py#L73-L95) & [train.py](train.py#L450)

**Problem:** 
- `build_trainable_scene_xml()` generates a full XML model every time the environment initializes
- `generate_base_human_xml()` calls an expensive external generator
- `load_generator()` dynamically imports a module every single time
- `validate_xml()` loads the entire MuJoCo model just to check syntax

**Impact:** This adds 5-30 seconds per training run initialization (XML generation + validation)

**Fix:**
- Cache generated XMLs based on HumanSpec hash
- Load and cache the generator module once instead of per-call
- Skip XML validation if file already exists and is valid
- Use a simple syntax check instead of full model loading for validation

---

### 2. **Domain Randomization Using vmap**
**Location:** [biomechanics_env.py](biomechanics_env.py#L330-L380)

**Problem:**
- `domain_randomize()` uses `jax.vmap` every step which might be inefficient
- Creates many intermediate arrays that aren't needed
- `tree_replace` operations are expensive

**Impact:** Slows down every training iteration during domain randomization

**Fix:**
- Pre-compile the randomization function with `jax.jit`
- Reduce intermediate array allocations
- Consider batching randomization differently

---

### 3. **Inefficient Observation Construction**
**Location:** [biomechanics_env.py](biomechanics_env.py#L198-L210)

**Problem:**
- `_get_obs()` uses `jp.hstack()` which creates multiple intermediate arrays
- This is called thousands of times during training

```python
obs = jp.hstack([
    data.qvel[:3],
    data.qvel[3:6],
    info["command"],
    joint_pos,
    joint_vel,
    info["last_action"],
])
```

**Impact:** Memory allocation overhead in hot loop

**Fix:**
- Pre-allocate observation array and copy directly
- Use `jp.concatenate()` instead of hstack where possible

---

## Moderate Issues

### 4. **Large Default Batch Sizes**
**Location:** [train.py](train.py#L177)

**Problem:**
```python
num_envs: int = 1024
batch_size: int = 256
```

These are quite large and may cause GPU memory pressure

**Fix:**
- Start with smaller batch sizes if OOM occurs
- Use gradient accumulation instead

---

### 5. **No Caching of XML Paths**
**Location:** [train.py](train.py#L451)

**Problem:**
- Each environment instance regenerates XML files unnecessarily
- Multiple trainer instances can race to generate the same files

**Fix:**
- Check if desired XML already exists before regenerating

---

## Recommendations (Priority Order)

1. **HIGH:** Add XML caching and remove redundant validation
2. **HIGH:** Pre-compile domain randomization 
3. **MEDIUM:** Optimize observation construction with pre-allocation
4. **MEDIUM:** Cache the generator module
5. **LOW:** Profile training to identify other bottlenecks

---

## Current Status After BVH Reference Review - 2026-06-21

The current bottleneck is not only Python startup overhead. The larger practical
problem is experiment latency: one full PPO run can take most of a day, so a bad
reward or reference design costs too much time.

Updated priority:

1. **Do short gate runs first:** use `5M-8M` timesteps with `4-5` evals before
   committing to `60M-80M`.
2. **Change one risky system at a time:** after the BVH mimic reward rework,
   test reference-state initialization only in a short `2M` probe, not in a
   full-day run.
3. **Keep eval cheap:** use `--num-eval-envs 16` unless final reporting needs
   lower-noise evaluation.
4. **Use `tier1_debug_10.txt` for iteration:** do not train on the full BVH tier
   list until the small list visibly improves gait.
5. **Resume from the best compatible checkpoint:** for the active V10 line, the
   latest confirmed good checkpoint is:

```text
runs/biomechanics_noerfi_nodr_forward_ref_bvh_v10_18350_resume_forward_accurate_ppo_BiomechanicsHumanJoystickStandard_20260621_0254_80m_seed7_running/checkpoints/000053739520
```

Compatibility-only gate after the BVH mimic reward rework:

```powershell
python train.py --device gpu --command-profile forward --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --resume-from runs/biomechanics_noerfi_nodr_forward_ref_bvh_v10_18350_resume_forward_accurate_ppo_BiomechanicsHumanJoystickStandard_20260621_0254_80m_seed7_running/checkpoints/000053739520 --xml-path generated_models/human_male_180cm_75kg_standard_trainfast_v16.xml --legacy-action-prior --reference-phase-randomization --reference-state-init --timesteps 2000000 --num-envs 1024 --num-evals 4 --num-eval-envs 16 --no-erfi --no-domain-randomization --run-tag v10_bvh_mimic_rework_probe
```

Use the command above only to isolate the reward change on the old V10
checkpoint. It intentionally keeps `v16 + --legacy-action-prior`, so it does
not test the new Unitree-style guardrails.

Recommended real gate with v15-style stable sole boxes, Unitree-style
illegal-contact guardrails, and the softened BVH contact-gated reward:

```powershell
python train.py --device gpu --command-profile forward --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 3000000 --num-envs 1024 --num-evals 5 --num-eval-envs 16 --no-erfi --no-domain-randomization --diagnostic-rollout --diagnostic-rollout-steps 10 --run-tag v19_stable_sole_bvh_soft_gate
```

Latest gate result:

```text
runs/biomechanics_noerfi_nodr_forward_ref_bvh_v19_stable_sole_bvh_soft_gate_768_accurate_ppo_BiomechanicsHumanJoystickStandard_20260622_2151_3m_seed7_rew_m95p9755_best_m95p9755_s
```

This run was a useful failure: it trained for `3.19M` steps and improved from
`reward=-109` to `reward=-95`, but it still had `swing_r=1.0`,
`double_r=1.0`, and `done_tip=1.0`. The clearest diagnosis is that the BVH
regularization/contact penalties still dominated the positive mimic/progress
signal: final `bvh_reg=152.6`, while `bvh_core + bvh_stab + bvh_task +
bvh_boot` was only about `46.8`. After this run, the BVH-only regularization was
relaxed so contact penalties act as a bootstrap guide instead of blocking the
first learned steps.

If CUDA reports OOM during JAX allocation, keep the same experiment but reduce
parallelism. For Brax PPO, `batch_size * num_minibatches` must be divisible by
`num_envs`. With our default `num_minibatches=8`, `--num-envs 768
--batch-size 256` is invalid, because `256 * 8 = 2048` is not divisible by
`768`.

Medium fallback:

```powershell
python train.py --device gpu --command-profile forward --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 3000000 --num-envs 768 --batch-size 384 --num-evals 5 --num-eval-envs 16 --no-erfi --no-domain-randomization --diagnostic-rollout --diagnostic-rollout-steps 10 --run-tag v19_bvh_relaxed_reg_768
```

Safer low-memory fallback:

```powershell
python train.py --device gpu --command-profile forward --reference-gait bvh --reference-gait-list BVH_walking_animation/tier1_debug_10.txt --timesteps 3000000 --num-envs 512 --batch-size 256 --num-evals 5 --num-eval-envs 16 --no-erfi --no-domain-randomization --diagnostic-rollout --diagnostic-rollout-steps 10 --run-tag v19_bvh_relaxed_reg_512
```

This uses the default `trainfast_v19` XML:

- restored v15-style box sole contacts, because the successful zombie-walk
  models used `ngeom=19` with wide sole boxes;
- tighter leg ranges from `v17`;
- Unitree-style action prior, because there is no `--legacy-action-prior`;
- soft joint-limit cost in the outer 10% of each controlled joint range;
- hip/thigh/shank floor-contact penalty;
- pelvis floor-contact termination.

Do not use `--reference-state-init` for this first gate. Our current BVH
targets are still partial leg-joint retargets, not full GMR/root/body retargeted
motions. DeepMimic/DRLoco-style RSI is a good idea only after the reference
state is physically consistent.

Stop/continue rule:

- Stop early if reward rises but `raw`, `mimic_rew`, `bvh_core`, and `loco_q`
  stay low. That means the visible score is not real imitation progress.
- Stop early if `double_r` or `swing_r` stay near `1.0` after the first evals.
  That means the policy is still doing double-support sliding instead of
  stepping.
- Stop early if `clip_lo`, `clip_hi`, or `fall_rew` are high. That means PPO is
  mostly seeing clipped/fall rewards instead of a useful gradient.
- Stop early if `bvh_reg` dominates `bvh_core + bvh_stab + bvh_task`.
- Stop early if `double_ct` and `swing_ct` stay high through the first evals.
- Stop early if `joint_limit`, `illegal_ct`, or `done_illegal` are non-trivial
  after the initial noisy phase.
- Continue only if `mimic_rew`, `ref_foot`, `ref_root`, and `gated_prog` rise
  together with visible foot lifting.

Avoid on this gate:

- full `tier1_forward_walk.txt`
- ERFI/domain randomization
- long `60M-80M` runs before the first visual check

Re-enable `--reference-phase-randomization --reference-state-init` only after a
standing-reset gate can survive and visibly step, or after GMR-style full-body
retargeting is available.

---

## How to Profile

Run with Python profiler to identify exact hot spots:
```bash
python -m cProfile -s cumtime train.py --debug-run
```

Or use `py-spy` for live profiling:
```bash
py-spy record -o profile.svg -- python train.py --debug-run
```
