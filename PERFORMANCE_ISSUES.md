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

## How to Profile

Run with Python profiler to identify exact hot spots:
```bash
python -m cProfile -s cumtime train.py --debug-run
```

Or use `py-spy` for live profiling:
```bash
py-spy record -o profile.svg -- python train.py --debug-run
```
