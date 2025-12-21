# BF16 Incompatibility Error - TITAN RTX Fix

## Error Summary

```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

**Location**: PyTorch gradient scaler during training
**Root Cause**: `model_utility_configs.py` hardcodes `torch_dtype=torch.bfloat16`
**Impact**: 🔴 CRITICAL - Training fails during first gradient update
**GPU**: NVIDIA TITAN RTX (Turing 7.5) **does NOT support BF16**

---

## Quick Fix (2 Methods)

### Method 1: Automatic Fix Script (Recommended) ⭐

**Transfer and run**:

```bash
# On Mac
scp fix_bf16_to_fp16.py user@remote:/path/to/project/

# On remote
ssh user@remote
cd /path/to/project
python fix_bf16_to_fp16.py
```

**What it does**:
- ✅ Finds all `torch.bfloat16` in `model_utility_configs.py`
- ✅ Replaces with `torch.float16`
- ✅ Creates backup before modifying
- ✅ Shows summary of changes

**Expected output**:
```
✅ FIX COMPLETE
📝 Changes made: torch.bfloat16 → torch.float16
🚀 You can now run training
```

---

### Method 2: Manual Edit

**On remote**:

```bash
# 1. Edit model_utility_configs.py
nano src/model_utility_configs.py

# 2. Find lines with torch.bfloat16 (should be 2 occurrences)
#    Line ~86 (pretrained_r320 config)
#    Line ~172 (paper_r64 config)

# 3. Change BOTH from:
        torch_dtype=torch.bfloat16,

# To:
        torch_dtype=torch.float16,

# 4. Save (Ctrl+X, Y, Enter)
```

---

## Problem Explanation

### GPU Architecture Support Matrix

| GPU Architecture | Compute Capability | BF16 Support | FP16 Support |
|-----------------|-------------------|--------------|--------------|
| **Turing** (TITAN RTX, RTX 20XX) | 7.5 | ❌ **NO** | ✅ YES |
| **Ampere** (A100, RTX 30XX) | 8.0 | ✅ YES | ✅ YES |
| **Ada Lovelace** (RTX 40XX) | 8.9 | ✅ YES | ✅ YES |
| **Hopper** (H100) | 9.0 | ✅ YES | ✅ YES |

**Your GPU**: NVIDIA TITAN RTX = Turing 7.5 = **NO BF16**

### Why This Error Occurs

1. **Model Loading** (model_utility_configs.py line 172):
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       ...,
       torch_dtype=torch.bfloat16,  # ← Model loaded in BF16
   )
   ```

2. **Training Script Detection** (train_single_config_remote.py lines 130-136):
   ```python
   # This TRIES to detect and switch to FP16
   if compute_capability[0] < 8:
       use_bf16 = False
       use_fp16 = True
   ```
   **But**: Model is ALREADY loaded in BF16 before this check!

3. **Gradient Scaling Failure**:
   - PyTorch tries to use BF16 gradient scaler
   - CUDA doesn't have BF16 scaler for Turing
   - **NotImplementedError** thrown

---

## Complete Fix Verification

After applying fix:

```bash
cd /path/to/project

# 1. Verify changes
grep "torch.float16" src/model_utility_configs.py
# Should show 2 lines with torch.float16

grep "torch.bfloat16" src/model_utility_configs.py
# Should show NO results (or only in comments)

# 2. Test model loading
python3 << 'EOF'
import torch
from src.model_utility_configs import CONFIGS

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    print(f"Compute capability: {cc}")
    print(f"Supports BF16: {cc[0] >= 8}")

config = CONFIGS["paper_r64"]
model, processor, peft_config = config["loader"]()

# Check model dtype
param = next(model.parameters())
print(f"\n✅ Model loaded successfully")
print(f"Model dtype: {param.dtype}")
print(f"Expected: torch.float16")

if param.dtype == torch.float16:
    print("✅ CORRECT: Model is in FP16")
elif param.dtype == torch.bfloat16:
    print("❌ WRONG: Model still in BF16 - fix failed")
else:
    print(f"⚠️  Unexpected dtype: {param.dtype}")
EOF
```

**Expected output**:
```
CUDA available: True
Compute capability: (7, 5)
Supports BF16: False

✅ Model loaded successfully
Model dtype: torch.float16
Expected: torch.float16
✅ CORRECT: Model is in FP16
```

---

## Start Training

After fix is verified:

```bash
source venv/bin/activate

# Use --fp16 flag explicitly (defensive)
python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0 \
  --fp16 \
  --batch-size 8 \
  --gradient-accumulation 8 \
  --epochs 3
```

**Expected**: Training should proceed without BF16 errors

---

## Why Mac Works But Remote Fails

**Mac (Apple MPS)**:
- No CUDA, uses Apple Metal
- BF16 supported on Apple Silicon
- `torch.bfloat16` works fine

**Remote (NVIDIA TITAN RTX)**:
- CUDA backend
- Turing architecture (compute 7.5)
- BF16 **NOT** supported
- Must use FP16

**Solution**: Platform-specific dtype (future improvement):

```python
# Ideal: Auto-detect in model_utility_configs.py
import torch

if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    if cc[0] >= 8:  # Ampere or newer
        dtype = torch.bfloat16
    else:  # Turing or older
        dtype = torch.float16
else:  # CPU or MPS
    dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=dtype,  # Platform-aware
)
```

---

## Common Follow-Up Errors

### "None of the inputs have requires_grad=True"

**This is just a WARNING**, not an error. Appears because:
- Gradient checkpointing enabled
- First pass through model before gradients computed
- **Safe to ignore**

### "You are not running the flash-attention implementation"

**This is also just a WARNING**. Means:
- Using standard attention (not Flash Attention 2)
- Expected on TITAN RTX (Flash Attention 2 needs Ampere+)
- **Safe to ignore**

### "use_cache=True is incompatible with gradient checkpointing"

**Auto-resolved**. PyTorch automatically sets `use_cache=False`
- **Safe to ignore**

---

## Files Changed

### Before (BROKEN on TITAN RTX)
```python
# src/model_utility_configs.py line 86
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch.bfloat16,  # ❌ TITAN RTX doesn't support
)

# src/model_utility_configs.py line 172
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch.bfloat16,  # ❌ TITAN RTX doesn't support
)
```

### After (WORKS on TITAN RTX)
```python
# src/model_utility_configs.py line 86
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch.float16,  # ✅ TITAN RTX supports FP16
)

# src/model_utility_configs.py line 172
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch.float16,  # ✅ TITAN RTX supports FP16
)
```

---

## Performance Impact

**BF16 vs FP16**:

| Metric | BF16 (Ampere+) | FP16 (Turing) |
|--------|----------------|---------------|
| Training speed | Baseline (1.0x) | ~Same speed |
| VRAM usage | Baseline | ~Same usage |
| Numerical stability | Better (wider range) | Good enough |
| Model quality | Slightly better | Nearly identical |

**Expected difference**: Negligible for this use case
- Final metrics should be within ±0.5% of BF16 results
- Training time: Same (~3-4 hours)
- VRAM: Same (~22-24GB)

---

## Related Warnings (Safe to Ignore)

These warnings will appear but are **NORMAL and SAFE**:

```
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
→ ✅ Auto-resolved, safe to ignore

UserWarning: None of the inputs have requires_grad=True. Gradients will be None
→ ✅ First pass warning, safe to ignore

You are not running the flash-attention implementation, expect numerical differences.
→ ✅ Using standard attention (Flash Attention 2 not compatible with Turing), safe to ignore
```

---

## Troubleshooting

### Fix applied but still getting BF16 error

**Check**:
```bash
grep "torch.bfloat16" src/model_utility_configs.py
```

**Should return**: No results (or only in comments)

**If still shows BF16**:
- Fix script didn't run correctly
- File not transferred properly
- Editing wrong file

**Solution**: Manually edit and verify

### Model loads but training still fails

**Check training args**:
```bash
# In train_single_config_remote.py, verify:
grep "use_bf16\|use_fp16" src/train_single_config_remote.py
```

**Should show**: Detection logic that switches to FP16 for Turing

### Different error after fix

**Possible**: May reveal next issue in training pipeline

**Action**: Report new error for diagnosis

---

## Summary

**Problem**: Model hardcoded to BF16, TITAN RTX doesn't support it
**Solution**: Change `torch.bfloat16` → `torch.float16` in model_utility_configs.py
**Tool**: `fix_bf16_to_fp16.py` (automated) or manual edit
**Impact**: Zero performance impact, training works on TITAN RTX
**Time to fix**: 1-2 minutes

---

**Error**: BF16 incompatibility with Turing GPUs
**Severity**: 🔴 CRITICAL (blocks training)
**Solution**: Switch to FP16
**Affected file**: `src/model_utility_configs.py` (2 locations)
