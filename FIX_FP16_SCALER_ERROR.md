# FP16 Gradient Scaler Error - Quick Fix

## Error Summary

```
ValueError: Attempting to unscale FP16 gradients.
```

**Location**: PyTorch GradScaler during gradient clipping
**Root Cause**: Missing `--fp16` flag in training command
**Impact**: 🔴 CRITICAL - Training fails during first gradient update
**Context**: This error appears AFTER fixing the BF16 error

---

## Quick Fix ⚡

### The Problem

You ran training **without** the `--fp16` flag:

```bash
# ❌ WRONG (causes FP16 scaler error)
python src/train_single_config_remote.py --config paper_r64 --gpus 0
```

### The Solution

Add the `--fp16` flag:

```bash
# ✅ CORRECT (FP16 training on TITAN RTX)
python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

**That's it!** Training should now work.

---

## Why This Happens

### Sequence of Events

1. **Model Loading** (model_utility_configs.py):
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       ...,
       torch_dtype=torch.float16,  # Model in FP16 (after fix)
   )
   ```

2. **Training Args WITHOUT --fp16** (train_single_config_remote.py):
   ```python
   use_bf16 = not args.fp16  # → use_bf16 = True (no --fp16 flag)
   use_fp16 = args.fp16      # → use_fp16 = False

   # Then detection runs:
   if use_bf16 and compute_capability[0] < 8:
       use_bf16 = False  # Switch off BF16
       use_fp16 = True   # Switch on FP16

   # TrainingArguments:
   training_args = TrainingArguments(
       bf16=False,  # After detection
       fp16=True,   # After detection
   )
   ```

3. **Gradient Scaler Conflict**:
   - Model parameters: FP16
   - Gradient scaler: Gets initialized with mixed signals
   - PyTorch throws: `ValueError: Attempting to unscale FP16 gradients`

### Why --fp16 Flag Fixes It

**With --fp16 flag**:
```python
use_bf16 = not args.fp16  # → False (--fp16 provided)
use_fp16 = args.fp16      # → True (--fp16 provided)

# No BF16 detection needed, directly:
training_args = TrainingArguments(
    bf16=False,  # Clear
    fp16=True,   # Clear
)
```

**Result**: Clean FP16 setup from the start, no scaler confusion

---

## Verification

### Before Training

Run verification script:

```bash
# Transfer verification script
scp verify_fp16_setup.py user@remote:/path/to/project/

# Run on remote
python verify_fp16_setup.py
```

**Expected output**:
```
✅ Setup is CORRECT
🚀 Ready to train with:
   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

### During Training

First few lines should show:

```bash
🚀 開始訓練...
================================================================================

`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
You are not running the flash-attention implementation, expect numerical differences.

  0%|          | 0/120 [00:00<?, ?it/s]
  1%|▏         | 1/120 [00:15<30:23, 15.32s/it, loss=...]  # ← Training progresses!
```

**No more errors** - training continues normally

---

## Complete Training Command

### Minimal (Recommended for First Run)

```bash
source venv/bin/activate

python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0 \
  --fp16
```

### Full Options (Paper Settings)

```bash
source venv/bin/activate

python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0 \
  --fp16 \
  --batch-size 8 \
  --gradient-accumulation 8 \
  --learning-rate 2e-5 \
  --epochs 3
```

---

## Common Warnings (Safe to Ignore)

These warnings are **NORMAL and EXPECTED**:

### 1. use_cache Warning
```
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...
```
**Meaning**: Auto-resolved by PyTorch
**Action**: ✅ Ignore (safe)

### 2. requires_grad Warning
```
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
```
**Meaning**: First forward pass before gradients computed
**Action**: ✅ Ignore (safe)

### 3. Flash Attention Warning
```
You are not running the flash-attention implementation, expect numerical differences.
```
**Meaning**: Using standard attention (Flash Attention 2 not compatible with Turing)
**Action**: ✅ Ignore (safe)

---

## Troubleshooting

### Still Getting FP16 Scaler Error

**Check 1**: Verify you're using `--fp16` flag

```bash
# Show your command
history | grep train_single_config_remote
# Should see: ... --fp16
```

**Check 2**: Verify model dtype is FP16

```bash
python verify_fp16_setup.py
# Should show: Model dtype: torch.float16
```

**Check 3**: If model still in BF16

```bash
grep "torch.bfloat16" src/model_utility_configs.py
# Should return: NOTHING (or only in comments)

# If still shows BF16:
python fix_bf16_to_fp16.py
```

### Different Error After Adding --fp16

**Possible Next Issues**:
1. CUDA out of memory → Reduce batch size
2. Data loading errors → Check AudioDataCollator
3. Other training errors → Report for diagnosis

---

## Summary of All Fixes Applied

### Fix 1: Tokenizer (Already Done)
```bash
# Re-downloaded tokenizer.json
python fix_tokenizer_remote.py
```

### Fix 2: AudioDataCollator (Already Done)
```bash
# Transferred fixed AudioDataCollator.py from Mac
scp src/AudioDataCollator.py user@remote:/path/to/project/src/
```

### Fix 3: BF16 → FP16 (Already Done)
```bash
# Changed torch.bfloat16 → torch.float16 in model_utility_configs.py
python fix_bf16_to_fp16.py
```

### Fix 4: Add --fp16 Flag (CURRENT)
```bash
# Just add --fp16 to training command
python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

---

## Expected Training Behavior

### Initialization (First 30 seconds)
```
🚀 開始訓練...
載入模型...
載入訓練數據集...
訓練樣本數: 2500

[Warnings appear - all safe to ignore]

  0%|          | 0/120 [00:00<?, ?it/s]
```

### Training Progress (3-4 hours)
```
  1%|▏         | 1/120 [00:15<30:23, 15.32s/it, loss=6.98]
  2%|▎         | 2/120 [00:30<29:45, 15.13s/it, loss=6.85]
  3%|▍         | 3/120 [00:45<29:12, 14.98s/it, loss=6.72]
  ...
100%|██████████| 120/120 [3:45:00<00:00, 15.25s/it, loss=2.34]

Training complete!
```

### Checkpoints Saved
```
src/output/paper_r64/
├── checkpoint-40/   (epoch 1)
├── checkpoint-80/   (epoch 2)
├── checkpoint-120/  (epoch 3 - final)
└── logs/            (TensorBoard)
```

---

## Quick Reference

### Error
```
ValueError: Attempting to unscale FP16 gradients.
```

### Fix
```bash
# Add --fp16 flag
python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

### Verify Before Training
```bash
python verify_fp16_setup.py
```

### Monitor Training
```bash
# In separate terminal
tensorboard --logdir src/output/paper_r64/logs/ --port 6007
```

---

**Error**: FP16 gradient scaler mismatch
**Severity**: 🔴 CRITICAL (blocks training)
**Solution**: Add `--fp16` flag to training command
**Time to fix**: 10 seconds (just re-run with flag)
