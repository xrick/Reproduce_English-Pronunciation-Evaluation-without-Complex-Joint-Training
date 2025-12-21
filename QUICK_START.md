# Quick Start: Fix Remote Training

Your remote training failed because **LoRA parameters are frozen**. Here's the complete fix.

---

## 🎯 What You Need to Do

### Option 1: Automated Setup (Recommended) ⭐

**Step 1: Edit transfer script** (on Mac, takes 30 seconds)
```bash
nano TRANSFER_TO_REMOTE.sh

# Change this line (line 8):
REMOTE_HOST="your-remote-host"

# To your actual remote hostname, for example:
REMOTE_HOST="192.168.1.100"
# or
REMOTE_HOST="gpu-server.university.edu"

# Save: Ctrl+X, Y, Enter
```

**Step 2: Transfer files** (on Mac, takes 1 minute)
```bash
./TRANSFER_TO_REMOTE.sh
```

**Step 3: Run setup** (on remote, takes 2 minutes)
```bash
ssh user@remote
cd /datas/store162/xrick/prjs/Reproduce_English_Pronunciation_Evaluation
bash REMOTE_SETUP_COMPLETE.sh
```

**Step 4: Start training** (on remote)
```bash
source venv/bin/activate
python src/train_single_config_remote.py --config paper_r64 --gpus 0
```

---

### Option 2: Manual Setup (If Automated Fails)

**Transfer files manually** (on Mac)
```bash
cd /Users/xrickliao/WorkSpaces/ResearchCodes/Reproduce_English_Pronunciation_Evaluation_without_Complex_Joint_Training

scp fix_lora_training.py user@remote:/path/to/project/
scp fix_bf16_to_fp16.py user@remote:/path/to/project/
scp patch_disable_amp.py user@remote:/path/to/project/
scp fix_processor_save.py user@remote:/path/to/project/
scp verify_all_fixes.py user@remote:/path/to/project/
```

**Apply fixes** (on remote)
```bash
ssh user@remote
cd /path/to/project

# Fix 1: BF16 → FP16
python fix_bf16_to_fp16.py

# Fix 2: Disable AMP
python patch_disable_amp.py

# Fix 3: Enable LoRA (CRITICAL)
python fix_lora_training.py

# Fix 4: Processor save (optional)
python fix_processor_save.py

# Verify
python verify_all_fixes.py
```

**Clean and restart** (on remote)
```bash
rm -rf src/output/paper_r64/

source venv/bin/activate
python src/train_single_config_remote.py --config paper_r64 --gpus 0
```

---

## ✅ How to Know It's Working

### During Model Loading
You **MUST** see:
```
🔧 Applying LoRA configuration to model...
✅ LoRA configuration applied - parameters are now trainable
可訓練參數: 200,000,000 (3.5%)
```

**If you see `可訓練參數: 0 (0.0%)`** → Fix NOT applied, try again!

### During Training
You **MUST** see:
```
{'loss': 6.98, 'epoch': 0.26}   ← Starts around 7.0
{'loss': 6.42, 'epoch': 0.51}   ← Decreases
{'loss': 5.89, 'epoch': 0.77}   ← Continues decreasing
...
{'loss': 2.85, 'epoch': 3.0}    ← Ends around 2.5-3.0
```

**If you see `loss: 10.6...`** → Still broken, fixes not applied!

You **MUST NOT** see:
```
UserWarning: None of the inputs have requires_grad=True
```

---

## 📋 Complete File Checklist

Files that should exist in your project root:

**Fix Scripts** (apply changes):
- [x] `fix_lora_training.py` - Enable LoRA training (CRITICAL)
- [x] `fix_bf16_to_fp16.py` - FP16 for TITAN RTX
- [x] `patch_disable_amp.py` - Disable AMP
- [x] `fix_processor_save.py` - Processor save error

**Utility Scripts**:
- [x] `verify_all_fixes.py` - Verify all fixes applied
- [x] `REMOTE_SETUP_COMPLETE.sh` - Automated setup
- [x] `TRANSFER_TO_REMOTE.sh` - Transfer from Mac

**Documentation**:
- [x] `URGENT_REMOTE_FIXES.md` - Detailed troubleshooting
- [x] `QUICK_START.md` - This file

---

## 🔧 Verification Commands

**On remote machine, run these to verify fixes**:

```bash
# Check 1: LoRA training enabled (MOST CRITICAL)
grep "get_peft_model" src/model_utility_configs.py
# Should show: model = get_peft_model(model, peft_config)

# Check 2: FP16 (not BF16)
grep "torch.bfloat16" src/model_utility_configs.py
# Should show: NOTHING (or only in comments)

# Check 3: AMP disabled
grep '"fp16": False' src/train_single_config_remote.py
# Should show: "fp16": False,  # Disabled AMP

# All checks passed?
python verify_all_fixes.py
# Should show: ✅ ALL FIXES VERIFIED - READY TO TRAIN
```

---

## 🚨 Troubleshooting

### "REMOTE_SETUP_COMPLETE.sh not found"

The script exists on Mac at:
```
/Users/xrickliao/WorkSpaces/ResearchCodes/Reproduce_English_Pronunciation_Evaluation_without_Complex_Joint_Training/REMOTE_SETUP_COMPLETE.sh
```

You need to **transfer it to remote** first using `TRANSFER_TO_REMOTE.sh`

### "Permission denied" when running .sh

```bash
chmod +x REMOTE_SETUP_COMPLETE.sh
bash REMOTE_SETUP_COMPLETE.sh
```

### Transfer script asks for password repeatedly

Set up SSH key authentication:
```bash
# On Mac
ssh-copy-id user@remote
```

### Still getting loss=10.6 after fixes

1. Verify fix was applied:
```bash
grep "get_peft_model" src/model_utility_configs.py
```

2. Check model loading output for:
```
可訓練參數: 200,000,000 (3.5%)
```

3. If still 0%, fix wasn't applied - try manual method

---

## 📊 Expected Timeline

| Step | Time |
|------|------|
| Edit TRANSFER_TO_REMOTE.sh | 30 seconds |
| Transfer files to remote | 1 minute |
| Run REMOTE_SETUP_COMPLETE.sh | 2 minutes |
| Clean old data | 10 seconds |
| **Training (3 epochs)** | **3-4 hours** |

**Total**: ~10 minutes setup + 3-4 hours training

---

## 🎯 Summary

**Problem**: Loss stuck at 10.6, LoRA parameters frozen
**Cause**: Fixes created on Mac but not applied on remote
**Solution**: Transfer and run fix scripts on remote machine
**Time**: 10 minutes to fix, 3-4 hours to train
**Result**: Working training with decreasing loss (7.0 → 2.5)

---

**Quick commands**:
```bash
# On Mac
nano TRANSFER_TO_REMOTE.sh  # Edit REMOTE_HOST
./TRANSFER_TO_REMOTE.sh     # Transfer files

# On Remote
bash REMOTE_SETUP_COMPLETE.sh  # Apply all fixes
source venv/bin/activate
python src/train_single_config_remote.py --config paper_r64 --gpus 0
```
