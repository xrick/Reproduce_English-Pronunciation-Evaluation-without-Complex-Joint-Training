# Sync Mac → Remote Checklist

Complete checklist to ensure remote machine has all fixes from Mac.

---

## 🔴 Critical Files That MUST Be Synced

These files were fixed on Mac and MUST be transferred to remote:

### 1. AudioDataCollator.py ⭐ CRITICAL

**Why**: Contains fix for Phi4MMProcessor API (tuple format)
**Error if missing**: `TypeError: got an unexpected keyword argument 'sampling_rate'`

**Transfer**:
```bash
scp src/AudioDataCollator.py user@remote:/path/to/project/src/
```

**Verify on remote**:
```bash
grep "audios.append((audio_array" src/AudioDataCollator.py
# Should show: audios.append((audio_array, f["sampling_rate"]))
```

---

### 2. model_utility_configs.py ⭐ CRITICAL

**Why**: Contains dual configuration (pretrained_r320 & paper_r64)
**Error if missing**: Config not found or wrong LoRA settings

**BUT**: Model path needs to be updated for remote (different from Mac)

**Action**:
```bash
# DO NOT directly copy (has Mac path)
# Instead, run update script on remote:
python update_model_path_remote.py
```

---

### 3. train_single_config_remote.py ⭐ NEW FILE

**Why**: CUDA-optimized training script for NVIDIA
**Error if missing**: Can't start remote training

**Transfer**:
```bash
scp src/train_single_config_remote.py user@remote:/path/to/project/src/
```

---

## 🟡 Important Support Files

### Helper Scripts

**Transfer these**:
```bash
scp update_model_path_remote.py user@remote:/path/to/project/
scp fix_tokenizer_remote.py user@remote:/path/to/project/
scp fix_remote_tokenizer_v2.sh user@remote:/path/to/project/
scp find_model_path.sh user@remote:/path/to/project/
```

---

## 🟢 Documentation (Optional but Helpful)

**Transfer documentation**:
```bash
scp README_TRAINING.md user@remote:/path/to/project/
scp REMOTE_FINAL_SETUP.md user@remote:/path/to/project/
scp REMOTE_QUICK_START.txt user@remote:/path/to/project/
scp TOKENIZER_ERROR_FIX.md user@remote:/path/to/project/
scp FIX_AUDIODATACOLLATOR_REMOTE.md user@remote:/path/to/project/
```

---

## Complete Sync Command (All at Once)

**Transfer all critical files in one command**:

```bash
# On Mac, from project root
rsync -avz --progress \
  src/AudioDataCollator.py \
  src/train_single_config_remote.py \
  src/data_utility.py \
  update_model_path_remote.py \
  fix_tokenizer_remote.py \
  fix_remote_tokenizer_v2.sh \
  find_model_path.sh \
  README_TRAINING.md \
  REMOTE_FINAL_SETUP.md \
  REMOTE_QUICK_START.txt \
  TOKENIZER_ERROR_FIX.md \
  FIX_AUDIODATACOLLATOR_REMOTE.md \
  user@remote:/path/to/project/
```

**Or sync entire src/ directory** (be careful with model_path):

```bash
# Backup remote config first
ssh user@remote "cp /path/to/project/src/model_utility_configs.py /path/to/project/src/model_utility_configs.py.remote.backup"

# Sync src/
rsync -avz --progress src/ user@remote:/path/to/project/src/

# Then update model path on remote
ssh user@remote "cd /path/to/project && python update_model_path_remote.py"
```

---

## Verification Checklist

After syncing, verify on remote:

### ✅ Check 1: AudioDataCollator Format

```bash
ssh user@remote "grep 'audios.append' /path/to/project/src/AudioDataCollator.py"
# Should show: audios.append((audio_array, f["sampling_rate"]))
```

### ✅ Check 2: Training Script Exists

```bash
ssh user@remote "ls -lh /path/to/project/src/train_single_config_remote.py"
# Should show file exists
```

### ✅ Check 3: Model Path Configured

```bash
ssh user@remote "grep 'model_path =' /path/to/project/src/model_utility_configs.py"
# Should NOT show Mac path (/Users/xrickliao/...)
# Should show remote path or online model
```

### ✅ Check 4: Helper Scripts

```bash
ssh user@remote "ls -1 /path/to/project/*.py /path/to/project/*.sh"
# Should list:
# - update_model_path_remote.py
# - fix_tokenizer_remote.py
# - fix_remote_tokenizer_v2.sh
# - find_model_path.sh
```

---

## Version Comparison Table

| File | Mac Version | Remote Needs | Critical |
|------|-------------|--------------|----------|
| AudioDataCollator.py | ✅ Fixed (tuple) | ⭐ Same as Mac | 🔴 YES |
| train_single_config_remote.py | ✅ New file | ⭐ Transfer | 🔴 YES |
| model_utility_configs.py | Mac path | ⚠️ Update path | 🔴 YES |
| data_utility.py | ✅ Working | Same as Mac | 🟡 Helpful |
| update_model_path_remote.py | ✅ New script | Transfer | 🟡 Helpful |
| fix_tokenizer_remote.py | ✅ New script | Transfer | 🟡 Helpful |

---

## Common Sync Mistakes

### ❌ Mistake 1: Copy model_utility_configs.py Directly

**Problem**: Mac path hardcoded in file
**Solution**: Run `update_model_path_remote.py` instead

### ❌ Mistake 2: Forget AudioDataCollator.py

**Problem**: Remote uses old version with wrong API
**Solution**: Always sync this file first

### ❌ Mistake 3: Sync venv/

**Problem**: Virtual environment is machine-specific
**Solution**: Never sync venv/, only sync source code

---

## Git-Based Sync (Recommended for Future)

**Setup once**:

```bash
# On Mac
git init  # if not already a git repo
git add src/*.py *.md *.sh *.txt
git commit -m "Initial commit with all fixes"

# Add remote (GitHub, GitLab, etc.)
git remote add origin <your-repo-url>
git push -u origin main
```

**Sync to remote**:

```bash
# On Remote
git clone <your-repo-url> /path/to/project
cd /path/to/project

# Configure model path for remote
python update_model_path_remote.py
```

**Future updates**:

```bash
# On Mac (after making changes)
git add src/AudioDataCollator.py  # or whatever changed
git commit -m "Fix: description"
git push

# On Remote
git pull
python update_model_path_remote.py  # if model_utility_configs.py changed
```

---

## Manual Sync Protocol (Step-by-Step)

### Phase 1: Critical Files (Must Do)

```bash
# 1. AudioDataCollator.py
scp src/AudioDataCollator.py user@remote:/path/to/project/src/

# 2. train_single_config_remote.py
scp src/train_single_config_remote.py user@remote:/path/to/project/src/

# 3. update_model_path_remote.py
scp update_model_path_remote.py user@remote:/path/to/project/

# 4. Configure model path on remote
ssh user@remote "cd /path/to/project && python update_model_path_remote.py"
```

### Phase 2: Helper Scripts (Recommended)

```bash
# 5. Tokenizer fix scripts
scp fix_tokenizer_remote.py user@remote:/path/to/project/
scp fix_remote_tokenizer_v2.sh user@remote:/path/to/project/
chmod +x fix_remote_tokenizer_v2.sh

# 6. Model finder
scp find_model_path.sh user@remote:/path/to/project/
chmod +x find_model_path.sh
```

### Phase 3: Documentation (Optional)

```bash
# 7. Quick reference docs
scp REMOTE_QUICK_START.txt user@remote:/path/to/project/
scp TOKENIZER_ERROR_FIX.md user@remote:/path/to/project/
scp FIX_AUDIODATACOLLATOR_REMOTE.md user@remote:/path/to/project/
```

### Phase 4: Verification

```bash
# 8. Run verification on remote
ssh user@remote "cd /path/to/project && bash" << 'EOF'
echo "Checking AudioDataCollator..."
grep "audios.append((audio_array" src/AudioDataCollator.py && echo "✅ AudioDataCollator OK"

echo "Checking training script..."
test -f src/train_single_config_remote.py && echo "✅ Training script exists"

echo "Checking model path..."
grep -q "/Users/xrickliao" src/model_utility_configs.py && echo "⚠️ Mac path still present" || echo "✅ Model path updated"
EOF
```

---

## Quick Fix for Current Error

**Immediate action for your current sampling_rate error**:

```bash
# Transfer fixed AudioDataCollator.py
scp src/AudioDataCollator.py user@remote:/datas/store162/xrick/prjs/Reproduce_English_Pronunciation_Evaluation/src/

# Verify
ssh user@remote "grep 'audios.append' /datas/store162/xrick/prjs/Reproduce_English_Pronunciation_Evaluation/src/AudioDataCollator.py"

# Restart training
ssh user@remote "cd /datas/store162/xrick/prjs/Reproduce_English_Pronunciation_Evaluation && source venv/bin/activate && python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16"
```

---

## Summary

**Must sync**:
1. ⭐ AudioDataCollator.py (critical fix)
2. ⭐ train_single_config_remote.py (new file)
3. ⭐ Run update_model_path_remote.py on remote

**Helpful to sync**:
4. Helper scripts (fix_tokenizer_remote.py, etc.)
5. Documentation (*.md files)

**Never sync**:
- ❌ venv/ directory
- ❌ src/output/ directory
- ❌ __pycache__/ directories
- ❌ .DS_Store files

**Verify after sync**:
- ✅ AudioDataCollator has tuple format
- ✅ Model path not Mac path
- ✅ Training script exists

---

**Time to complete full sync**: 5-10 minutes
**Time to fix current error**: 1-2 minutes (just transfer AudioDataCollator.py)
