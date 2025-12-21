# AudioDataCollator Remote Error - Quick Fix

## Error Summary

```
TypeError: Phi4MMProcessor.__call__() got an unexpected keyword argument 'sampling_rate'
```

**Location**: `src/AudioDataCollator.py`, line 19
**Root Cause**: Remote machine has OLD version of AudioDataCollator with wrong audio format
**Impact**: 🔴 CRITICAL - Training fails immediately during first batch

---

## Problem Explanation

**Old Code (Remote - BROKEN)**:
```python
# ❌ WRONG: Passes sampling_rate as separate argument
batch = self.processor(
    text=text_inputs,
    audios=audio_arrays,
    sampling_rate=16000,  # ← This parameter doesn't exist!
    return_tensors="pt",
    padding=True
)
```

**New Code (Mac - WORKING)**:
```python
# ✅ CORRECT: Passes audios as List[Tuple[array, sampling_rate]]
audios = []
for f in features:
    audio_array = f["audio_array"]
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)
    audios.append((audio_array, f["sampling_rate"]))  # ← Tuple format!

batch = self.processor(
    text=text_inputs,
    audios=audios,  # ← List of tuples
    return_tensors="pt",
    padding=True
)
```

---

## Quick Fix (2 Methods)

### Method 1: Transfer Fixed File (Fastest) ⭐

**On Mac**:
```bash
# Transfer the fixed AudioDataCollator.py to remote
scp src/AudioDataCollator.py user@remote:/path/to/project/src/
```

**On Remote**:
```bash
# Verify file transferred
ls -lh src/AudioDataCollator.py

# Check it has the correct code (should see tuple format)
grep -A 5 "audios.append" src/AudioDataCollator.py
# Expected output should show: audios.append((audio_array, f["sampling_rate"]))
```

### Method 2: Manual Edit on Remote

**On Remote**:
```bash
ssh user@remote
cd /path/to/project

# Backup old file
cp src/AudioDataCollator.py src/AudioDataCollator.py.backup

# Edit file
nano src/AudioDataCollator.py  # or vim
```

**Replace entire file content with**:

```python
from transformers import DataCollatorForSeq2Seq
import numpy as np

class AudioDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # features 是來自資料集的字典列表
        # Phi4MMProcessor expects audios as List[Tuple[numpy_array, sampling_rate]]
        # Convert audio_array from list to numpy array if needed
        audios = []
        for f in features:
            audio_array = f["audio_array"]
            if isinstance(audio_array, list):
                audio_array = np.array(audio_array, dtype=np.float32)
            audios.append((audio_array, f["sampling_rate"]))

        text_inputs = [f["text_input"] for f in features]

        # 處理器處理複雜的多模態填充和 token 化
        batch = self.processor(
            text=text_inputs,
            audios=audios,
            return_tensors="pt",
            padding=True
        )

        # 建立標籤 (Labels) 的邏輯
        # 我們希望遮罩 Prompt 的損失，以便我們只訓練 JSON 輸出。
        # 簡化版：
        batch["labels"] = batch["input_ids"].clone()

        # (選用但推薦) 遮罩填充 token
        if self.processor.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return batch
```

**Save and exit** (Ctrl+X, then Y, then Enter in nano)

---

## Verification

**Test the fix**:

```bash
cd /path/to/project
source venv/bin/activate

# Quick test
python3 << 'EOF'
from src.AudioDataCollator import AudioDataCollator
from transformers import AutoProcessor

# Load processor
processor = AutoProcessor.from_pretrained(
    "/your/model/path/Phi-4-multimodal-instruct",
    local_files_only=True,
    trust_remote_code=True
)

# Test collator
collator = AudioDataCollator(processor)
print("✅ AudioDataCollator imported and initialized successfully")

# Verify it has the correct method
import inspect
source = inspect.getsource(collator.__call__)
if "audios.append((audio_array" in source:
    print("✅ Correct tuple format found")
else:
    print("❌ Still using old format")
EOF
```

**Expected output**:
```
✅ AudioDataCollator imported and initialized successfully
✅ Correct tuple format found
```

---

## After Fix - Start Training

```bash
source venv/bin/activate

# Restart training
python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

**Expected**: Training should start processing batches without the sampling_rate error.

---

## Why This Error Occurred

**Root Cause**: The remote repository has an **old version** of AudioDataCollator.py

**Likely Scenarios**:
1. You transferred the project before we fixed AudioDataCollator on Mac
2. The fix was made on Mac but not synced to remote
3. Remote pulled from an old git commit

**How Mac Training Works**: We fixed this on Mac during earlier debugging, so Mac training runs successfully.

---

## Key Differences Between Versions

| Aspect | Old (Remote - BROKEN) | New (Mac - WORKING) |
|--------|----------------------|---------------------|
| Audio format | List of arrays | List of tuples |
| Sampling rate | Separate argument | Inside tuple |
| Numpy conversion | Missing | Included |
| API compatibility | ❌ Broken | ✅ Correct |

---

## Prevention

**To keep files in sync between Mac and Remote**:

### Option 1: Use Git (Recommended)

```bash
# On Mac (after making fixes)
git add src/AudioDataCollator.py
git commit -m "Fix: AudioDataCollator tuple format for Phi4MMProcessor"
git push

# On Remote
git pull
```

### Option 2: Use rsync

```bash
# Sync entire src/ directory from Mac to Remote
rsync -avz --progress src/ user@remote:/path/to/project/src/
```

### Option 3: Manual Transfer (Quick but error-prone)

```bash
# Transfer specific file
scp src/AudioDataCollator.py user@remote:/path/to/project/src/
```

---

## Related Errors (All Fixed by This)

This fix resolves:
1. ✅ `TypeError: Phi4MMProcessor.__call__() got an unexpected keyword argument 'sampling_rate'`
2. ✅ `AttributeError: 'list' object has no attribute 'ndim'` (numpy conversion)
3. ✅ Audio batch processing failures

---

## Timeline of This Fix

1. **Original Mac Error**: Mac training failed with sampling_rate error
2. **Mac Fix Applied**: Updated AudioDataCollator.py to tuple format (SUCCESS)
3. **Mac Training Started**: Training runs successfully on Mac
4. **Remote Transfer**: Project transferred to remote (OLD version)
5. **Remote Error**: Same error appears on remote (needs fix)
6. **Solution**: Transfer updated AudioDataCollator.py to remote

---

## Quick Reference

### Error
```
TypeError: Phi4MMProcessor.__call__() got an unexpected keyword argument 'sampling_rate'
```

### Quick Fix
```bash
# Method 1: Transfer from Mac (fastest)
scp src/AudioDataCollator.py user@remote:/path/to/project/src/

# Method 2: Manual edit on remote
nano src/AudioDataCollator.py
# Replace with correct code (see above)
```

### Verify
```bash
grep "audios.append((audio_array" src/AudioDataCollator.py
# Should find the tuple format
```

### Train
```bash
python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

---

## Related Documentation

- [claudedocs/audiodecoder_compatibility_fix.md](claudedocs/audiodecoder_compatibility_fix.md) - Original fix
- [BUGFIX_SUMMARY.md](BUGFIX_SUMMARY.md) - All bugs fixed
- [REMOTE_FINAL_SETUP.md](REMOTE_FINAL_SETUP.md) - Complete setup

---

**Error**: AudioDataCollator API incompatibility
**Severity**: 🔴 CRITICAL
**Solution**: Transfer fixed file from Mac or manual edit
**Time to fix**: 1-2 minutes
