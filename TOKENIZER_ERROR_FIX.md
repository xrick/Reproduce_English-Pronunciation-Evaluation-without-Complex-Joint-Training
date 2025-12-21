# Tokenizer.json Corruption Error - Quick Fix Guide

## Error Summary

```
Exception: expected value at line 1 column 1
```

**Error Location**: `transformers/tokenization_utils_fast.py`, line 117
**Root Cause**: `tokenizer.json` file is corrupted, empty, or incomplete
**Impact**: 🔴 CRITICAL - Prevents model loading completely

---

## Quick Fix (3 Methods)

### Method 1: Automatic Fix Script (Recommended) ⭐

**Transfer script to remote**:
```bash
# On Mac
scp fix_tokenizer_remote.py user@remote:/path/to/project/

# On remote
ssh user@remote
cd /path/to/project
python fix_tokenizer_remote.py
```

**What it does**:
- ✅ Auto-detects model path from config
- ✅ Checks if tokenizer.json is valid
- ✅ Re-downloads from HuggingFace if corrupted
- ✅ Verifies fix worked

**Expected output**:
```
✅ TOKENIZER FIXED SUCCESSFULLY
📝 You can now run training:
   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

---

### Method 2: Manual Download (If Script Fails)

**On remote machine**:

```bash
# 1. Find your model directory
# (Check src/model_utility_configs.py for model_path)

# 2. Go to model directory
cd /your/model/path/Phi-4-multimodal-instruct

# 3. Backup corrupted file
mv tokenizer.json tokenizer.json.backup

# 4. Download fresh tokenizer files
python3 << 'EOF'
from huggingface_hub import hf_hub_download

files = ["tokenizer.json", "tokenizer_config.json"]

for f in files:
    print(f"Downloading {f}...")
    hf_hub_download(
        repo_id="microsoft/Phi-4-multimodal-instruct",
        filename=f,
        local_dir=".",
        local_dir_use_symlinks=False,
        force_download=True
    )
print("✅ Done!")
EOF

# 5. Verify
python3 -c "import json; json.load(open('tokenizer.json')); print('✅ Valid JSON')"
```

---

### Method 3: Transfer from Mac (If No Internet)

If remote machine can't access HuggingFace:

**On Mac**:
```bash
# Find your local model path (check src/model_utility_configs.py)
LOCAL_MODEL="/Users/xrickliao/WorkSpaces/LLM_Repo/models/Phi-4-multimodal-instruct"

# Transfer tokenizer files to remote
scp ${LOCAL_MODEL}/tokenizer.json user@remote:/remote/model/path/
scp ${LOCAL_MODEL}/tokenizer_config.json user@remote:/remote/model/path/
```

**On remote**:
```bash
# Verify
cd /remote/model/path
python3 -c "import json; json.load(open('tokenizer.json')); print('✅ Valid JSON')"
```

---

## Diagnosis Steps

### Step 1: Verify Model Path

```bash
# On remote machine
cd /path/to/project

# Check what path is configured
grep -n "model_path =" src/model_utility_configs.py

# Example output:
# 47:    model_path = "/datas/store162/xrick/models/Phi-4-multimodal-instruct"
# 132:   model_path = "/datas/store162/xrick/models/Phi-4-multimodal-instruct"
```

### Step 2: Check Tokenizer File

```bash
# Go to model directory (use path from Step 1)
cd /datas/store162/xrick/models/Phi-4-multimodal-instruct

# Check if file exists and size
ls -lh tokenizer.json

# Expected: ~2-4MB file
# Problem: 0 bytes or very small file
```

### Step 3: Validate JSON

```bash
# Test if it's valid JSON
python3 -c "import json; json.load(open('tokenizer.json'))"

# If corrupted, you'll see:
# json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

---

## Common Causes

### 1. Incomplete Download
- **Symptom**: tokenizer.json is 0 bytes or very small
- **Cause**: Download interrupted or failed
- **Fix**: Re-download using Method 1 or 2

### 2. Transfer Corruption
- **Symptom**: File exists but JSON is invalid
- **Cause**: File corrupted during rsync/scp transfer
- **Fix**: Transfer again using Method 3 or re-download

### 3. Disk Space Issues
- **Symptom**: File created but incomplete
- **Cause**: Disk full during download
- **Fix**: Free up space, then re-download
```bash
# Check disk space
df -h /datas/store162/
```

### 4. Permission Issues
- **Symptom**: Cannot read tokenizer.json
- **Cause**: Wrong file permissions
- **Fix**: Fix permissions
```bash
chmod 644 tokenizer.json
```

---

## Prevention

### Best Practices for Model Transfer

**When downloading from HuggingFace**:
```python
# Always use force_download for critical files
hf_hub_download(
    repo_id="microsoft/Phi-4-multimodal-instruct",
    filename="tokenizer.json",
    local_dir=".",
    local_dir_use_symlinks=False,
    force_download=True  # Ensures complete download
)
```

**When transferring between machines**:
```bash
# Use rsync with checksum verification
rsync -avz --checksum \
  /local/model/path/tokenizer.json \
  user@remote:/remote/model/path/

# Or use scp with verification
scp tokenizer.json user@remote:/path/
ssh user@remote "md5sum /path/tokenizer.json"
# Compare with local: md5sum tokenizer.json
```

**When using shared storage**:
```bash
# Verify file integrity after copy
python3 -c "import json; json.load(open('tokenizer.json')); print('✅ OK')"
```

---

## Verification After Fix

**Complete verification**:

```bash
cd /path/to/project
source venv/bin/activate

# Test 1: Load processor
python3 << 'EOF'
from transformers import AutoProcessor

model_path = "/your/model/path/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True
)
print(f"✅ Processor loaded")
print(f"✅ Vocab size: {processor.tokenizer.vocab_size:,}")
EOF

# Test 2: Load full config
python3 << 'EOF'
from model_utility_configs import CONFIGS

config = CONFIGS["paper_r64"]
model, processor, peft_config = config["loader"]()
print("✅ Full model loaded successfully")
print(f"✅ Ready to train!")
EOF
```

**Expected output**:
```
✅ Patched PEFT to handle Phi-4's missing prepare_inputs_for_generation method
✅ Processor loaded
✅ Vocab size: 51,200

📊 【論文配置 r=64】參數統計:
  ...
✅ Full model loaded successfully
✅ Ready to train!
```

---

## Troubleshooting the Fix

### "huggingface_hub not installed"

```bash
source venv/bin/activate
pip install huggingface-hub
```

### "Cannot connect to huggingface.co"

**Options**:
1. Use Method 3 (transfer from Mac)
2. Configure proxy if available:
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   python fix_tokenizer_remote.py
   ```
3. Use HuggingFace mirror (China):
   ```python
   # Edit fix_tokenizer_remote.py
   # Add before hf_hub_download:
   import os
   os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
   ```

### "Permission denied"

```bash
# Fix permissions on model directory
chmod 755 /path/to/model/directory
chmod 644 /path/to/model/directory/tokenizer.json
```

### Fix completes but training still fails

**Check all tokenizer files**:
```bash
cd /model/path

# Should all exist and be valid
ls -lh tokenizer.json tokenizer_config.json

# Verify both are valid JSON
python3 -c "
import json
for f in ['tokenizer.json', 'tokenizer_config.json']:
    json.load(open(f))
    print(f'✅ {f} is valid')
"
```

---

## Alternative: Use Online Model

If local model keeps having issues, switch to online model:

**Edit `src/model_utility_configs.py`**:

```python
# Line 47 and 132, change:
# FROM:
model_path = "/datas/store162/xrick/models/Phi-4-multimodal-instruct"

# TO:
model_path = "microsoft/Phi-4-multimodal-instruct"
```

**Advantages**:
- ✅ Automatic integrity verification
- ✅ No manual file management
- ✅ Always gets correct files

**Disadvantages**:
- ⚠️ Requires internet on first run
- ⚠️ First download takes 15-20 minutes
- ⚠️ Uses disk cache (~15GB)

---

## Quick Reference Card

### Error
```
Exception: expected value at line 1 column 1
```

### Quick Fix
```bash
# Method 1: Auto-fix script
python fix_tokenizer_remote.py

# Method 2: Manual download
cd /model/path
python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('microsoft/Phi-4-multimodal-instruct', 'tokenizer.json', \
local_dir='.', force_download=True)"

# Method 3: Transfer from Mac
scp /mac/model/path/tokenizer.json user@remote:/remote/model/path/
```

### Verify
```bash
python3 -c "import json; json.load(open('tokenizer.json')); print('✅ OK')"
```

### Then Train
```bash
python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

---

## Related Documentation

- [REMOTE_ERROR_QUICKFIX.md](REMOTE_ERROR_QUICKFIX.md) - General remote errors
- [fix_remote_tokenizer_v2.sh](fix_remote_tokenizer_v2.sh) - Interactive bash version
- [REMOTE_FINAL_SETUP.md](REMOTE_FINAL_SETUP.md) - Complete setup guide

---

**Error**: tokenizer.json corruption (JSON parsing error)
**Severity**: 🔴 CRITICAL
**Solution**: Re-download tokenizer files
**Time to fix**: 2-5 minutes
