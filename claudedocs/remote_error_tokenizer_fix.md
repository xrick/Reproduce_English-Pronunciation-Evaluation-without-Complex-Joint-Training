# Remote Training Error: Tokenizer File Corruption

## Error Description

```
Exception: expected value at line 1 column 1
```

**發生位置**: Loading GPT2TokenizerFast from `tokenizer.json`

## Root Cause Analysis

The error occurs when the tokenizer file (`tokenizer.json`) is:
1. **Corrupted** during file transfer (incomplete copy)
2. **Empty** or malformed
3. **Permission issues** preventing proper read

## Solution 1: Verify and Re-download Model

### Step 1: Check Tokenizer File Integrity

```bash
# On remote machine
cd /datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct

# Check file size (should be > 0 bytes)
ls -lh tokenizer.json

# Check if file is valid JSON
python3 -c "import json; json.load(open('tokenizer.json'))" && echo "✅ Valid JSON" || echo "❌ Invalid JSON"

# Check file permissions
ls -l tokenizer.json
```

### Step 2: Re-download Corrupted Files

If tokenizer.json is corrupted:

```bash
# Backup corrupted file
mv tokenizer.json tokenizer.json.backup

# Re-download using HuggingFace CLI
pip install -U huggingface_hub
huggingface-cli download microsoft/phi-4-multimodal-instruct \
  --local-dir /datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct \
  --local-dir-use-symlinks False \
  --include "tokenizer.json"

# Or download specific file
wget https://huggingface.co/microsoft/phi-4-multimodal-instruct/resolve/main/tokenizer.json \
  -O tokenizer.json
```

### Step 3: Verify Fix

```bash
python3 << 'EOF'
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct",
    trust_remote_code=True
)
print("✅ Processor loaded successfully!")
EOF
```

## Solution 2: Use Online Model Loading (Temporary)

If local model is corrupted, use HuggingFace Hub directly:

### Modify model_utility.py or model_utility_configs.py

```python
# Instead of local path:
# model_path = "/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct"

# Use HuggingFace Hub (will cache locally):
model_path = "microsoft/phi-4-multimodal-instruct"
```

**Pros**:
- Automatic validation and download
- Guaranteed file integrity

**Cons**:
- Requires internet connection during first run
- Slower initial load

## Solution 3: Copy from Working Mac Machine

If Mac local model works correctly:

```bash
# On Mac (local)
cd /Users/xrickliao/WorkSpaces/LLM_Repo/models/Phi-4-multimodal-instruct

# Verify local tokenizer.json is valid
python3 -c "import json; json.load(open('tokenizer.json'))" && echo "✅ Local tokenizer valid"

# Copy to remote via SCP
scp tokenizer.json user@remote:/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct/

# Or compress and copy entire model directory
tar czf phi4-model.tar.gz Phi-4-multimodal-instruct/
scp phi4-model.tar.gz user@remote:/datas/store162/xrick/LLM_Repo/models/
# On remote:
tar xzf phi4-model.tar.gz
```

## Solution 4: Ignore Warnings and Focus on Real Error

The warnings before the error are **harmless**:

```python
# These are just warnings, not errors:
"The module name  (originally ) is not a valid Python identifier."
# These can be ignored - they don't affect training
```

**The real error** is the tokenizer loading failure.

## Prevention: Pre-flight Check Script

Create `check_remote_env.py`:

```python
#!/usr/bin/env python3
"""
Remote environment validation script
Run before training to catch issues early
"""

import os
import json
import torch
from pathlib import Path

def check_cuda():
    print("🔍 Checking CUDA...")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

def check_model_files(model_path):
    print(f"🔍 Checking model files in: {model_path}")
    model_dir = Path(model_path)

    critical_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]

    for filename in critical_files:
        filepath = model_dir / filename
        if not filepath.exists():
            print(f"  ❌ Missing: {filename}")
            continue

        # Check file size
        size = filepath.stat().st_size
        if size == 0:
            print(f"  ❌ Empty file: {filename}")
            continue

        # Validate JSON
        if filename.endswith('.json'):
            try:
                with open(filepath) as f:
                    json.load(f)
                print(f"  ✅ Valid: {filename} ({size:,} bytes)")
            except json.JSONDecodeError as e:
                print(f"  ❌ Corrupted JSON: {filename} - {e}")
        else:
            print(f"  ✅ Exists: {filename} ({size:,} bytes)")
    print()

def check_tokenizer(model_path):
    print("🔍 Checking tokenizer loading...")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("  ✅ Tokenizer loaded successfully!")
        print(f"  Vocab size: {processor.tokenizer.vocab_size}")
    except Exception as e:
        print(f"  ❌ Tokenizer loading failed: {e}")
    print()

def check_dataset(data_path):
    print(f"🔍 Checking dataset: {data_path}")
    if not os.path.exists(data_path):
        print(f"  ❌ Dataset directory not found: {data_path}")
        return

    dataset_info = Path(data_path) / "dataset_info.json"
    if dataset_info.exists():
        with open(dataset_info) as f:
            info = json.load(f)
        print(f"  ✅ Dataset info found")
        print(f"  Splits: {list(info.get('splits', {}).keys())}")
    else:
        print(f"  ⚠️  No dataset_info.json found")
    print()

if __name__ == "__main__":
    print("="*60)
    print("Remote Environment Pre-flight Check")
    print("="*60)
    print()

    check_cuda()

    model_path = "/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct"
    check_model_files(model_path)
    check_tokenizer(model_path)

    data_path = "../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/train/"
    check_dataset(data_path)

    print("="*60)
    print("Pre-flight check complete!")
    print("="*60)
```

**Usage**:
```bash
python check_remote_env.py
```

## Immediate Action for Remote Machine

### Quick Fix Commands

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Navigate to model directory
cd /datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct

# 3. Check tokenizer.json
ls -lh tokenizer.json
python3 -c "import json; f=open('tokenizer.json'); json.load(f); print('Valid')"

# 4a. If corrupted - re-download
pip install -U huggingface_hub
huggingface-cli download microsoft/phi-4-multimodal-instruct \
  --local-dir . \
  --local-dir-use-symlinks False \
  --include "tokenizer*.json"

# 4b. OR use online model (modify config)
# Edit src/model_utility_configs.py:
# Change: model_path = "microsoft/phi-4-multimodal-instruct"

# 5. Test loading
python3 << 'EOF'
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    ".",  # or use online: "microsoft/phi-4-multimodal-instruct"
    trust_remote_code=True
)
print("✅ Success!")
EOF
```

## Summary

**Most Likely Cause**: Corrupted `tokenizer.json` file
**Quick Fix**: Re-download tokenizer files from HuggingFace
**Alternative**: Use online model path instead of local

**Recommended Steps**:
1. Check tokenizer.json integrity
2. Re-download if corrupted
3. Verify with test script
4. Proceed with training

The warnings about "module name" can be **safely ignored** - they don't affect functionality.
