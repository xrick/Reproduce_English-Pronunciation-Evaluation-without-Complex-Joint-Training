#!/bin/bash
# Remote Tokenizer Fix Script
# 自動修復遠程機器的 tokenizer 問題

set -e  # Exit on error

MODEL_DIR="/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct"
BACKUP_DIR="${MODEL_DIR}_backup_$(date +%Y%m%d_%H%M%S)"

echo "=================================="
echo "Remote Tokenizer Fix Script"
echo "=================================="
echo ""

# Step 1: Check environment
echo "📋 Step 1: Checking environment..."
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

cd "$MODEL_DIR"
echo "✅ Model directory found"
echo ""

# Step 2: Check tokenizer.json
echo "📋 Step 2: Checking tokenizer.json..."
if [ ! -f "tokenizer.json" ]; then
    echo "❌ Error: tokenizer.json not found!"
    REDOWNLOAD=true
else
    SIZE=$(stat -f%z "tokenizer.json" 2>/dev/null || stat -c%s "tokenizer.json" 2>/dev/null)
    echo "File size: $SIZE bytes"

    if [ "$SIZE" -eq 0 ]; then
        echo "❌ Error: tokenizer.json is empty!"
        REDOWNLOAD=true
    else
        # Test JSON validity
        if python3 -c "import json; json.load(open('tokenizer.json'))" 2>/dev/null; then
            echo "✅ tokenizer.json is valid JSON"
            REDOWNLOAD=false
        else
            echo "❌ Error: tokenizer.json is corrupted (invalid JSON)"
            REDOWNLOAD=true
        fi
    fi
fi
echo ""

# Step 3: Backup if needed
if [ "$REDOWNLOAD" = true ]; then
    echo "📋 Step 3: Backing up current model directory..."
    echo "Creating backup: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cp tokenizer*.json "$BACKUP_DIR/" 2>/dev/null || true
    echo "✅ Backup complete"
    echo ""

    # Step 4: Re-download
    echo "📋 Step 4: Re-downloading tokenizer files..."
    echo "This may take a few minutes..."

    # Check if huggingface_hub is installed
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Installing huggingface_hub..."
        pip install -U huggingface_hub
    fi

    # Download tokenizer files
    python3 << 'PYEOF'
from huggingface_hub import hf_hub_download
import os

model_id = "microsoft/phi-4-multimodal-instruct"
local_dir = os.getcwd()

files = [
    "tokenizer.json",
    "tokenizer_config.json",
]

print("Downloading files...")
for filename in files:
    print(f"  - {filename}")
    try:
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            force_download=True
        )
        print(f"    ✅ Downloaded")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print("✅ Download complete!")
PYEOF

    echo ""
fi

# Step 5: Verify fix
echo "📋 Step 5: Verifying tokenizer loading..."
python3 << 'PYEOF'
import sys
try:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        ".",
        trust_remote_code=True
    )
    print("✅ Tokenizer loaded successfully!")
    print(f"✅ Vocab size: {processor.tokenizer.vocab_size:,}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ Fix Complete!"
    echo "=================================="
    echo ""
    echo "You can now run training:"
    echo "  python src/train_single_config_remote.py --config paper_r64 --gpus 0"
    echo ""
else
    echo ""
    echo "=================================="
    echo "❌ Fix Failed"
    echo "=================================="
    echo ""
    echo "Please try manual steps:"
    echo "1. Check model directory: $MODEL_DIR"
    echo "2. Verify internet connection"
    echo "3. Try using online model:"
    echo "   Edit src/model_utility_configs.py"
    echo "   Change model_path to: 'microsoft/phi-4-multimodal-instruct'"
    echo ""
fi
