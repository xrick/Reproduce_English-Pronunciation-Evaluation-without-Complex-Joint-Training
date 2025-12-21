#!/bin/bash
# Remote Tokenizer Fix Script v2
# 自動修復遠程機器的 tokenizer 問題
# 支持自定義模型路徑

set -e  # Exit on error

echo "=================================="
echo "Remote Tokenizer Fix Script v2"
echo "=================================="
echo ""

# Ask for model directory
echo "Please enter your Phi-4-multimodal-instruct model directory path:"
echo "Examples:"
echo "  - /datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct"
echo "  - $HOME/models/Phi-4-multimodal-instruct"
echo "  - $HOME/.cache/huggingface/hub/models--microsoft--phi-4-multimodal-instruct/snapshots/..."
echo ""
read -p "Model path: " MODEL_DIR

# Trim whitespace
MODEL_DIR=$(echo "$MODEL_DIR" | xargs)

if [ -z "$MODEL_DIR" ]; then
    echo "❌ Error: No path provided"
    exit 1
fi

# Expand ~ to home directory
MODEL_DIR="${MODEL_DIR/#\~/$HOME}"

echo ""
echo "Using model directory: $MODEL_DIR"
echo ""

# Step 1: Check environment
echo "📋 Step 1: Checking environment..."
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    echo ""
    echo "Suggestions:"
    echo "1. Check if path is correct"
    echo "2. Run find_model_path.sh to locate the model"
    echo "3. Or use online model path (see below)"
    exit 1
fi

cd "$MODEL_DIR"
echo "✅ Model directory found"
echo ""

# Step 2: Check tokenizer.json
echo "📋 Step 2: Checking tokenizer.json..."
REDOWNLOAD=false

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
    BACKUP_DIR="${MODEL_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "📋 Step 3: Backing up current files..."
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
    echo "Model path verified: $MODEL_DIR"
    echo ""
    echo "📝 Update your training script:"
    echo "Edit src/model_utility_configs.py or src/model_utility.py"
    echo "Set MODEL_PATH to: $MODEL_DIR"
    echo ""
    echo "Or run training with:"
    echo "  python src/train_single_config_remote.py --config paper_r64 --gpus 0"
    echo ""
else
    echo ""
    echo "=================================="
    echo "❌ Fix Failed"
    echo "=================================="
    echo ""
    echo "Alternative: Use online model (recommended)"
    echo ""
    echo "Edit src/model_utility_configs.py:"
    echo "Change model_path to:"
    echo "  model_path = 'microsoft/phi-4-multimodal-instruct'"
    echo ""
    echo "This will:"
    echo "  ✅ Automatically download and cache the model"
    echo "  ✅ Verify file integrity"
    echo "  ✅ No manual file management needed"
    echo ""
fi
