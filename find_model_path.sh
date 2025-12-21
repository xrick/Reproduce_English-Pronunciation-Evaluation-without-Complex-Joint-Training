#!/bin/bash
# Find Phi-4-multimodal-instruct model location on remote machine

echo "=================================="
echo "Model Path Discovery Script"
echo "=================================="
echo ""

# Check common locations
echo "🔍 Checking common model locations..."
echo ""

COMMON_PATHS=(
    "/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct"
    "$HOME/models/Phi-4-multimodal-instruct"
    "$HOME/.cache/huggingface/hub/models--microsoft--phi-4-multimodal-instruct"
    "/models/Phi-4-multimodal-instruct"
    "/data/models/Phi-4-multimodal-instruct"
    "./models/Phi-4-multimodal-instruct"
)

FOUND=false
for path in "${COMMON_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✅ Found: $path"
        FOUND=true

        # Check for critical files
        if [ -f "$path/config.json" ]; then
            echo "   ✅ config.json exists"
        fi
        if [ -f "$path/tokenizer.json" ]; then
            echo "   ✅ tokenizer.json exists"
        else
            echo "   ⚠️  tokenizer.json missing or corrupted"
        fi
        echo ""
    fi
done

if [ "$FOUND" = false ]; then
    echo "❌ Model not found in common locations"
    echo ""
    echo "🔍 Searching entire filesystem (this may take a while)..."
    echo ""

    # Search for model directory
    find / -type d -name "Phi-4-multimodal-instruct" 2>/dev/null | while read -r dir; do
        echo "Found potential location: $dir"
        if [ -f "$dir/config.json" ]; then
            echo "✅ This appears to be a valid model directory"
        fi
    done

    # Also search in HuggingFace cache
    echo ""
    echo "🔍 Checking HuggingFace cache..."
    if [ -d "$HOME/.cache/huggingface/hub" ]; then
        find "$HOME/.cache/huggingface/hub" -type d -name "*phi*4*multimodal*" 2>/dev/null | while read -r dir; do
            echo "Found in cache: $dir"
        done
    fi
fi

echo ""
echo "=================================="
echo "📋 Next Steps:"
echo "=================================="
echo ""
echo "1. Note the path where model was found above"
echo "2. Update model path in your code:"
echo ""
echo "   Edit: src/model_utility_configs.py"
echo "   OR: src/model_utility.py"
echo ""
echo "   Change MODEL_PATH to the found location"
echo ""
echo "3. Or use online model (easiest):"
echo "   model_path = 'microsoft/phi-4-multimodal-instruct'"
echo ""
