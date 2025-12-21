#!/bin/bash
# Complete Remote Setup Script
# Run this on REMOTE machine after transferring all fix scripts

set -e  # Exit on error

echo "================================================================================"
echo "COMPLETE REMOTE SETUP - Apply All Fixes"
echo "================================================================================"
echo ""

# Check we're in the right directory
if [ ! -f "src/train_single_config_remote.py" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   Current: $(pwd)"
    echo "   Expected: Contains src/train_single_config_remote.py"
    exit 1
fi

echo "📍 Working directory: $(pwd)"
echo ""

# Stop any running training
echo "=== Step 1: Stop Running Training ==="
pkill -f train_single_config_remote.py 2>/dev/null || echo "✅ No training processes running"
echo ""

# Apply Fix 1: BF16 → FP16
echo "=== Step 2: Fix BF16 → FP16 (TITAN RTX Compatibility) ==="
if [ -f "fix_bf16_to_fp16.py" ]; then
    python fix_bf16_to_fp16.py
else
    echo "⚠️  Warning: fix_bf16_to_fp16.py not found - skipping"
fi
echo ""

# Apply Fix 2: Disable AMP
echo "=== Step 3: Disable AMP (Native FP16) ==="
if [ -f "patch_disable_amp.py" ]; then
    python patch_disable_amp.py
else
    echo "⚠️  Warning: patch_disable_amp.py not found - skipping"
fi
echo ""

# Apply Fix 3: Enable LoRA Training (CRITICAL)
echo "=== Step 4: Enable LoRA Training (CRITICAL FIX) ==="
if [ -f "fix_lora_training.py" ]; then
    python fix_lora_training.py
else
    echo "❌ ERROR: fix_lora_training.py not found - THIS IS REQUIRED!"
    exit 1
fi
echo ""

# Apply Fix 4: Processor Save Error
echo "=== Step 5: Fix Processor Save Error ==="
if [ -f "fix_processor_save.py" ]; then
    python fix_processor_save.py
else
    echo "⚠️  Warning: fix_processor_save.py not found - skipping (non-critical)"
fi
echo ""

# Verify all fixes
echo "=== Step 6: Verify All Fixes ==="
if [ -f "verify_all_fixes.py" ]; then
    python verify_all_fixes.py
    VERIFY_RESULT=$?
else
    echo "⚠️  Warning: verify_all_fixes.py not found - cannot verify"
    VERIFY_RESULT=0
fi
echo ""

# Clean old training data
echo "=== Step 7: Clean Old Training Data ==="
if [ -d "src/output/paper_r64" ]; then
    echo "Deleting src/output/paper_r64/..."
    rm -rf src/output/paper_r64/
    echo "✅ Old training data deleted"
else
    echo "✅ No old training data to clean"
fi
echo ""

# Final status
echo "================================================================================"
if [ $VERIFY_RESULT -eq 0 ]; then
    echo "✅ SETUP COMPLETE - READY TO TRAIN"
    echo "================================================================================"
    echo ""
    echo "🚀 Start training with:"
    echo "   source venv/bin/activate"
    echo "   python src/train_single_config_remote.py --config paper_r64 --gpus 0"
    echo ""
    echo "📊 Expected model loading output:"
    echo "   🔧 Applying LoRA configuration to model..."
    echo "   ✅ LoRA configuration applied - parameters are now trainable"
    echo "   可訓練參數: 200,000,000 (3.5%)"
    echo ""
    echo "📈 Expected training behavior:"
    echo "   loss: 6.98 → 6.42 → 5.89 → ... → 2.5-3.0"
    echo "   (Clear downward trend, NOT stuck at 10.6!)"
else
    echo "❌ SETUP INCOMPLETE - FIXES NEEDED"
    echo "================================================================================"
    echo ""
    echo "⚠️  Some fixes failed verification"
    echo "Review output above and apply missing fixes manually"
fi
