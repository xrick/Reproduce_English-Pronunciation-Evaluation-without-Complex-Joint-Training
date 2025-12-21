#!/bin/bash
# Quick check that all necessary files exist

echo "================================================================================"
echo "Checking Fix Files"
echo "================================================================================"
echo ""

FILES=(
    "fix_lora_training.py:CRITICAL"
    "fix_bf16_to_fp16.py:Important"
    "patch_disable_amp.py:Important"
    "fix_processor_save.py:Optional"
    "verify_all_fixes.py:Utility"
    "REMOTE_SETUP_COMPLETE.sh:Automation"
    "TRANSFER_TO_REMOTE.sh:Automation"
    "QUICK_START.md:Documentation"
)

ALL_FOUND=1
for item in "${FILES[@]}"; do
    IFS=':' read -r file importance <<< "$item"
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        printf "✅ %-30s [%s] %s\n" "$file" "$importance" "$size"
    else
        printf "❌ %-30s [%s] MISSING!\n" "$file" "$importance"
        ALL_FOUND=0
    fi
done

echo ""
echo "================================================================================"
if [ $ALL_FOUND -eq 1 ]; then
    echo "✅ ALL FILES PRESENT"
    echo "================================================================================"
    echo ""
    echo "📝 Next steps:"
    echo "1. Edit TRANSFER_TO_REMOTE.sh (set REMOTE_HOST)"
    echo "2. Run: ./TRANSFER_TO_REMOTE.sh"
    echo "3. On remote: bash REMOTE_SETUP_COMPLETE.sh"
    echo ""
    echo "📖 Read QUICK_START.md for detailed instructions"
else
    echo "❌ SOME FILES MISSING"
    echo "================================================================================"
fi
