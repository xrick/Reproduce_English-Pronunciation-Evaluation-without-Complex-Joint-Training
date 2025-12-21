#!/bin/bash
# ONE-LINE FIX FOR REMOTE MACHINE
# Copy each command block to your remote terminal

cat << 'EOF'
================================================================================
ONE-LINE FIX FOR REMOTE MACHINE
================================================================================

STEP 1: Stop Current Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pkill -f train_single_config_remote.py


STEP 2: Go to Project Directory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cd /datas/store162/xrick/prjs/Reproduce_English_Pronunciation_Evaluation


STEP 3: Apply LoRA Fix (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sed -i.backup '/task_type="CAUSAL_LM",$/a\
\
    # CRITICAL: Apply LoRA configuration to enable training\
    print("\\n🔧 Applying LoRA configuration to model...")\
    model = get_peft_model(model, peft_config)\
    print("✅ LoRA configuration applied - parameters are now trainable")' src/model_utility_configs.py


STEP 4: Fix FP16
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sed -i 's/torch\.bfloat16/torch.float16/g' src/model_utility_configs.py


STEP 5: Disable AMP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sed -i 's/"fp16": use_fp16,/"fp16": False,  # Disabled AMP/' src/train_single_config_remote.py


STEP 6: Verify All Fixes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "=== LoRA Fix Check ==="
grep -A 2 "get_peft_model" src/model_utility_configs.py && echo "✅ LoRA fix applied" || echo "❌ LoRA fix MISSING"

echo ""
echo "=== FP16 Fix Check ==="
grep "torch.float16" src/model_utility_configs.py | head -1 && echo "✅ FP16 fix applied" || echo "❌ FP16 fix MISSING"

echo ""
echo "=== AMP Fix Check ==="
grep '"fp16": False' src/train_single_config_remote.py && echo "✅ AMP fix applied" || echo "❌ AMP fix MISSING"


STEP 7: Clean and Restart
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rm -rf src/output/paper_r64/
source venv/bin/activate
python src/train_single_config_remote.py --config paper_r64 --gpus 0


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT (Model Loading):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 Applying LoRA configuration to model...
✅ LoRA configuration applied - parameters are now trainable
可訓練參數: 200,000,000 (3.5%)  ← MUST BE ~200M, NOT 0!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT (Training):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{'loss': 6.98, 'epoch': 0.26}  ← Starts around 7.0
{'loss': 6.42, 'epoch': 0.51}  ← Decreases
{'loss': 5.89, 'epoch': 0.77}  ← Continues down

NO WARNING about "requires_grad"!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF
