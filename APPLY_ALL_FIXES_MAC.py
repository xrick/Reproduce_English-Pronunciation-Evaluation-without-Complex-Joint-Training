#!/usr/bin/env python3
"""
CRITICAL FIX SCRIPT - Apply ALL fixes on Mac before transferring to remote

This script fixes:
1. LoRA parameter training (BOTH r=320 and r=64 configs)
2. BF16 → FP16 conversion for TITAN RTX
3. AMP disabling to prevent GradScaler errors

Run on Mac, then transfer fixed files to remote.
"""

import re
import shutil
from pathlib import Path

def backup_file(filepath):
    """Create backup with .backup extension"""
    backup_path = f"{filepath}.backup"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path

def fix_model_utility_configs():
    """
    Fix model_utility_configs.py - BOTH LoRA configurations

    Critical issues:
    1. Missing get_peft_model() call after LoraConfig creation (BOTH configs)
    2. torch.bfloat16 → torch.float16 (TITAN RTX compatibility)
    """
    filepath = "src/model_utility_configs.py"
    print(f"\n🔧 Fixing {filepath}...")

    backup_file(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix 1: Change BF16 → FP16 (both occurrences)
    content = content.replace('torch.bfloat16', 'torch.float16')
    print("  ✅ Changed torch.bfloat16 → torch.float16")

    # Fix 2: Add get_peft_model() for PRETRAINED config (r=320)
    # Find the pattern after task_type="CAUSAL_LM", in first function
    pretrained_pattern = r'(def get_model_and_processor_pretrained.*?peft_config = LoraConfig\(.*?task_type="CAUSAL_LM",\s*\))\s*(\n\s*# 統計參數)'

    pretrained_replacement = r'''\1

    # CRITICAL: Apply LoRA configuration to enable training
    print("\n🔧 Applying LoRA configuration to model...")
    model = get_peft_model(model, peft_config)
    print("✅ LoRA configuration applied - parameters are now trainable")
\2'''

    content = re.sub(pretrained_pattern, pretrained_replacement, content, flags=re.DOTALL)
    print("  ✅ Added get_peft_model() to pretrained config (r=320)")

    # Fix 3: Add get_peft_model() for PAPER config (r=64)
    # Find the pattern after task_type="CAUSAL_LM", in second function
    paper_pattern = r'(def get_model_and_processor_paper.*?peft_config = LoraConfig\(.*?task_type="CAUSAL_LM",\s*\))\s*(\n\s*# 統計參數)'

    paper_replacement = r'''\1

    # CRITICAL: Apply LoRA configuration to enable training
    print("\n🔧 Applying LoRA configuration to model...")
    model = get_peft_model(model, peft_config)
    print("✅ LoRA configuration applied - parameters are now trainable")
\2'''

    content = re.sub(paper_pattern, paper_replacement, content, flags=re.DOTALL)
    print("  ✅ Added get_peft_model() to paper config (r=64)")

    # Write fixed content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Fixed {filepath}")
    return True

def fix_train_single_config_remote():
    """
    Fix train_single_config_remote.py

    Critical issue: AMP enabled causes GradScaler errors
    Solution: Disable AMP (fp16=False)
    """
    filepath = "src/train_single_config_remote.py"
    print(f"\n🔧 Fixing {filepath}...")

    backup_file(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix: Disable AMP
    content = re.sub(
        r'"fp16":\s*use_fp16,',
        '"fp16": False,  # Disabled AMP - using native FP16 model',
        content
    )
    print("  ✅ Disabled AMP (fp16=False)")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Fixed {filepath}")
    return True

def verify_fixes():
    """Verify all fixes were applied correctly"""
    print("\n🔍 Verifying fixes...")

    errors = []

    # Verify model_utility_configs.py
    with open("src/model_utility_configs.py", 'r') as f:
        content = f.read()

    # Check for get_peft_model calls (should be 2)
    peft_calls = content.count('model = get_peft_model(model, peft_config)')
    if peft_calls != 2:
        errors.append(f"Expected 2 get_peft_model() calls, found {peft_calls}")
    else:
        print("  ✅ Both LoRA configs have get_peft_model() call")

    # Check for FP16 (should have no BF16)
    if 'torch.bfloat16' in content:
        errors.append("Still contains torch.bfloat16")
    else:
        print("  ✅ All torch.bfloat16 → torch.float16")

    # Verify train_single_config_remote.py
    with open("src/train_single_config_remote.py", 'r') as f:
        content = f.read()

    if '"fp16": False' in content:
        print("  ✅ AMP disabled (fp16=False)")
    else:
        errors.append("AMP not disabled in training args")

    if errors:
        print("\n❌ VERIFICATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ ALL FIXES VERIFIED!")
        return True

def main():
    print("="*80)
    print("APPLYING ALL FIXES ON MAC")
    print("="*80)

    success = True

    try:
        # Apply all fixes
        success &= fix_model_utility_configs()
        success &= fix_train_single_config_remote()

        # Verify
        success &= verify_fixes()

        if success:
            print("\n" + "="*80)
            print("✅ ALL FIXES APPLIED SUCCESSFULLY")
            print("="*80)
            print("\nNext steps:")
            print("1. Transfer fixed files to remote:")
            print("   scp src/model_utility_configs.py remote:/path/to/project/src/")
            print("   scp src/train_single_config_remote.py remote:/path/to/project/src/")
            print("   scp src/compat_trainer.py remote:/path/to/project/src/")
            print("\n2. On remote, start training:")
            print("   cd /datas/store162/xrick/prjs/Reproduce_English_Pronunciation_Evaluation")
            print("   rm -rf src/output/paper_r64/")
            print("   source venv/bin/activate")
            print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0")
            print("\n3. Expected output:")
            print("   🔧 Applying LoRA configuration to model...")
            print("   ✅ LoRA configuration applied - parameters are now trainable")
            print("   可訓練參數: 200,000,000 (3.5%)  ← MUST BE ~200M, NOT 0!")
            print("   loss: 6.98 → 6.42 → 5.89  ← MUST DECREASE")
            print("="*80)
        else:
            print("\n❌ FIXES FAILED - Check errors above")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
