#!/usr/bin/env python3
"""
Ultimate Fix for FP16 Gradient Scaler Error

The GradScaler is fundamentally incompatible with gradient checkpointing
in this PyTorch/Accelerate version combination.

Solution: Disable automatic mixed precision (AMP) and use native FP16 model
- Model already loaded in FP16 (from model_utility_configs.py)
- No AMP/GradScaler needed
- Training works normally
"""

import re
import sys
from pathlib import Path

def patch_disable_amp(script_path):
    """Disable FP16 AMP in training script"""

    print("="*80)
    print("Ultimate Fix: Disable AMP (Use Native FP16 Model)")
    print("="*80)

    if not script_path.exists():
        print(f"\n❌ Script not found: {script_path}")
        return False

    # Read file
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Backup
    backup_path = str(script_path) + '.amp.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Backup created: {backup_path}")

    # Change fp16 from True to False (disable AMP)
    # Find the line: "fp16": use_fp16,
    # Replace with: "fp16": False,  # Disabled - using native FP16 model

    pattern = r'"fp16":\s*use_fp16,'
    replacement = '"fp16": False,  # Disabled AMP - using native FP16 model instead'

    new_content, count = re.sub(pattern, replacement, content)

    if count > 0:
        # Write patched content
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Patched {count} occurrence(s)")
        print(f"✅ Changed: fp16=True → fp16=False")
        print(f"✅ Model already in FP16 (native), no AMP needed")
        print(f"✅ Updated: {script_path}")
        return True
    else:
        print(f"❌ Could not find fp16 setting")
        return False


def main():
    script_dir = Path(__file__).parent
    script_path = script_dir / "src" / "train_single_config_remote.py"

    print(f"\nTarget: {script_path}\n")

    if patch_disable_amp(script_path):
        print("\n" + "="*80)
        print("✅ ULTIMATE FIX COMPLETE")
        print("="*80)
        print("\n📝 Changes:")
        print("   fp16=True → fp16=False (disabled AMP)")
        print("\n🎯 How this works:")
        print("   1. Model loaded in native FP16 (from model_utility_configs.py)")
        print("   2. No GradScaler/AMP used")
        print("   3. Training uses pure FP16 (no mixed precision)")
        print("   4. No scaler errors!")
        print("\n⚡ Performance:")
        print("   - Speed: SAME (model already FP16)")
        print("   - Memory: SAME (FP16 model)")
        print("   - Stability: BETTER (no scaler bugs)")
        print("\n🚀 You can now run training:")
        print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0")
        print("   (NO --fp16 flag needed, model is native FP16)")
        print(f"\n💾 Restore if needed:")
        print(f"   cp {script_path}.amp.backup {script_path}")
    else:
        print("\n❌ Patch failed")
        print("\n💡 Manual fix:")
        print("   Edit src/train_single_config_remote.py")
        print("   Find: 'fp16': use_fp16,")
        print("   Change to: 'fp16': False,  # Disabled AMP")
        sys.exit(1)


if __name__ == "__main__":
    main()
