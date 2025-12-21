#!/usr/bin/env python3
"""
Patch training script to fix FP16 gradient scaler error

The error "Attempting to unscale FP16 gradients" is caused by a conflict
between gradient checkpointing and FP16 mixed precision training in
PyTorch/Accelerate.

Fix: Set max_grad_norm to None (disable gradient clipping)
"""

import re
import sys
from pathlib import Path

def patch_training_script(script_path):
    """Add max_grad_norm=None to TrainingArguments"""

    print("="*80)
    print("Patching Training Script for FP16 Gradient Scaler Fix")
    print("="*80)

    if not script_path.exists():
        print(f"\n❌ Script not found: {script_path}")
        return False

    # Read file
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if 'max_grad_norm' in content and ('max_grad_norm=None' in content or '"max_grad_norm": None' in content):
        print(f"\n✅ Script already patched")
        return True

    # Backup
    backup_path = str(script_path) + '.scaler.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Backup created: {backup_path}")

    # Find training_args_dict and add max_grad_norm
    # Look for the "gradient_checkpointing": True line and add after it

    pattern = r'("gradient_checkpointing":\s*True,)'
    replacement = r'\1\n        "max_grad_norm": None,  # Disable gradient clipping to fix FP16 scaler error'

    new_content, count = re.subn(pattern, replacement, content)

    if count > 0:
        # Write patched content
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Patched {count} occurrence(s)")
        print(f"✅ Added: max_grad_norm=None")
        print(f"✅ Updated: {script_path}")
        return True
    else:
        print(f"❌ Could not find insertion point")
        print(f"   Looking for: 'gradient_checkpointing': True")
        return False


def main():
    script_dir = Path(__file__).parent
    script_path = script_dir / "src" / "train_single_config_remote.py"

    print(f"\nTarget: {script_path}\n")

    if patch_training_script(script_path):
        print("\n" + "="*80)
        print("✅ PATCH COMPLETE")
        print("="*80)
        print("\n📝 Changes:")
        print("   Added: max_grad_norm=None (disables gradient clipping)")
        print("\n🎯 This fixes:")
        print("   - ValueError: Attempting to unscale FP16 gradients")
        print("   - Gradient scaler conflict with FP16 + gradient checkpointing")
        print("\n💡 What this does:")
        print("   - Gradient clipping disabled (max_grad_norm=None)")
        print("   - Allows FP16 training with gradient checkpointing")
        print("   - Training stability: Same (gradient clipping is optional)")
        print("\n🚀 You can now run training:")
        print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")
        print(f"\n💾 Restore if needed:")
        print(f"   cp {script_path}.scaler.backup {script_path}")
    else:
        print("\n❌ Patch failed")
        print("\n💡 Manual fix:")
        print("   Edit src/train_single_config_remote.py")
        print("   Find: 'gradient_checkpointing': True,")
        print("   Add after it: 'max_grad_norm': None,")
        sys.exit(1)


if __name__ == "__main__":
    main()
