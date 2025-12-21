#!/usr/bin/env python3
"""
Fix BF16 → FP16 for NVIDIA TITAN RTX

Error: NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
Cause: model_utility_configs.py hardcodes torch_dtype=torch.bfloat16
Fix: Change to torch.float16 for Turing GPUs
"""

import re
import sys
from pathlib import Path

def fix_dtype_in_config(config_file):
    """Replace torch.bfloat16 with torch.float16 in model_utility_configs.py"""

    print("="*80)
    print("Fix BF16 → FP16 for TITAN RTX")
    print("="*80)

    if not config_file.exists():
        print(f"\n❌ File not found: {config_file}")
        return False

    # Read file
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check current dtype
    if 'torch.bfloat16' not in content:
        print(f"\n✅ File already uses torch.float16 or no dtype specified")
        return True

    # Count occurrences
    bf16_count = content.count('torch.bfloat16')
    print(f"\n🔍 Found {bf16_count} occurrence(s) of torch.bfloat16")

    # Backup
    backup_file = str(config_file) + '.bf16.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Backup created: {backup_file}")

    # Replace
    new_content = content.replace('torch.bfloat16', 'torch.float16')

    # Write
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✅ Replaced {bf16_count} occurrence(s) with torch.float16")
    print(f"✅ Updated: {config_file}")

    return True


def main():
    script_dir = Path(__file__).parent
    config_file = script_dir / "src" / "model_utility_configs.py"

    print(f"\nTarget file: {config_file}\n")

    if fix_dtype_in_config(config_file):
        print("\n" + "="*80)
        print("✅ FIX COMPLETE")
        print("="*80)
        print("\n📝 Changes made:")
        print("   torch.bfloat16 → torch.float16")
        print("\n🎯 This fixes:")
        print("   - BF16 incompatibility with TITAN RTX (Turing)")
        print("   - NotImplementedError during gradient scaling")
        print("\n🚀 You can now run training:")
        print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")
        print(f"\n💾 Restore backup if needed:")
        print(f"   cp {config_file}.bf16.backup {config_file}")
    else:
        print("\n❌ Fix failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
