#!/usr/bin/env python3
"""
CRITICAL FIX: Enable LoRA Parameter Training for paper_r64 Configuration

Problem:
--------
Training completed with loss=0.0 because LoraConfig was created but never applied to model.
Warning: "None of the inputs have requires_grad=True. Gradients will be None"

Root Cause:
-----------
src/model_utility_configs.py lines 178-185:
- LoraConfig created with r=64, alpha=128
- But missing: model = get_peft_model(model, peft_config)
- LoRA parameters exist from pretrained weights but are FROZEN

Solution:
---------
Add get_peft_model() call to properly apply LoRA configuration and enable training.
"""

import re
import sys
from pathlib import Path

def fix_lora_training(config_file):
    """Apply PEFT LoRA configuration to model"""

    print("="*80)
    print("CRITICAL FIX: Enable LoRA Training for paper_r64")
    print("="*80)

    if not config_file.exists():
        print(f"\n❌ File not found: {config_file}")
        return False

    # Read file
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already fixed
    if 'model = get_peft_model(model, peft_config)' in content:
        print(f"\n✅ Already fixed: get_peft_model() call exists")
        return True

    # Backup
    backup_path = str(config_file) + '.lora_fix.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Backup created: {backup_path}")

    # Find the section to patch
    # After LoraConfig creation (line 185), before stats collection (line 187)
    pattern = r'(    peft_config = LoraConfig\(\s+r=64,\s+lora_alpha=128,\s+target_modules="all-linear",\s+lora_dropout=0\.05,\s+bias="none",\s+task_type="CAUSAL_LM",\s+\))\n\n(    # 統計參數)'

    replacement = r'''\1

    # 🔧 CRITICAL FIX: Apply LoRA configuration to enable training
    print("\\n🔧 Applying LoRA configuration to model...")
    model = get_peft_model(model, peft_config)
    print("✅ LoRA configuration applied - parameters are now trainable")

\2'''

    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if count > 0:
        # Write patched content
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Patched {count} occurrence(s)")
        print(f"✅ Added: model = get_peft_model(model, peft_config)")
        print(f"✅ Location: After LoraConfig creation, before stats collection")
        print(f"✅ Updated: {config_file}")
        return True
    else:
        print(f"❌ Could not find insertion point")
        print(f"   Looking for: LoraConfig creation pattern")

        # Try alternative pattern (more flexible)
        alt_pattern = r'(    peft_config = LoraConfig\([^)]+\))\n'

        alt_replacement = r'''\1

    # 🔧 CRITICAL FIX: Apply LoRA configuration to enable training
    print("\\n🔧 Applying LoRA configuration to model...")
    model = get_peft_model(model, peft_config)
    print("✅ LoRA configuration applied - parameters are now trainable")
'''

        new_content, alt_count = re.subn(alt_pattern, alt_replacement, content, flags=re.MULTILINE | re.DOTALL)

        if alt_count > 0:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ Applied alternative patch ({alt_count} occurrence(s))")
            print(f"✅ Updated: {config_file}")
            return True
        else:
            return False


def main():
    script_dir = Path(__file__).parent
    config_file = script_dir / "src" / "model_utility_configs.py"

    print(f"\nTarget: {config_file}\n")

    if fix_lora_training(config_file):
        print("\n" + "="*80)
        print("✅ CRITICAL FIX COMPLETE")
        print("="*80)
        print("\n📝 Changes:")
        print("   Added: model = get_peft_model(model, peft_config)")
        print("   Location: src/model_utility_configs.py after line 185")
        print("\n🎯 What this does:")
        print("   1. Properly applies LoraConfig to the model")
        print("   2. Enables gradient computation on LoRA parameters")
        print("   3. Sets requires_grad=True for trainable LoRA layers")
        print("   4. Fixes 'loss=0.0' and 'no gradients' warning")
        print("\n⚡ Expected results:")
        print("   - Trainable parameters: ~200M (3.5%)")
        print("   - Non-zero loss during training")
        print("   - Gradients computed and applied")
        print("   - Actual learning occurs")
        print("\n🚀 You can now restart training:")
        print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0")
        print(f"\n💾 Restore if needed:")
        print(f"   cp {config_file}.lora_fix.backup {config_file}")
        print("\n⚠️  IMPORTANT:")
        print("   Delete old checkpoint directory before retraining:")
        print("   rm -rf src/output/paper_r64/checkpoint-*")
    else:
        print("\n❌ Patch failed")
        print("\n💡 Manual fix:")
        print("   Edit src/model_utility_configs.py")
        print("   After line 185 (LoraConfig creation), add:")
        print("   ")
        print("   # Apply LoRA configuration to enable training")
        print("   model = get_peft_model(model, peft_config)")
        sys.exit(1)


if __name__ == "__main__":
    main()
