#!/usr/bin/env python3
"""
Fix Processor Save Error

The Phi4MMProcessor has a bug where save_pretrained() fails with:
AttributeError: 'Phi4MMProcessor' object has no attribute 'audio_tokenizer'

This script patches train_single_config_remote.py to handle the error gracefully.
The model and LoRA adapters are saved correctly - only processor save fails.
"""

import re
import sys
from pathlib import Path

def fix_processor_save(script_path):
    """Add try/except around processor.save_pretrained()"""

    print("="*80)
    print("Fix: Processor Save Error (Phi4MMProcessor Bug)")
    print("="*80)

    if not script_path.exists():
        print(f"\n❌ Script not found: {script_path}")
        return False

    # Read file
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already fixed
    if 'except AttributeError' in content and 'processor.save_pretrained' in content:
        print(f"\n✅ Already fixed: try/except wrapper exists")
        return True

    # Backup
    backup_path = str(script_path) + '.processor_fix.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"💾 Backup created: {backup_path}")

    # Find processor.save_pretrained and wrap in try/except
    # Pattern: processor.save_pretrained(final_model_dir)
    pattern = r'(\s+)(processor\.save_pretrained\(final_model_dir\))'

    replacement = r'''\1try:
\1    \2
\1except AttributeError as e:
\1    print(f"⚠️  Processor save failed (known Phi4MMProcessor bug): {e}")
\1    print("✅ Model saved successfully - processor can be loaded from base model")'''

    new_content, count = re.subn(pattern, replacement, content)

    if count > 0:
        # Write patched content
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ Patched {count} occurrence(s)")
        print(f"✅ Wrapped processor.save_pretrained() in try/except")
        print(f"✅ Updated: {script_path}")
        return True
    else:
        print(f"❌ Could not find processor.save_pretrained()")
        return False


def main():
    script_dir = Path(__file__).parent
    script_path = script_dir / "src" / "train_single_config_remote.py"

    print(f"\nTarget: {script_path}\n")

    if fix_processor_save(script_path):
        print("\n" + "="*80)
        print("✅ FIX COMPLETE")
        print("="*80)
        print("\n📝 Changes:")
        print("   Wrapped processor.save_pretrained() in try/except block")
        print("\n🎯 What this does:")
        print("   - Catches Phi4MMProcessor.save_pretrained() AttributeError")
        print("   - Model and LoRA adapters still saved correctly")
        print("   - Processor can be loaded from base model path later")
        print("   - Training completes without error")
        print("\n💡 Note:")
        print("   This is a cosmetic fix for a known Phi4MMProcessor bug")
        print("   The important parts (model + LoRA) are saved successfully")
        print(f"\n💾 Restore if needed:")
        print(f"   cp {script_path}.processor_fix.backup {script_path}")
    else:
        print("\n❌ Patch failed")
        print("\n💡 Manual fix:")
        print("   Edit src/train_single_config_remote.py")
        print("   Find: processor.save_pretrained(final_model_dir)")
        print("   Wrap in try/except:")
        print("   ")
        print("   try:")
        print("       processor.save_pretrained(final_model_dir)")
        print("   except AttributeError as e:")
        print("       print(f'Processor save failed: {e}')")
        sys.exit(1)


if __name__ == "__main__":
    main()
