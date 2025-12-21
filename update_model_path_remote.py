#!/usr/bin/env python3
"""
Update model paths for remote machine training

This script updates model_utility_configs.py to use the correct model path
based on where the user downloaded the model on the remote machine.

Usage:
  python update_model_path_remote.py [model_path]

Examples:
  python update_model_path_remote.py /datas/store162/xrick/models/Phi-4-multimodal-instruct
  python update_model_path_remote.py microsoft/phi-4-multimodal-instruct  # Use online model
"""

import sys
import os
import re
from pathlib import Path

def update_model_paths(config_file: str, new_path: str, verify_exists: bool = False):
    """
    Update all model_path assignments in the config file.

    Args:
        config_file: Path to model_utility_configs.py
        new_path: New model path to use
        verify_exists: If True, verify the path exists (skip for online models)
    """

    # Verify new path if it's a local path
    if verify_exists and not new_path.startswith("microsoft/"):
        if not os.path.exists(new_path):
            print(f"⚠️  Warning: Model path does not exist: {new_path}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("❌ Aborted")
                return False

    # Read the file
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace all model_path assignments
    # Pattern matches: model_path = "..." or model_path = '...'
    pattern = r'model_path\s*=\s*["\'].*?["\']'
    replacement = f'model_path = "{new_path}"'

    # Count matches
    matches = re.findall(pattern, content)
    if not matches:
        print(f"❌ No model_path found in {config_file}")
        return False

    print(f"\n🔍 Found {len(matches)} model_path assignments:")
    for i, match in enumerate(matches, 1):
        print(f"  {i}. {match}")

    # Replace all occurrences
    new_content = re.sub(pattern, replacement, content)

    # Backup original
    backup_file = config_file + '.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\n💾 Backup saved: {backup_file}")

    # Write updated content
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✅ Updated {len(matches)} occurrences to: {new_path}")
    return True


def detect_model_path():
    """
    Try to detect where the model might be on the remote machine.

    Returns common paths to check.
    """
    common_paths = [
        "/datas/store162/xrick/models/Phi-4-multimodal-instruct",
        "/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct",
        os.path.expanduser("~/models/Phi-4-multimodal-instruct"),
        os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct"),
    ]

    print("\n🔍 Checking common model locations...")
    found_paths = []

    for path in common_paths:
        if os.path.exists(path):
            config_json = os.path.join(path, "config.json")
            if os.path.exists(config_json):
                print(f"✅ Found valid model at: {path}")
                found_paths.append(path)
            else:
                print(f"⚠️  Directory exists but no config.json: {path}")
        else:
            print(f"❌ Not found: {path}")

    return found_paths


def main():
    script_dir = Path(__file__).parent
    config_file = script_dir / "src" / "model_utility_configs.py"

    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)

    print("="*80)
    print("Update Model Path for Remote Training")
    print("="*80)

    # Check if path provided as argument
    if len(sys.argv) > 1:
        new_path = sys.argv[1]
        print(f"\n📌 Using provided path: {new_path}")
    else:
        # Try to detect model path
        found_paths = detect_model_path()

        if found_paths:
            print(f"\n📋 Found {len(found_paths)} valid model location(s)")
            if len(found_paths) == 1:
                new_path = found_paths[0]
                print(f"✅ Auto-selected: {new_path}")
            else:
                print("\nMultiple locations found:")
                for i, path in enumerate(found_paths, 1):
                    print(f"  {i}. {path}")
                choice = input(f"\nSelect location (1-{len(found_paths)}): ")
                try:
                    idx = int(choice) - 1
                    new_path = found_paths[idx]
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    sys.exit(1)
        else:
            print("\n❌ No local model found")
            print("\n💡 Options:")
            print("  1. Specify model path: python update_model_path_remote.py /path/to/model")
            print("  2. Use online model: python update_model_path_remote.py microsoft/phi-4-multimodal-instruct")

            choice = input("\nEnter model path (or press Enter to use online model): ").strip()
            if not choice:
                new_path = "microsoft/phi-4-multimodal-instruct"
                print(f"✅ Using online model: {new_path}")
            else:
                new_path = choice

    # Determine if this is a local path that should be verified
    verify_exists = not new_path.startswith("microsoft/")

    # Update the config file
    if update_model_paths(str(config_file), new_path, verify_exists):
        print("\n" + "="*80)
        print("✅ Configuration Updated Successfully")
        print("="*80)

        if new_path.startswith("microsoft/"):
            print("\n📝 Next Steps (Using Online Model):")
            print("  1. Ensure internet connectivity to huggingface.co")
            print("  2. First run will download ~15GB (15-20 minutes)")
            print("  3. Model cached at: ~/.cache/huggingface/hub/")
            print("  4. Start training:")
            print(f"     python src/train_single_config_remote.py --config paper_r64 --gpus 0")
        else:
            print("\n📝 Next Steps (Using Local Model):")
            print("  1. Verify model files are complete:")
            print(f"     ls -lh {new_path}/config.json")
            print(f"     ls -lh {new_path}/tokenizer.json")
            print("  2. Start training:")
            print(f"     python src/train_single_config_remote.py --config paper_r64 --gpus 0")

        print(f"\n💾 Backup saved at: {config_file}.backup")
        print(f"   (Restore: cp {config_file}.backup {config_file})")
    else:
        print("\n❌ Update failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
