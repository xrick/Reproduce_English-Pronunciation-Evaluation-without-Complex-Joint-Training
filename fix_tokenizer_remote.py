#!/usr/bin/env python3
"""
Quick tokenizer.json fix for remote machine

Error: Exception: expected value at line 1 column 1
Cause: tokenizer.json is corrupted or empty
Fix: Re-download tokenizer files from HuggingFace
"""

import os
import sys
import json
from pathlib import Path

def find_model_path_in_config():
    """Extract model_path from model_utility_configs.py"""
    config_file = Path(__file__).parent / "src" / "model_utility_configs.py"

    if not config_file.exists():
        return None

    with open(config_file, 'r') as f:
        content = f.read()

    # Find model_path assignments
    import re
    matches = re.findall(r'model_path\s*=\s*["\'](.+?)["\']', content)

    if matches:
        # Return first match (should be the same for both configs)
        return matches[0]

    return None


def check_tokenizer(model_path):
    """Check if tokenizer.json is valid"""
    tokenizer_file = Path(model_path) / "tokenizer.json"

    if not tokenizer_file.exists():
        return False, "tokenizer.json does not exist"

    try:
        with open(tokenizer_file, 'r') as f:
            json.load(f)
        return True, "tokenizer.json is valid"
    except json.JSONDecodeError as e:
        return False, f"tokenizer.json is corrupted: {e}"
    except Exception as e:
        return False, f"Error reading tokenizer.json: {e}"


def fix_tokenizer(model_path):
    """Re-download tokenizer files"""
    print(f"\n🔧 Fixing tokenizer for: {model_path}")

    # Change to model directory
    original_dir = os.getcwd()
    os.chdir(model_path)

    try:
        print("\n📥 Downloading tokenizer files from HuggingFace...")

        from huggingface_hub import hf_hub_download

        files = [
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        for filename in files:
            print(f"\n  Downloading {filename}...")
            try:
                hf_hub_download(
                    repo_id="microsoft/Phi-4-multimodal-instruct",
                    filename=filename,
                    local_dir=".",
                    local_dir_use_symlinks=False,
                    force_download=True
                )
                print(f"  ✅ {filename} downloaded")
            except Exception as e:
                print(f"  ❌ Failed to download {filename}: {e}")
                return False

        print("\n✅ Tokenizer files re-downloaded successfully")
        return True

    except ImportError:
        print("\n❌ huggingface_hub not installed")
        print("   Install: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        return False
    finally:
        os.chdir(original_dir)


def main():
    print("="*80)
    print("Tokenizer.json Quick Fix for Remote Machine")
    print("="*80)

    # Get model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"\n📌 Using provided path: {model_path}")
    else:
        model_path = find_model_path_in_config()
        if model_path:
            print(f"\n📌 Found model_path in config: {model_path}")
        else:
            print("\n❌ Could not find model_path")
            print("\nUsage: python fix_tokenizer_remote.py [model_path]")
            print("\nExample:")
            print("  python fix_tokenizer_remote.py /datas/store162/xrick/models/Phi-4-multimodal-instruct")
            sys.exit(1)

    # Check if path exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model directory does not exist: {model_path}")
        sys.exit(1)

    # Check tokenizer
    print(f"\n🔍 Checking tokenizer at: {model_path}")
    is_valid, message = check_tokenizer(model_path)
    print(f"   {message}")

    if is_valid:
        print("\n✅ Tokenizer is already valid, no fix needed")
        sys.exit(0)

    # Fix tokenizer
    print("\n⚠️  Tokenizer is corrupted, attempting fix...")

    if fix_tokenizer(model_path):
        # Verify fix
        print("\n🔍 Verifying fix...")
        is_valid, message = check_tokenizer(model_path)
        print(f"   {message}")

        if is_valid:
            print("\n" + "="*80)
            print("✅ TOKENIZER FIXED SUCCESSFULLY")
            print("="*80)
            print("\n📝 You can now run training:")
            print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")
        else:
            print("\n❌ Fix failed, tokenizer still invalid")
            sys.exit(1)
    else:
        print("\n❌ Fix failed")
        print("\n💡 Alternative solutions:")
        print("   1. Check internet connection to huggingface.co")
        print("   2. Transfer tokenizer.json from working machine:")
        print("      scp /path/to/working/model/tokenizer.json user@remote:{model_path}/")
        print("   3. Use online model path (auto-downloads on first use):")
        print("      Edit src/model_utility_configs.py")
        print("      Change: model_path = 'microsoft/phi-4-multimodal-instruct'")
        sys.exit(1)


if __name__ == "__main__":
    main()
