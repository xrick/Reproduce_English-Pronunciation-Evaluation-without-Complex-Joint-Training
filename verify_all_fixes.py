#!/usr/bin/env python3
"""
Comprehensive Verification: All Remote Fixes Applied

Checks that all 5 critical fixes are correctly applied:
1. ✅ Tokenizer files (re-downloaded)
2. ✅ AudioDataCollator API (tuple format)
3. ✅ FP16 dtype (not BF16 for TITAN RTX)
4. ✅ AMP disabled (native FP16, no GradScaler)
5. ✅ LoRA training enabled (get_peft_model applied)
"""

import sys
from pathlib import Path
import json

def check_tokenizer():
    """Fix 1: Tokenizer files present and valid"""
    print("\n" + "="*80)
    print("Fix 1: Tokenizer Files")
    print("="*80)

    model_path = Path.home() / ".cache/huggingface/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots"

    # Find latest snapshot
    if not model_path.exists():
        print("❌ Model cache directory not found")
        return False

    snapshots = list(model_path.iterdir())
    if not snapshots:
        print("❌ No model snapshots found")
        return False

    latest = max(snapshots, key=lambda p: p.stat().st_mtime)
    tokenizer_json = latest / "tokenizer.json"

    if not tokenizer_json.exists():
        print(f"❌ tokenizer.json not found in {latest}")
        return False

    # Validate JSON
    try:
        with open(tokenizer_json, 'r') as f:
            data = json.load(f)
        if "model" in data and "vocab" in data:
            print(f"✅ tokenizer.json valid: {tokenizer_json}")
            print(f"   Size: {tokenizer_json.stat().st_size:,} bytes")
            return True
        else:
            print(f"❌ tokenizer.json malformed (missing required keys)")
            return False
    except Exception as e:
        print(f"❌ tokenizer.json corrupted: {e}")
        return False


def check_audio_collator():
    """Fix 2: AudioDataCollator using tuple format"""
    print("\n" + "="*80)
    print("Fix 2: AudioDataCollator API")
    print("="*80)

    script_path = Path("src/AudioDataCollator.py")
    if not script_path.exists():
        print(f"❌ File not found: {script_path}")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    # Check for tuple format
    if "audios.append((audio_array," in content or "audios.append((f[\"audio_array\"]," in content:
        print(f"✅ AudioDataCollator uses tuple format")
        print(f"   Pattern: audios.append((audio_array, sampling_rate))")
        return True
    else:
        print(f"❌ AudioDataCollator NOT using tuple format")
        print(f"   Missing: audios.append((audio_array, sampling_rate))")
        return False


def check_fp16_dtype():
    """Fix 3: torch.float16 (not bfloat16) for TITAN RTX"""
    print("\n" + "="*80)
    print("Fix 3: FP16 Dtype (TITAN RTX Compatibility)")
    print("="*80)

    script_path = Path("src/model_utility_configs.py")
    if not script_path.exists():
        print(f"❌ File not found: {script_path}")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    # Check for bfloat16 in model loading
    if "torch.bfloat16" in content:
        lines_with_bf16 = [i+1 for i, line in enumerate(content.split('\n')) if "torch.bfloat16" in line and not line.strip().startswith("#")]
        if lines_with_bf16:
            print(f"❌ Still using torch.bfloat16 at lines: {lines_with_bf16}")
            print(f"   TITAN RTX (Turing 7.5) does NOT support BF16")
            return False

    # Check for float16
    if "torch.float16" in content:
        print(f"✅ Using torch.float16 (correct for TITAN RTX)")
        return True
    else:
        print(f"⚠️  Warning: torch.float16 not found (check dtype setting)")
        return False


def check_amp_disabled():
    """Fix 4: AMP disabled (fp16=False in training args)"""
    print("\n" + "="*80)
    print("Fix 4: AMP Disabled (Native FP16)")
    print("="*80)

    script_path = Path("src/train_single_config_remote.py")
    if not script_path.exists():
        print(f"❌ File not found: {script_path}")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    # Check for fp16=False with AMP disable comment
    if '"fp16": False' in content and "Disabled AMP" in content:
        print(f"✅ AMP disabled: fp16=False")
        print(f"   Using native FP16 model (no GradScaler)")
        return True
    elif '"fp16": use_fp16' in content:
        print(f"❌ AMP still enabled: fp16=use_fp16")
        print(f"   Should be: fp16=False (disable AMP)")
        return False
    else:
        print(f"⚠️  Warning: Cannot determine fp16 setting")
        return False


def check_lora_training():
    """Fix 5: LoRA training enabled (get_peft_model applied)"""
    print("\n" + "="*80)
    print("Fix 5: LoRA Training Enabled")
    print("="*80)

    script_path = Path("src/model_utility_configs.py")
    if not script_path.exists():
        print(f"❌ File not found: {script_path}")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    # Check for get_peft_model call in paper config
    lines = content.split('\n')

    # Find get_model_and_processor_paper function
    in_paper_func = False
    has_peft_config = False
    has_get_peft_model = False

    for i, line in enumerate(lines):
        if "def get_model_and_processor_paper" in line:
            in_paper_func = True
        elif in_paper_func and "def " in line and "get_model_and_processor_paper" not in line:
            break
        elif in_paper_func:
            if "peft_config = LoraConfig" in line:
                has_peft_config = True
            if "model = get_peft_model(model, peft_config)" in line:
                has_get_peft_model = True

    if has_peft_config and has_get_peft_model:
        print(f"✅ LoRA training enabled: get_peft_model() applied")
        print(f"   LoRA parameters will be trainable")
        return True
    elif has_peft_config and not has_get_peft_model:
        print(f"❌ LoRA training DISABLED: get_peft_model() NOT applied")
        print(f"   Found: LoraConfig created")
        print(f"   Missing: model = get_peft_model(model, peft_config)")
        print(f"   Result: Training will fail with loss=0.0")
        return False
    else:
        print(f"❌ LoRA configuration not found")
        return False


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE FIX VERIFICATION")
    print("Checking All 5 Critical Fixes for Remote Training")
    print("="*80)

    results = {
        "Tokenizer Files": check_tokenizer(),
        "AudioDataCollator API": check_audio_collator(),
        "FP16 Dtype": check_fp16_dtype(),
        "AMP Disabled": check_amp_disabled(),
        "LoRA Training": check_lora_training(),
    }

    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for fix, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {fix}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL FIXES VERIFIED - READY TO TRAIN")
        print("="*80)
        print("\n🚀 Start training with:")
        print("   source venv/bin/activate")
        print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0")
        print("\n📊 Expected results:")
        print("   - Trainable parameters: ~200M (3.5%)")
        print("   - Non-zero loss from step 1")
        print("   - Loss decreases over time")
        print("   - No 'requires_grad' warnings")
        sys.exit(0)
    else:
        print("❌ SOME FIXES MISSING - DO NOT START TRAINING")
        print("="*80)
        failed = [fix for fix, status in results.items() if not status]
        print(f"\n⚠️  Failed checks: {', '.join(failed)}")
        print("\n📝 Apply missing fixes:")
        if not results["Tokenizer Files"]:
            print("   python fix_tokenizer_remote.py")
        if not results["AudioDataCollator API"]:
            print("   Transfer AudioDataCollator.py from Mac")
        if not results["FP16 Dtype"]:
            print("   python fix_bf16_to_fp16.py")
        if not results["AMP Disabled"]:
            print("   python patch_disable_amp.py")
        if not results["LoRA Training"]:
            print("   python fix_lora_training.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
