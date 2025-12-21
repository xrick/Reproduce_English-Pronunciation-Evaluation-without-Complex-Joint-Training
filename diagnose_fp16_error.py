#!/usr/bin/env python3
"""
Emergency Diagnostic for FP16 Gradient Scaler Error

Checks all possible causes and provides specific fix
"""

import torch
import sys
import os

def check_model_dtype():
    """Check if model loads in FP16"""
    print("="*80)
    print("1. Model Dtype Check")
    print("="*80)

    try:
        sys.path.insert(0, 'src')
        from model_utility_configs import CONFIGS

        config = CONFIGS["paper_r64"]
        model, processor, peft_config = config["loader"]()

        param = next(model.parameters())
        dtype = param.dtype

        print(f"Model dtype: {dtype}")

        if dtype == torch.float16:
            print("✅ Model is correctly in FP16")
            return True
        elif dtype == torch.bfloat16:
            print("❌ Model is in BF16 (WRONG for TITAN RTX)")
            print("\n🔧 Fix:")
            print("   python fix_bf16_to_fp16.py")
            return False
        else:
            print(f"⚠️  Unexpected dtype: {dtype}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_gpu():
    """Check GPU compute capability"""
    print("\n" + "="*80)
    print("2. GPU Capability Check")
    print("="*80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    cc = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {cc}")
    print(f"Supports BF16: {'YES' if cc[0] >= 8 else 'NO'}")
    print(f"Supports FP16: YES")

    if cc[0] < 8:
        print("\n💡 TITAN RTX (Turing 7.5) MUST use FP16")
        print("   Required flag: --fp16")

    return True


def check_training_args_logic():
    """Show what training args will be set"""
    print("\n" + "="*80)
    print("3. Training Args Logic")
    print("="*80)

    # Simulate WITHOUT --fp16
    print("\n❌ WITHOUT --fp16 flag:")
    print("   use_bf16 = not False → True")
    print("   use_fp16 = False → False")
    print("   [Turing detection]")
    print("   use_bf16 = False (after detection)")
    print("   use_fp16 = True (after detection)")
    print("   Result: MIXED SIGNALS → Scaler Error")

    # Simulate WITH --fp16
    print("\n✅ WITH --fp16 flag:")
    print("   use_bf16 = not True → False")
    print("   use_fp16 = True → True")
    print("   [No BF16 detection needed]")
    print("   Result: CLEAN FP16 → Works")


def show_correct_command():
    """Show correct training command"""
    print("\n" + "="*80)
    print("4. Correct Training Command")
    print("="*80)

    print("\n✅ CORRECT (with --fp16):")
    print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")

    print("\n❌ WRONG (without --fp16):")
    print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0")

    print("\n💡 Or use wrapper script:")
    print("   bash start_training_remote.sh")


def check_model_config_file():
    """Check if model_utility_configs.py has been fixed"""
    print("\n" + "="*80)
    print("5. Model Config File Check")
    print("="*80)

    config_file = "src/model_utility_configs.py"

    if not os.path.exists(config_file):
        print(f"❌ File not found: {config_file}")
        return False

    with open(config_file, 'r') as f:
        content = f.read()

    bf16_count = content.count('torch.bfloat16')
    fp16_count = content.count('torch.float16')

    print(f"torch.bfloat16 occurrences: {bf16_count}")
    print(f"torch.float16 occurrences: {fp16_count}")

    if bf16_count > 0 and 'torch_dtype=torch.bfloat16' in content:
        print("❌ File still has torch.bfloat16 in model loading")
        print("\n🔧 Fix:")
        print("   python fix_bf16_to_fp16.py")
        return False
    elif fp16_count >= 2:
        print("✅ File correctly uses torch.float16")
        return True
    else:
        print("⚠️  Cannot determine dtype setting")
        return False


def main():
    print("\n🔍 FP16 Gradient Scaler Error Diagnostic\n")

    # Run checks
    check1 = check_gpu()
    check2 = check_model_config_file()
    check3 = check_model_dtype()

    check_training_args_logic()
    show_correct_command()

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    if check1 and check2 and check3:
        print("\n✅ All checks PASSED")
        print("\n🎯 The issue is: YOU'RE NOT USING --fp16 FLAG")
        print("\n🔧 Solution:")
        print("   1. Add --fp16 to your training command:")
        print("      python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")
        print("\n   2. Or use wrapper script:")
        print("      bash start_training_remote.sh")
    elif not check2 or not check3:
        print("\n❌ Model configuration issue detected")
        print("\n🔧 Fix:")
        print("   1. Run: python fix_bf16_to_fp16.py")
        print("   2. Then train with: python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")
    else:
        print("\n⚠️  Some checks failed")
        print("   Review output above for specific issues")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
