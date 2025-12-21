#!/usr/bin/env python3
"""
Verify FP16 setup for TITAN RTX training

Checks:
1. Model dtype is FP16
2. GPU compute capability
3. Training args will use FP16
"""

import torch
import sys

def check_gpu():
    """Check GPU capabilities"""
    print("="*80)
    print("GPU Check")
    print("="*80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    cc = torch.cuda.get_device_capability()
    print(f"✅ CUDA available")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {cc}")
    print(f"Supports BF16: {'✅ YES' if cc[0] >= 8 else '❌ NO (Turing)'}")
    print(f"Supports FP16: ✅ YES")
    print(f"\n💡 Recommended: FP16 (BF16 not supported)")

    return True


def check_model_dtype():
    """Check model loading dtype"""
    print("\n" + "="*80)
    print("Model Dtype Check")
    print("="*80)

    try:
        from src.model_utility_configs import CONFIGS

        print("Loading paper_r64 configuration...")
        config = CONFIGS["paper_r64"]
        model, processor, peft_config = config["loader"]()

        # Check model dtype
        param = next(model.parameters())
        dtype = param.dtype

        print(f"\n✅ Model loaded successfully")
        print(f"Model dtype: {dtype}")

        if dtype == torch.float16:
            print("✅ CORRECT: Model is in FP16")
            return True
        elif dtype == torch.bfloat16:
            print("❌ WRONG: Model is in BF16 (will fail on TITAN RTX)")
            print("\n💡 Fix: Run fix_bf16_to_fp16.py")
            return False
        else:
            print(f"⚠️  Unexpected dtype: {dtype}")
            return False

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def check_training_command():
    """Show recommended training command"""
    print("\n" + "="*80)
    print("Training Command")
    print("="*80)

    print("\n✅ CORRECT command (with --fp16 flag):")
    print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")

    print("\n❌ WRONG command (without --fp16 flag):")
    print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0")
    print("   (Will cause: ValueError: Attempting to unscale FP16 gradients)")


def main():
    print("\n🔍 FP16 Setup Verification for TITAN RTX\n")

    # Check 1: GPU
    gpu_ok = check_gpu()

    # Check 2: Model dtype
    model_ok = check_model_dtype()

    # Check 3: Training command
    check_training_command()

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    if gpu_ok and model_ok:
        print("\n✅ Setup is CORRECT")
        print("\n🚀 Ready to train with:")
        print("   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")
    elif not model_ok:
        print("\n❌ Setup INCORRECT - Model dtype issue")
        print("\n🔧 Fix:")
        print("   1. Run: python fix_bf16_to_fp16.py")
        print("   2. Then train with: python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16")
    else:
        print("\n⚠️  GPU check failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
