#!/usr/bin/env python3
"""
Test script to verify model loading with built-in LoRA configuration
"""

import sys
sys.path.insert(0, '/Users/xrickliao/WorkSpaces/ResearchCodes/Reproduce_English_Pronunciation_Evaluation_without_Complex_Joint_Training/src')

try:
    from model_utility import get_model_and_processor

    print("="*80)
    print("Testing Phi-4-multimodal model loading with configured built-in LoRA...")
    print("="*80)

    model, processor, peft_config = get_model_and_processor()

    print("\n✅ SUCCESS: Model loaded successfully!")
    print("\nModel type:", type(model))

    # Count trainable parameters
    trainable_params = [(name, param.shape) for name, param in model.named_parameters() if param.requires_grad]
    trainable_count = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    all_count = sum(p.numel() for _, p in model.named_parameters())

    print(f"\nTrainable parameters: {trainable_count:,} / {all_count:,} ({100*trainable_count/all_count:.4f}%)")

    if trainable_params:
        print("\n" + "="*80)
        print("Trainable parameter layers:")
        print("="*80)
        for name, shape in trainable_params[:20]:  # Show first 20
            print(f"  {name} - {shape}")
        if len(trainable_params) > 20:
            print(f"  ... and {len(trainable_params) - 20} more")
    else:
        print("\n⚠️  WARNING: No trainable parameters found!")

except Exception as e:
    print("\n❌ FAILED:", str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
