#!/usr/bin/env python3
"""Check what LoRA parameters exist in the model"""

import sys
sys.path.insert(0, '/Users/xrickliao/WorkSpaces/ResearchCodes/Reproduce_English_Pronunciation_Evaluation_without_Complex_Joint_Training/src')

from model_utility import get_model_and_processor

print("Loading model...")
model, processor, peft_config = get_model_and_processor()

print("\n" + "="*80)
print("Searching for LoRA-related parameters...")
print("="*80)

lora_params = []
for name, param in model.named_parameters():
    if "lora" in name.lower():
        lora_params.append((name, param.shape, param.requires_grad, param.dtype))

if lora_params:
    print(f"\nFound {len(lora_params)} LoRA parameters:")
    for name, shape, requires_grad, dtype in lora_params[:30]:
        grad_status = "✓ trainable" if requires_grad else "✗ frozen"
        print(f"  {grad_status} | {dtype} | {name} - {shape}")
    if len(lora_params) > 30:
        print(f"  ... and {len(lora_params) - 30} more")
else:
    print("\n⚠️  No LoRA parameters found in model!")
    print("\nThis suggests Phi-4's built-in LoRA system may not be working as expected.")
    print("The model may need to be initialized differently to enable LoRA adapters.")
