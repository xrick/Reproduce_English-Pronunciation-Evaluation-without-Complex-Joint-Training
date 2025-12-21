# Model Loading for Evaluation: LoRA and Base Models

Complete guide on how to load models for the `evaluate_model()` function.

---

## Overview

When evaluating a trained LoRA model, you have **three loading scenarios**:

1. **Base Model Only** (No LoRA) - Baseline comparison
2. **LoRA-Fine-tuned Model** (After Training) - Your trained model
3. **Pretrained LoRA Model** (Before Training) - Starting point comparison

---

## Scenario 1: Base Model Only (No LoRA)

**Use case**: Baseline performance comparison to show LoRA improvement

### Loading Code

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

def load_base_model():
    """Load base Phi-4-multimodal without any LoRA adapters"""

    model_path = "microsoft/Phi-4-multimodal-instruct"
    # Or local path: "/path/to/models/Phi-4-multimodal-instruct"

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Load base model without LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # or torch.float16 for NVIDIA Turing
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()  # Set to evaluation mode

    return model, processor

# Usage
base_model, processor = load_base_model()
evaluate_model(base_model, test_dataset)
```

**Key points**:
- ✅ No PEFT/LoRA imports needed
- ✅ Direct loading from HuggingFace or local path
- ✅ Use for baseline metrics

---

## Scenario 2: LoRA Fine-tuned Model (After Training)

**Use case**: Evaluate your trained model from checkpoint

### Method A: Load from Checkpoint Directory (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch

def load_finetuned_lora_model(checkpoint_path):
    """
    Load LoRA model from training checkpoint

    Args:
        checkpoint_path: Path to checkpoint directory
                        e.g., "src/output/paper_r64/checkpoint-120"
    """

    # Step 1: Load base model
    base_model_path = "microsoft/Phi-4-multimodal-instruct"

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # Match training dtype
        device_map="auto",
        trust_remote_code=True
    )

    # Step 2: Load LoRA adapters from checkpoint
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,  # Points to checkpoint directory
        is_trainable=False  # Evaluation mode - freeze adapters
    )

    model.eval()

    return model, processor

# Usage
checkpoint = "src/output/paper_r64/checkpoint-120"
model, processor = load_finetuned_lora_model(checkpoint)
evaluate_model(model, test_dataset)
```

**Checkpoint directory structure**:
```
src/output/paper_r64/checkpoint-120/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights
├── optimizer.pt                 # Optimizer state (not needed for eval)
├── rng_state.pth               # RNG state (not needed for eval)
├── scheduler.pt                # Scheduler state (not needed for eval)
└── trainer_state.json          # Training state (not needed for eval)
```

### Method B: Load Final Model (After Training Complete)

```python
def load_final_lora_model(output_dir):
    """
    Load LoRA model from final output directory

    Args:
        output_dir: Training output directory
                   e.g., "src/output/paper_r64"
    """

    base_model_path = "microsoft/Phi-4-multimodal-instruct"

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load from final output directory
    model = PeftModel.from_pretrained(
        base_model,
        output_dir,  # Training output directory
        is_trainable=False
    )

    model.eval()

    return model, processor

# Usage
output_dir = "src/output/paper_r64"
model, processor = load_final_lora_model(output_dir)
evaluate_model(model, test_dataset)
```

**Key points**:
- ✅ `PeftModel.from_pretrained()` loads LoRA adapters
- ✅ Base model + LoRA adapters = complete fine-tuned model
- ✅ Set `is_trainable=False` for evaluation
- ✅ Use `model.eval()` to disable dropout

---

## Scenario 3: Pretrained LoRA Model (Before Training)

**Use case**: Compare against pretrained LoRA checkpoint before fine-tuning

### Loading Code

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch

def load_pretrained_lora_model():
    """Load model with pretrained LoRA adapters (r=320)"""

    # Base model path (has pretrained LoRA weights)
    model_path = "/path/to/Phi-4-multimodal-instruct"

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Load model with pretrained LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Pretrained uses BF16
        device_map="auto",
        trust_remote_code=True
    )

    # Pretrained model already has LoRA layers built-in
    # No need to apply PeftModel.from_pretrained()

    model.eval()

    return model, processor

# Usage
pretrained_model, processor = load_pretrained_lora_model()
evaluate_model(pretrained_model, test_dataset)
```

**Key points**:
- ✅ Pretrained Phi-4-multimodal already includes LoRA weights
- ✅ No need for `PeftModel.from_pretrained()` - LoRA is built-in
- ✅ Use for comparison: "before fine-tuning" vs "after fine-tuning"

---

## Complete Evaluation Script

Here's a complete script that evaluates all three scenarios:

```python
#!/usr/bin/env python3
"""
Complete Model Evaluation Script
Compares: Base Model, Pretrained LoRA, Fine-tuned LoRA
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from datasets import load_dataset
from scipy.stats import pearsonr
import json

def load_base_model(model_path):
    """Scenario 1: Base model without LoRA"""
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    return model, processor


def load_pretrained_lora(model_path):
    """Scenario 3: Pretrained LoRA model (before fine-tuning)"""
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Pretrained uses BF16
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    return model, processor


def load_finetuned_lora(base_model_path, checkpoint_path):
    """Scenario 2: Fine-tuned LoRA model (after training)"""
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False
    )

    model.eval()
    return model, processor


def evaluate_model(model, processor, test_dataset):
    """Evaluate model on test dataset"""
    model.eval()
    predictions = []
    references = []

    for sample in test_dataset:
        # Prepare input
        audio_array = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']

        # Create prompt (APA task)
        prompt = "<|user|><|APA|><|audio_1|>Evaluate pronunciation...<|end|><|assistant|>"

        inputs = processor(
            text=prompt,
            audios=[(audio_array, sampling_rate)],  # Tuple format
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,  # Lower temperature for evaluation
                do_sample=False   # Deterministic for reproducibility
            )

        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # Parse JSON output
        try:
            json_str = generated_text.split("<|assistant|>")[-1].strip()
            pred_json = json.loads(json_str)
            predictions.append(pred_json)
            references.append(sample)
        except Exception as e:
            print(f"Failed to parse: {e}")
            continue

    # Calculate metrics
    pred_acc = [p.get('accuracy', 0) for p in predictions]
    ref_acc = [r['accuracy'] for r in references]

    if len(pred_acc) > 1:
        pcc, p_value = pearsonr(pred_acc, ref_acc)
        print(f"Accuracy PCC: {pcc:.4f} (p={p_value:.4f})")
        return pcc
    else:
        print("Insufficient predictions for PCC calculation")
        return None


def main():
    # Load test dataset
    test_dataset = load_dataset(
        "mispeech/speechocean762",
        split="test"
    )

    base_model_path = "microsoft/Phi-4-multimodal-instruct"
    checkpoint_path = "src/output/paper_r64/checkpoint-120"

    print("="*80)
    print("EVALUATION: Comparing Three Model Configurations")
    print("="*80)

    # Scenario 1: Base model (no LoRA)
    print("\n1️⃣  Base Model (No LoRA)")
    print("-"*80)
    base_model, processor = load_base_model(base_model_path)
    base_pcc = evaluate_model(base_model, processor, test_dataset)
    del base_model
    torch.cuda.empty_cache()

    # Scenario 3: Pretrained LoRA (before fine-tuning)
    print("\n3️⃣  Pretrained LoRA Model (Before Fine-tuning)")
    print("-"*80)
    pretrained_model, processor = load_pretrained_lora(base_model_path)
    pretrained_pcc = evaluate_model(pretrained_model, processor, test_dataset)
    del pretrained_model
    torch.cuda.empty_cache()

    # Scenario 2: Fine-tuned LoRA (after training)
    print("\n2️⃣  Fine-tuned LoRA Model (After Training)")
    print("-"*80)
    finetuned_model, processor = load_finetuned_lora(
        base_model_path,
        checkpoint_path
    )
    finetuned_pcc = evaluate_model(finetuned_model, processor, test_dataset)
    del finetuned_model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Base Model PCC:        {base_pcc:.4f}")
    print(f"Pretrained LoRA PCC:   {pretrained_pcc:.4f}")
    print(f"Fine-tuned LoRA PCC:   {finetuned_pcc:.4f}")
    print(f"\nImprovement:")
    print(f"  Pretrained → Fine-tuned: {finetuned_pcc - pretrained_pcc:+.4f}")
    print(f"  Base → Fine-tuned:       {finetuned_pcc - base_pcc:+.4f}")


if __name__ == "__main__":
    main()
```

---

## Key Differences Summary

| Aspect | Base Model | Pretrained LoRA | Fine-tuned LoRA |
|--------|-----------|-----------------|-----------------|
| **Loading Method** | `AutoModelForCausalLM.from_pretrained()` | Same (LoRA built-in) | `PeftModel.from_pretrained()` |
| **PEFT Import** | ❌ Not needed | ❌ Not needed | ✅ Required |
| **LoRA Adapters** | None | Built-in (r=320) | From checkpoint (r=64) |
| **Dtype** | float16/bfloat16 | bfloat16 | float16 (match training) |
| **Use Case** | Baseline | Starting point | Final evaluation |
| **Expected PCC** | ~0.3-0.4 | ~0.5-0.6 | ~0.65-0.73 (paper) |

---

## Common Issues and Solutions

### Issue 1: "KeyError: 'adapter_config.json'"

**Cause**: Checkpoint path points to wrong directory

**Solution**:
```python
# ❌ Wrong
checkpoint_path = "src/output/paper_r64"  # Directory without adapter files

# ✅ Correct
checkpoint_path = "src/output/paper_r64/checkpoint-120"  # Has adapter_config.json
```

### Issue 2: "RuntimeError: Expected all tensors on same device"

**Cause**: Model and inputs on different devices

**Solution**:
```python
# Ensure inputs moved to model device
inputs = processor(...).to(model.device)
```

### Issue 3: "LoRA adapters not found in checkpoint"

**Cause**: Training didn't save adapters properly

**Solution**:
```python
# Check checkpoint directory contents
import os
checkpoint_dir = "src/output/paper_r64/checkpoint-120"
print(os.listdir(checkpoint_dir))
# Should see: adapter_config.json, adapter_model.safetensors
```

### Issue 4: Different results between training and evaluation

**Cause**: Different dtype or evaluation settings

**Solution**:
```python
# Match training configuration exactly
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch.float16,  # MUST match training dtype
)

# Use deterministic generation for reproducibility
generated_ids = model.generate(
    ...,
    do_sample=False,      # Deterministic
    temperature=0.1,      # Low temperature
)
```

---

## Best Practices

### 1. Memory Management

```python
# Clear GPU memory between model loads
del model
torch.cuda.empty_cache()
```

### 2. Deterministic Evaluation

```python
# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Use deterministic generation
model.generate(..., do_sample=False)
```

### 3. Batch Evaluation (Faster)

```python
# Process multiple samples in one batch
batch_size = 8
for i in range(0, len(test_dataset), batch_size):
    batch = test_dataset[i:i+batch_size]
    # Process batch...
```

### 4. Error Handling

```python
try:
    pred_json = json.loads(json_str)
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print(f"Raw output: {json_str}")
    continue  # Skip this sample
```

---

## Quick Reference

### Load Base Model
```python
model, processor = load_base_model("microsoft/Phi-4-multimodal-instruct")
```

### Load Fine-tuned Model
```python
model, processor = load_finetuned_lora(
    base_model_path="microsoft/Phi-4-multimodal-instruct",
    checkpoint_path="src/output/paper_r64/checkpoint-120"
)
```

### Evaluate
```python
model.eval()
pcc = evaluate_model(model, processor, test_dataset)
```

---

**Summary**:
- Base model: Direct `AutoModelForCausalLM.from_pretrained()`
- Fine-tuned: `PeftModel.from_pretrained(base_model, checkpoint_path)`
- Key: Match dtype and ensure `model.eval()` before evaluation
