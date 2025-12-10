# Implementation Fixes Required

**Priority System**:
- 🔴 **P0 - CRITICAL**: Must fix before training (training will fail without these)
- 🟡 **P1 - IMPORTANT**: Should fix for paper-accurate results
- 🟢 **P2 - RECOMMENDED**: Nice-to-have improvements

---

## 🔴 P0: Critical Fixes (Must Fix Before Training)

### Fix 1: Learning Rate Correction
**Priority**: 🔴 P0
**File**: [src/SFTTrainer.py:8](../src/SFTTrainer.py#L8)
**Effort**: 5 minutes
**Impact**: Training will diverge without this fix

#### Current Code (WRONG)
```python
training_args = SFTConfig(
    # ... other params ...
    learning_rate=2e-4,  # ❌ WRONG: 10x too high
    # ... rest ...
)
```

#### Fixed Code
```python
training_args = SFTConfig(
    # ... other params ...
    learning_rate=2e-5,  # ✅ CORRECT: matches paper (2×10⁻⁵)
    # ... rest ...
)
```

#### Why Critical
- Paper specifies 2×10⁻⁵ (Section 4.2, Page 4)
- Current value is 10x too high
- Will cause unstable training, oscillating loss, potential NaN

---

### Fix 2: Implement Control Tokens
**Priority**: 🔴 P0
**File**: [src/data_transform.py:36-40](../src/data_transform.py#L36-L40)
**Effort**: 1-2 hours
**Impact**: Model cannot differentiate APA vs MDD tasks without this

#### Current Code (WRONG)
```python
def format_sample(self, sample):
    # ... extraction code ...

    # Generic prompt without control tokens
    user_prompt = "<|user|><|audio_1|>Analyze the pronunciation of the audio. Provide accuracy, fluency, prosodic, and total scores, along with word and phoneme transcripts in JSON format.<|end|>"
    assistant_response = f"<|assistant|>{json.dumps(target_output)}<|end|>"

    return {
        "audio_path": sample['audio']['path'],
        "audio_array": sample['audio']['array'],
        "sampling_rate": sample['audio']['sampling_rate'],
        "text_input": user_prompt + assistant_response,
        "prompt_only": user_prompt
    }
```

#### Fixed Code (Paper Section 3.1)
```python
# Define separate prompt templates with control tokens
APA_PROMPT_TEMPLATE = """<|user|><|APA|><|audio_1|>Rate the pronunciation of the audio.

**Accuracy**
Score range: 0 - 10
* 9-10: The overall pronunciation of the sentence is excellent, with accurate phonology and no obvious pronunciation mistakes
* 7-8: The overall pronunciation of the sentence is good, with a few pronunciation mistakes
* 5-6: The overall pronunciation of the sentence is understandable, with many pronunciation mistakes and accent, but it does not affect the understanding of basic meanings
* 3-4: Poor, clumsy and rigid pronunciation of the sentence as a whole, with serious pronunciation mistakes
* 0-2: Extremely poor pronunciation and only one or two words are recognizable

**Fluency**
Score range: 0 - 10
* 8-10: Fluent without noticeable pauses or stammering
* 6-7: Fluent in general, with a few pauses, repetition, and stammering
* 4-5: The speech is a little disfluent, with many pauses, repetition, and stammering
* 0-3: Intermittent, very disfluent speech, with lots of pauses, repetition, and stammering

**Prosodic**
Score range: 0 - 10
* 9-10: Correct intonation at a stable speaking speed, speak with cadence, and can speak like a native
* 7-8: Nearly correct intonation at a stable speaking speed, nearly smooth and coherent, but with little stammering and few pauses
* 5-6: Unstable speech speed, many stammering and pauses with a poor sense of rhythm
* 3-4: Unstable speech speed, speak too fast or too slow, without the sense of rhythm
* 0-2: Poor intonation and lots of stammering and pauses, unable to read a complete sentence

**Total**
Score range: 0 - 10
Provide an overall assessment of the pronunciation quality considering all aspects of the speech.
* 9-10: Excellent overall pronunciation that sounds nearly native-like
* 7-8: Good pronunciation with minor issues that don't affect comprehension
* 5-6: Fair pronunciation with noticeable non-native features but generally understandable
* 3-4: Poor pronunciation that requires effort to understand
* 0-2: Very poor pronunciation that is largely incomprehensible

Provide the results in the following JSON format:
{{'accuracy': ACCURACY_SCORE, 'fluency': FLUENCY_SCORE, 'prosodic': PROSODIC_SCORE, 'total': TOTAL_SCORE}}
<|end|><|assistant|>"""

MDD_PROMPT_TEMPLATE = """<|user|><|MDD|><|audio_1|>Transcribe the audio utterance, providing both a word-level transcript and phoneme-level breakdown.

For the phoneme breakdown, use the CMU Pronouncing Dictionary format (e.g., AA, IH).
If a word or phoneme is unclear, mark it with '<unk>'.

Provide the results in the following JSON format:
{{'word_transcript': 'That's an interesting observation.', 'phoneme_transcript': 'DH EH S AX N IH N T AX R EH S T IH NG AA B Z AX R V EY IH SH AX N'}}
<|end|><|assistant|>"""

def format_apa_sample(self, sample):
    """Format sample for APA (pronunciation scoring) task"""
    scores = {
        "accuracy": sample['accuracy'],
        "fluency": sample['fluency'],
        "prosodic": sample['prosodic'],
        "total": sample['total']
    }

    target_output = json.dumps(scores)

    return {
        "audio_path": sample['audio']['path'],
        "audio_array": sample['audio']['array'],
        "sampling_rate": sample['audio']['sampling_rate'],
        "text_input": APA_PROMPT_TEMPLATE + target_output,
        "prompt_only": APA_PROMPT_TEMPLATE,
        "task_type": "APA"
    }

def format_mdd_sample(self, sample):
    """Format sample for MDD (transcription) task"""
    # Build phoneme transcript with word boundaries
    phonemes = []
    for word in sample['words']:
        phonemes.extend(word['phones'])

    target_output = {
        "word_transcript": sample['text'],
        "phoneme_transcript": " ".join(phonemes)
    }

    target_json = json.dumps(target_output)

    return {
        "audio_path": sample['audio']['path'],
        "audio_array": sample['audio']['array'],
        "sampling_rate": sample['audio']['sampling_rate'],
        "text_input": MDD_PROMPT_TEMPLATE + target_json,
        "prompt_only": MDD_PROMPT_TEMPLATE,
        "task_type": "MDD"
    }

def format_sample(self, sample):
    """Format sample for both APA and MDD tasks"""
    # In practice, you would create separate training samples for each task
    # or alternate between them during training
    # For now, return APA format (modify based on your training strategy)
    return self.format_apa_sample(sample)
```

#### Why Critical (Paper Section 3.1)
- Control tokens (`<|APA|>` and `<|MDD|>`) enable the model to differentiate tasks
- Without them, model receives mixed signals and cannot learn task-specific behaviors
- This is the **core innovation** of the unified training approach

---

### Fix 3: Implement Prompt Masking
**Priority**: 🔴 P0
**File**: [src/data_collator.py:22-29](../src/data_collator.py#L22-L29)
**Effort**: 30 minutes
**Impact**: Model currently trains on prompts instead of just answers

#### Current Code (WRONG)
```python
def __call__(self, features):
    # ... existing code ...

    # Build labels (targets for loss calculation)
    batch["labels"] = batch["input_ids"].clone()

    # Mask padding tokens
    if self.processor.tokenizer.pad_token_id is not None:
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100

    return batch
```

#### Fixed Code
```python
def __call__(self, features):
    # ... existing code ...

    # Build labels (targets for loss calculation)
    batch["labels"] = batch["input_ids"].clone()

    # CRITICAL FIX: Mask prompt tokens (only train on assistant response)
    assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|assistant|>")

    for i, label_seq in enumerate(batch["labels"]):
        # Find the position of <|assistant|> token
        assistant_positions = (label_seq == assistant_token_id).nonzero(as_tuple=True)[0]

        if len(assistant_positions) > 0:
            assistant_pos = assistant_positions[0].item()
            # Mask everything BEFORE and INCLUDING the assistant token
            # Only train on the JSON response after it
            batch["labels"][i, :assistant_pos+1] = -100

    # Mask padding tokens
    if self.processor.tokenizer.pad_token_id is not None:
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100

    return batch
```

#### Why Critical
- **Current behavior**: Model trains on entire sequence (prompt + answer)
- **Problem**: Model learns to predict prompt tokens, not just the answer
- **Fix**: Mask prompt with -100 → PyTorch ignores these tokens in loss calculation
- **Result**: Model only learns to generate JSON responses, not prompts

---

### Fix 4: Adjust Batch Configuration
**Priority**: 🔴 P0
**File**: [src/SFTTrainer.py:6-7](../src/SFTTrainer.py#L6-L7)
**Effort**: 5 minutes
**Impact**: Training dynamics differ from paper

#### Current Code
```python
training_args = SFTConfig(
    # ... other params ...
    per_device_train_batch_size=4,  # ❌ Should be 8
    gradient_accumulation_steps=16,  # ❌ Should be 8
    # ... rest ...
)
```

#### Fixed Code (Paper Section 4.2)
```python
training_args = SFTConfig(
    # ... other params ...
    per_device_train_batch_size=8,  # ✅ Matches paper
    gradient_accumulation_steps=8,  # ✅ Matches paper (effective batch = 64)
    # ... rest ...
)
```

#### Why Important
- Paper uses batch_size=8, gradient_accumulation=8
- Effective batch size = 8 × 8 = 64 (same as paper)
- Different batch sizes can affect convergence speed and final performance

---

## 🟡 P1: Important Fixes (Should Fix for Validation)

### Fix 5: Comprehensive Evaluation Metrics
**Priority**: 🟡 P1
**File**: [src/estimate.py:34-39](../src/estimate.py#L34-L39)
**Effort**: 3-4 hours
**Impact**: Cannot validate results against paper without this

#### Current Code (INCOMPLETE)
```python
def evaluate_model(model, test_dataset):
    model.eval()
    predictions = []
    references = []

    # ... generation loop ...

    # Only calculates PCC for accuracy
    pred_acc = [p.get('accuracy', 0) for p in predictions]
    ref_acc = [r['accuracy'] for r in references]
    pcc = pearsonr(pred_acc, ref_acc)

    print(f"Accuracy PCC: {pcc}")
```

#### Fixed Code (Paper Table 3)
```python
from scipy.stats import pearsonr
import jiwer
import numpy as np

def evaluate_model(model, processor, test_dataset):
    """Comprehensive evaluation matching Paper Table 3"""
    model.eval()

    results = {
        # APA metrics
        'accuracy_pred': [], 'accuracy_ref': [],
        'fluency_pred': [], 'fluency_ref': [],
        'prosodic_pred': [], 'prosodic_ref': [],
        'total_pred': [], 'total_ref': [],
        # MDD metrics
        'word_hyp': [], 'word_ref': [],
        'phone_hyp': [], 'phone_ref': [],
    }

    for sample in test_dataset:
        # Prepare input
        inputs = processor(
            text=f"<|user|><|audio_1|>Analyze the pronunciation...<|end|><|assistant|>",
            audios=[sample['audio']['array']],
            return_tensors="pt",
            sampling_rate=16000
        ).to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Parse JSON (robust extraction)
        try:
            # Extract text between <|assistant|> and <|end|>
            response = generated_text.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
            pred_json = json.loads(response)

            # Store APA predictions
            for metric in ['accuracy', 'fluency', 'prosodic', 'total']:
                results[f'{metric}_pred'].append(pred_json.get(metric, 0))
                results[f'{metric}_ref'].append(sample.get(metric, 0))

            # Store MDD predictions
            results['word_hyp'].append(pred_json.get('word_transcript', ''))
            results['word_ref'].append(sample['text'])
            results['phone_hyp'].append(pred_json.get('phoneme_transcript', ''))
            results['phone_ref'].append(sample['phoneme_transcript'])

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Generated: {response[:200]}")
            continue

    # Calculate all metrics (matching Paper Table 3)
    metrics = {}

    # APA: Pearson Correlation Coefficients
    for metric in ['accuracy', 'fluency', 'prosodic', 'total']:
        pcc, p_value = pearsonr(
            results[f'{metric}_pred'],
            results[f'{metric}_ref']
        )
        metrics[f'{metric}_pcc'] = pcc
        metrics[f'{metric}_pcc_pvalue'] = p_value

        # Also calculate MAE
        mae = np.mean(np.abs(
            np.array(results[f'{metric}_pred']) -
            np.array(results[f'{metric}_ref'])
        ))
        metrics[f'{metric}_mae'] = mae

    # MDD: Word Error Rate (WER)
    metrics['wer'] = jiwer.wer(results['word_ref'], results['word_hyp'])

    # MDD: Phoneme Error Rate (PER)
    metrics['per'] = jiwer.wer(
        [' '.join(ref.split()) for ref in results['phone_ref']],
        [' '.join(hyp.split()) for hyp in results['phone_hyp']]
    )

    # MDD: F1-score, Precision, Recall for mispronunciation detection
    # (Requires phoneme-level alignment - simplified version here)
    tp, fp, fn = 0, 0, 0
    for ref, hyp in zip(results['phone_ref'], results['phone_hyp']):
        ref_phones = set(ref.split())
        hyp_phones = set(hyp.split())
        tp += len(ref_phones & hyp_phones)
        fp += len(hyp_phones - ref_phones)
        fn += len(ref_phones - hyp_phones)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1_score

    # Print results in Paper Table 3 format
    print("\\n=== Evaluation Results (Paper Table 3 Format) ===")
    print(f"Accuracy PCC: {metrics['accuracy_pcc']:.3f} (p={metrics['accuracy_pcc_pvalue']:.4f})")
    print(f"Fluency PCC:  {metrics['fluency_pcc']:.3f} (p={metrics['fluency_pcc_pvalue']:.4f})")
    print(f"Prosodic PCC: {metrics['prosodic_pcc']:.3f} (p={metrics['prosodic_pcc_pvalue']:.4f})")
    print(f"Total PCC:    {metrics['total_pcc']:.3f} (p={metrics['total_pcc_pvalue']:.4f})")
    print(f"\\nWER:         {metrics['wer']:.3f}")
    print(f"PER:          {metrics['per']:.3f}")
    print(f"F1-score:     {metrics['f1_score']:.3f}")
    print(f"Precision:    {metrics['precision']:.3f}")
    print(f"Recall:       {metrics['recall']:.3f}")

    # Compare with paper benchmarks
    print("\\n=== Comparison with Paper (LoRA, Epoch 3) ===")
    paper_benchmarks = {
        'accuracy_pcc': 0.656,
        'fluency_pcc': 0.727,
        'prosodic_pcc': 0.711,
        'total_pcc': 0.675,
        'wer': 0.140,
        'per': 0.114,
        'f1_score': 0.724
    }

    for metric, paper_value in paper_benchmarks.items():
        our_value = metrics[metric]
        diff = our_value - paper_value
        status = "✅" if abs(diff) < 0.05 else "⚠️"
        print(f"{status} {metric}: Ours={our_value:.3f}, Paper={paper_value:.3f}, Diff={diff:+.3f}")

    return metrics
```

#### Why Important
- Paper reports 7 key metrics (Table 3)
- Current implementation only evaluates 1 metric (accuracy PCC)
- Cannot validate reproduction success without comprehensive evaluation

---

### Fix 6: Reduce Training to 3 Epochs
**Priority**: 🟡 P1
**File**: [src/SFTTrainer.py:5](../src/SFTTrainer.py#L5)
**Effort**: 5 minutes
**Impact**: Paper's best results at epoch 3, not 4

#### Current Code
```python
training_args = SFTConfig(
    # ... other params ...
    num_train_epochs=4,  # ⚠️ Paper's best at epoch 3
    # ... rest ...
)
```

#### Fixed Code
```python
training_args = SFTConfig(
    # ... other params ...
    num_train_epochs=3,  # ✅ Paper's optimal stopping point
    # ... rest ...
)
```

#### Why Important (Paper Table 3)
- LoRA-only best performance at epoch 3:
  - Epoch 3: PER=0.114, F1=0.724
  - Epoch 4: PER=0.121, F1=0.721 (slightly worse)
- Training for 4 epochs leads to overfitting on MDD task

---

### Fix 7: Remove Audio Encoder Unfreezing Logic
**Priority**: 🟡 P1
**File**: [src/lora_config.py:47-62](../src/lora_config.py#L47-L62)
**Effort**: 5 minutes
**Impact**: Paper shows LoRA-only performs better

#### Current Code (PROBLEMATIC)
```python
model = get_peft_model(model, peft_config)

# Attempt to unfreeze audio encoder layers
for name, param in model.named_parameters():
    if "audio" in name and "lora" not in name:
        param.requires_grad = True
        # Note: This is problematic with 4-bit quantized params

model.print_trainable_parameters()
```

#### Fixed Code (Paper's LoRA-only Approach)
```python
model = get_peft_model(model, peft_config)

# Paper's LoRA-only approach: Keep audio encoder frozen
# Best results achieved without unfreezing (Table 3):
# - LoRA-only (epoch 3): PER=0.114, F1=0.724
# - Unfreeze (epoch 4): PER=0.142, F1=0.667
#
# Audio encoder and projector remain frozen as trained by Microsoft

model.print_trainable_parameters()
```

#### Why Important (Paper Finding)
- Paper experimentally compared two strategies:
  1. **LoRA-only**: Freeze audio encoder + projector
  2. **Unfreeze**: Train LoRA + audio encoder + projector
- Result: **LoRA-only achieved better MDD performance**
- Attempting to unfreeze 4-bit quantized parameters is problematic anyway

---

## 🟢 P2: Recommended Improvements (Nice to Have)

### Improvement 1: Add Learning Rate Scheduler
**Priority**: 🟢 P2
**File**: [src/SFTTrainer.py](../src/SFTTrainer.py)
**Effort**: 30 minutes

```python
training_args = SFTConfig(
    # ... existing params ...
    lr_scheduler_type="cosine",  # Smooth learning rate decay
    warmup_ratio=0.03,  # 3% of training for warmup
    # ... rest ...
)
```

**Benefit**: Smoother convergence, potentially better final performance

---

### Improvement 2: Add Experiment Tracking
**Priority**: 🟢 P2
**File**: [src/SFTTrainer.py](../src/SFTTrainer.py)
**Effort**: 1 hour

```python
training_args = SFTConfig(
    # ... existing params ...
    report_to="wandb",  # or "tensorboard"
    logging_steps=5,
    logging_first_step=True,
    # ... rest ...
)
```

**Benefit**: Monitor training dynamics, debug issues faster

---

### Improvement 3: Add Validation Split
**Priority**: 🟢 P2
**File**: [src/data_transform.py](../src/data_transform.py), [src/SFTTrainer.py](../src/SFTTrainer.py)
**Effort**: 2 hours

```python
# Split dataset
train_dataset = processed_dataset.select(range(0, 2250))  # 90%
val_dataset = processed_dataset.select(range(2250, 2500))  # 10%

# Update training args
training_args = SFTConfig(
    # ... existing params ...
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # ... rest ...
)
```

**Benefit**: Detect overfitting, enable early stopping

**Note**: Paper does not use validation split, so this is optional

---

## Summary of All Fixes

### Time Estimates

| Priority | Fixes | Total Effort | Impact |
|----------|-------|--------------|--------|
| 🔴 P0 (Critical) | 4 fixes | 2-3 hours | Training will fail without these |
| 🟡 P1 (Important) | 3 fixes | 4-5 hours | Cannot validate results without these |
| 🟢 P2 (Recommended) | 3 improvements | 3-4 hours | Incremental improvements |
| **Total** | **10 items** | **9-12 hours** | **Paper-accurate reproduction** |

### Implementation Order

**Day 1** (2-3 hours):
1. Fix learning rate (5 min)
2. Fix batch configuration (5 min)
3. Remove audio encoder unfreezing (5 min)
4. Reduce epochs to 3 (5 min)
5. Implement prompt masking (30 min)
6. Implement control tokens (1-2 hours)

**Day 2** (4-5 hours):
7. Implement comprehensive evaluation (3-4 hours)
8. Test evaluation on small subset (1 hour)

**Day 3** (Optional improvements):
9. Add learning rate scheduler (30 min)
10. Add experiment tracking (1 hour)
11. Add validation split (2 hours)

**Day 4-6** (Training):
- Run full training (8-12 GPU hours)
- Monitor and validate results
- Compare against paper benchmarks

---

## Testing Strategy

### After Each Fix

1. **Syntax Check**:
   ```bash
   python -m py_compile src/data_transform.py
   python -m py_compile src/data_collator.py
   python -m py_compile src/SFTTrainer.py
   python -m py_compile src/estimate.py
   ```

2. **Small-Scale Test** (before full training):
   ```python
   # Test on 10 samples
   test_dataset = processed_dataset.select(range(10))

   # Test data collation
   batch = data_collator([test_dataset[0], test_dataset[1]])

   # Test forward pass
   outputs = model(**batch)
   print(f"Loss: {outputs.loss}")
   ```

3. **Validation Checklist**:
   - ✅ Control tokens present in prompts
   - ✅ Prompt masking working (labels have -100 for prompt tokens)
   - ✅ Learning rate is 2e-5
   - ✅ Batch size is 8
   - ✅ Model loads without errors
   - ✅ Evaluation runs on small subset

### Full Training Validation

After training completes:
```python
# Compare your results with paper benchmarks
metrics = evaluate_model(model, processor, test_dataset)

# Check if within 5% of paper values
assert metrics['accuracy_pcc'] > 0.62  # Paper: 0.656
assert metrics['fluency_pcc'] > 0.69   # Paper: 0.727
assert metrics['per'] < 0.13           # Paper: 0.114
```

---

## Troubleshooting

### If Training Loss Doesn't Decrease

**Possible causes**:
1. Learning rate still too high (check it's 2e-5, not 2e-4)
2. Prompt masking not working (labels should have -100 for prompt)
3. Control tokens not in prompts (model receives mixed signals)

**Debug steps**:
```python
# Check learning rate
print(f"Learning rate: {trainer.args.learning_rate}")

# Check prompt structure
sample = processed_dataset[0]
print(f"Prompt: {sample['text_input'][:200]}")

# Check label masking
batch = data_collator([sample])
print(f"Labels with -100: {(batch['labels'] == -100).sum()}")
print(f"Labels > 0: {(batch['labels'] > 0).sum()}")
```

### If Evaluation Fails

**Possible causes**:
1. JSON parsing fails (model output format incorrect)
2. Missing keys in generated JSON
3. Type mismatches (string vs int)

**Debug steps**:
```python
# Print raw model output
print(f"Generated text:\\n{generated_text}")

# Try manual JSON parsing
response = generated_text.split("<|assistant|>")[-1].split("<|end|>")[0]
print(f"Extracted response:\\n{response}")

try:
    parsed = json.loads(response)
    print(f"Parsed successfully: {parsed}")
except json.JSONDecodeError as e:
    print(f"Parse failed: {e}")
```
