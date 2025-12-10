# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research reproduction project implementing the paper "English Pronunciation Evaluation without Complex Joint Training: LoRA Fine-tuned Speech Multimodal LLM" (2025). The system performs both Automatic Pronunciation Assessment (APA) and Mispronunciation Detection and Diagnosis (MDD) using Microsoft's Phi-4-multimodal-instruct model fine-tuned with LoRA.

**Key Capabilities**:
- **APA**: Scores pronunciation on accuracy, fluency, prosodic quality, and total quality (0-10 scale)
- **MDD**: Provides word-level and phoneme-level transcription using CMUDict format
- **Unified Training**: Both tasks trained simultaneously within a single model using control tokens

**Dataset**: SpeechOcean762 (2,500 training + 2,500 test samples of Mandarin L1 speakers reading English)

## Architecture

### High-Level Design

```
Audio Input → Phi-4-Multimodal-Instruct
              ├─ Audio Encoder (pre-trained, frozen/unfrozen)
              ├─ Audio Projector (frozen/unfrozen)
              ├─ LoRA Adapters (trainable)
              └─ LLM (Phi-4-mini, frozen except LoRA)

Training:     Control Tokens (<|APA|> or <|MDD|>) → Task-specific JSON outputs
Optimization: QLoRA (4-bit NF4 quantization) + Flash Attention 2
```

### Critical Components

**1. Control Token System** (Paper Section 3.1)
- `<|APA|>`: Triggers pronunciation scoring task
- `<|MDD|>`: Triggers transcription task
- **CRITICAL**: These tokens MUST be included in prompts to differentiate tasks

**2. Prompt Engineering** (Paper Appendix 7.1 & 7.2)
- APA prompts require detailed 0-10 scoring rubrics (133 lines)
- MDD prompts specify CMUDict phoneme format
- Prompts are the primary mechanism for teaching scoring criteria

**3. Training Strategy** (Paper Section 3.2)
- **LoRA-only**: Train only LoRA adapters (Paper's best approach)
- **Unfreeze**: Additionally train audio encoder + projector (Paper shows worse MDD performance)

## File Structure

```
src/
├── lora_config.py       # QLoRA config + model loading + LoRA setup
├── data_transform.py    # SpeechOcean762 → training format conversion
├── data_collator.py     # Batch collation with padding + label masking
├── SFTTrainer.py        # Training configuration (SFTConfig + SFTTrainer)
└── estimate.py          # Evaluation metrics (PCC, WER, PER, F1)

refdata/md_src/          # Reference documentation for each component
paper/                   # Original research paper (English + Chinese)
reproduceenv/            # Python 3.11.6 virtual environment
```

## Development Setup

### Environment Activation

```bash
# Activate virtual environment
source reproduceenv/bin/activate

# Verify Python version (should be 3.11.6)
python --version
```

### Key Dependencies

- **transformers**: Model loading and training
- **peft**: LoRA implementation
- **trl**: SFTTrainer for supervised fine-tuning
- **bitsandbytes**: 4-bit quantization (QLoRA)
- **datasets**: SpeechOcean762 dataset loading
- **scipy**: Pearson correlation coefficient (PCC)
- **jiwer**: WER calculation

## Training Configuration (Paper Section 4.2)

### CRITICAL: Paper-Accurate Hyperparameters

```python
# From paper Table 3 (best results at epoch 3)
batch_size = 8                      # per_device_train_batch_size
gradient_accumulation_steps = 8     # Effective batch = 64
learning_rate = 2e-5                # NOTE: 2×10⁻⁵, NOT 2×10⁻⁴
optimizer = "Adam"
num_train_epochs = 3                # Paper's best results at epoch 3
max_seq_length = 2048
bf16 = True
```

**WARNING**: The current implementation in `src/SFTTrainer.py` has INCORRECT hyperparameters:
- Learning rate is 10x too high (2e-4 instead of 2e-5)
- Batch size is 4 instead of 8
- Gradient accumulation is 16 instead of 8
- Trains for 4 epochs instead of optimal 3

### Expected Performance (Paper Table 3, LoRA-only, Epoch 3)

| Metric | Target Value |
|--------|--------------|
| Accuracy PCC | 0.656 |
| Fluency PCC | 0.727 |
| Prosodic PCC | 0.711 |
| Total PCC | 0.675 |
| WER | 0.140 |
| PER | 0.114 |
| F1-score | 0.724 |

## Known Implementation Gaps

### CRITICAL Issues (Must fix before training)

1. **Missing Control Tokens** ([src/data_transform.py](src/data_transform.py))
   - Current: Generic prompts without `<|APA|>` or `<|MDD|>` tokens
   - Required: Prepend control tokens to differentiate tasks
   - Impact: Model cannot learn task-specific behaviors

2. **Missing Detailed Scoring Rubrics** ([src/data_transform.py](src/data_transform.py))
   - Current: Simple generic prompt
   - Required: Full 133-line rubric from Paper Appendix 7.1
   - Impact: Model has no scoring criteria to learn from

3. **Prompt Masking Not Implemented** ([src/data_collator.py](src/data_collator.py))
   - Current: Loss calculated on entire sequence including user prompt
   - Required: Mask all tokens before `<|assistant|>` token with -100
   - Impact: Model trains to predict prompts instead of just answers

4. **Incorrect Hyperparameters** ([src/SFTTrainer.py](src/SFTTrainer.py))
   - Learning rate 10x too high will prevent convergence
   - See "Training Configuration" section above for corrections

### Important Issues

5. **Incomplete Evaluation** ([src/estimate.py](src/estimate.py))
   - Current: Only calculates PCC for accuracy score
   - Required: PCC for all 4 scores + WER + PER + F1/Precision/Recall
   - Impact: Cannot validate against paper's benchmarks

6. **Dataset Completeness Metric** ([src/data_transform.py](src/data_transform.py))
   - Current: Correctly excludes completeness (all values = 10)
   - Status: ✓ Matches paper specification

## Data Format

### Input Format (Training)

```python
# APA Task Sample
{
    "audio_array": np.array(...),
    "sampling_rate": 16000,
    "text_input": "<|user|><|APA|><|audio_1|>[Detailed scoring rubrics]<|end|><|assistant|>{'accuracy': 8, 'fluency': 7, 'prosodic': 8, 'total': 8}",
    "prompt_only": "<|user|><|APA|><|audio_1|>[Detailed scoring rubrics]<|end|><|assistant|>"
}

# MDD Task Sample
{
    "audio_array": np.array(...),
    "sampling_rate": 16000,
    "text_input": "<|user|><|MDD|><|audio_1|>[Transcription instructions]<|end|><|assistant|>{'word_transcript': 'The city was...', 'phoneme_transcript': 'DH AX...'}",
    "prompt_only": "<|user|><|MDD|><|audio_1|>[Transcription instructions]<|end|><|assistant|>"
}
```

### Output Format (Inference)

```json
{
  "accuracy": 8,
  "fluency": 7,
  "prosodic": 8,
  "total": 8,
  "word_transcript": "The city was...",
  "phoneme_transcript": "DH AX S IH T IY W AA Z..."
}
```

## Hardware Requirements

- **Minimum**: 1× NVIDIA A100 (80GB) or RTX 4090 (24GB)
- **VRAM Usage**: ~22GB with current QLoRA configuration
- **Training Time**: 8-12 hours for 3-4 epochs on SpeechOcean762

## Key Implementation Details

### QLoRA Configuration

```python
# 4-bit quantization for memory efficiency
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True       # Double quantization
)
```

### LoRA Parameters (Following "LoRA Without Regret")

```python
LoraConfig(
    r=64,                              # Rank
    lora_alpha=128,                    # Alpha (ratio 2:1 for stability)
    target_modules="all-linear",       # Apply to all linear layers
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]  # Preserve critical layers
)
```

### Label Masking Strategy

**Required Implementation** (currently missing):
```python
# In data collator, after creating labels
assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|assistant|>")
for i, label_seq in enumerate(batch["labels"]):
    assistant_pos = (label_seq == assistant_token_id).nonzero(as_tuple=True)[0]
    if len(assistant_pos) > 0:
        # Mask prompt tokens: only train on assistant response
        batch["labels"][i, :assistant_pos[0]+1] = -100
```

## Evaluation Metrics

### APA Evaluation
- **Metric**: Pearson Correlation Coefficient (PCC)
- **Scope**: accuracy, fluency, prosodic, total scores
- **Interpretation**: PCC > 0.7 indicates strong correlation with human scores

### MDD Evaluation
- **WER**: Word Error Rate (lower is better, target < 0.15)
- **PER**: Phoneme Error Rate (lower is better, target < 0.12)
- **F1-score**: Mispronunciation detection accuracy (higher is better, target > 0.72)

### Validation Strategy (Paper Insight)
- Paper shows correlation between PER and accuracy scores (r = -0.4463)
- Lower PER → Higher accuracy scores validates genuine pronunciation understanding

## Reference Documentation

The `refdata/md_src/` directory contains detailed explanations for each component:
- `lora_config.md`: Model loading and LoRA setup
- `data_transformer.md`: Dataset formatting logic
- `data_collator.md`: Batch collation and padding
- `SFTTrainer.md`: Training configuration
- `estimate.md`: Evaluation methodology

## Paper-Specific Insights

1. **LoRA-only outperforms Unfreezing** (Paper Table 3)
   - LoRA-only PER: 0.114 (epoch 3)
   - Unfreeze PER: 0.142 (epoch 4)
   - Recommendation: Use LoRA-only approach (freeze audio encoder + projector)

2. **Optimal Training Duration**
   - Best results achieved at epoch 3, not epoch 4
   - Further training leads to overfitting on MDD task

3. **Control Tokens Enable Unified Training**
   - Single model handles both APA and MDD without separate task heads
   - Eliminates need for complex joint training architectures

4. **Prompt Engineering > Architecture Changes**
   - Detailed scoring rubrics in prompts teach the model evaluation criteria
   - No need for separate datasets or model modifications per task

## Workflow for Reproducing Paper Results

1. **Fix Critical Issues**:
   - Implement control tokens in `data_transform.py`
   - Add detailed prompts from Paper Appendix 7.1 & 7.2
   - Implement prompt masking in `data_collator.py`
   - Correct hyperparameters in `SFTTrainer.py`

2. **Prepare Data**:
   ```python
   from datasets import load_dataset
   raw_dataset = load_dataset("mispeech/speechocean762", split="train")
   ```

3. **Train Model**:
   ```python
   trainer = SFTTrainer(model=model, args=training_args, ...)
   trainer.train()
   ```

4. **Evaluate**:
   - Calculate PCC for all 4 scores
   - Calculate WER, PER, F1 for transcription
   - Compare against Paper Table 3 benchmarks

5. **Expected Timeline**:
   - Code fixes: 4-6 hours
   - Training (3 epochs): 8-12 hours GPU time
   - Evaluation: 1-2 hours
   - Total: ~2-3 days for full reproduction
