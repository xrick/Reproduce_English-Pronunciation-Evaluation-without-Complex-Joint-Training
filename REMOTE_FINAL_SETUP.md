# Remote NVIDIA TITAN RTX - Final Setup Guide

## Current Status

✅ **Model Downloaded**: You have the Phi-4-multimodal-instruct model on remote machine
✅ **Scripts Ready**: All training scripts created and optimized for TITAN RTX
⏳ **Configuration**: Need to update model path to point to downloaded model

---

## GPU Specifications (NVIDIA TITAN RTX)

- **Architecture**: Turing (Compute Capability 7.5)
- **VRAM**: 24GB
- **CUDA Version**: 12.8
- **⚠️ CRITICAL**: **NO BF16 support** - must use FP16
- **Performance**: ~3-4 hours training time for paper_r64 config

---

## Quick Setup (3 Steps)

### Step 1: Update Model Path Configuration

First, transfer the update script to your remote machine:

```bash
# On Mac (in project directory)
scp update_model_path_remote.py user@remote:/path/to/project/

# Then SSH to remote
ssh user@remote
cd /path/to/project
```

Then run the update script on remote machine:

```bash
# Option 1: Auto-detect model location
python update_model_path_remote.py

# Option 2: Specify exact path where you downloaded model
python update_model_path_remote.py /your/actual/model/path/Phi-4-multimodal-instruct

# Option 3: Use online model (if you prefer HuggingFace auto-download)
python update_model_path_remote.py microsoft/phi-4-multimodal-instruct
```

**What this does**:
- ✅ Finds all `model_path =` assignments in `src/model_utility_configs.py`
- ✅ Updates both pretrained_r320 and paper_r64 configurations
- ✅ Creates backup before modifying
- ✅ Verifies model exists (for local paths)

### Step 2: Verify Model Loading

Test that the model loads correctly:

```bash
source venv/bin/activate
cd src

# Test model loading (should take ~30 seconds)
python -c "
from model_utility_configs import CONFIGS
print('Loading paper_r64 configuration...')
model, processor, peft_config = CONFIGS['paper_r64']['loader']()
print('✅ Model loaded successfully!')
print(f'Model device: {next(model.parameters()).device}')
print(f'Model dtype: {next(model.parameters()).dtype}')
"
```

**Expected output**:
```
✅ Patched PEFT to handle Phi-4's missing prepare_inputs_for_generation method
Loading paper_r64 configuration...

📊 【論文配置 r=64】參數統計:
  總參數: 5,563,727,360
  可訓練參數: 194,895,872 (3.5%)
  LoRA 層數: 288
  可訓練 LoRA 層: 288
  Speech LoRA: r=64, alpha=128, dp=0.05 (論文規格,從零訓練)
  Vision LoRA: r=256, alpha=512, dp=0.0 (保持預訓練)
  💾 訓練輸出目錄: output/paper_r64/
  ⚠️  注意: Speech LoRA 權重被重新初始化(形狀不匹配)

✅ Model loaded successfully!
Model device: cuda:0
Model dtype: torch.float16  # Should be FP16 on TITAN RTX
```

### Step 3: Start Training

```bash
source venv/bin/activate

# Paper configuration (r=64) with TITAN RTX optimizations
python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0 \
  --batch-size 8 \
  --gradient-accumulation 8 \
  --learning-rate 2e-5 \
  --epochs 3 \
  --fp16  # CRITICAL: TITAN RTX requires FP16, not BF16
```

**Training will automatically**:
- ✅ Detect TITAN RTX and use FP16 (not BF16)
- ✅ Enable CUDA optimizations (pin_memory, num_workers=4)
- ✅ Use gradient checkpointing to save VRAM
- ✅ Log to TensorBoard at `output/paper_r64/logs/`
- ✅ Save checkpoints every epoch
- ✅ Complete in ~3-4 hours

---

## Advanced Options

### Multi-GPU Training (if you have multiple GPUs)

```bash
# Check available GPUs
nvidia-smi

# Use multiple GPUs (e.g., GPU 0 and 1)
python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0,1 \
  --batch-size 8 \
  --gradient-accumulation 8 \
  --epochs 3 \
  --fp16
```

### Monitor Training Progress

```bash
# In a separate terminal, start TensorBoard
tensorboard --logdir src/output/paper_r64/logs/ --port 6007

# Access TensorBoard:
# http://remote-machine-ip:6007
```

### Custom Hyperparameters

```bash
# Larger batch size (if VRAM allows)
python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0 \
  --batch-size 16 \
  --gradient-accumulation 4 \
  --epochs 3 \
  --fp16

# Faster learning rate (experimental)
python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0 \
  --learning-rate 5e-5 \
  --epochs 4 \
  --fp16
```

---

## Troubleshooting

### Problem 1: "Model directory not found"

**Solution**: Run the update script again and verify path

```bash
python update_model_path_remote.py
# Then check the path it shows
ls -la /the/path/shown/config.json
```

### Problem 2: "CUDA out of memory"

**Solution**: Reduce batch size

```bash
python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0 \
  --batch-size 4 \
  --gradient-accumulation 16 \
  --fp16
```

### Problem 3: "tokenizer.json corrupted"

**Solution**: Re-download tokenizer files

```bash
cd /your/model/path
python3 << 'EOF'
from huggingface_hub import hf_hub_download

files = ["tokenizer.json", "tokenizer_config.json"]
for f in files:
    hf_hub_download(
        repo_id="microsoft/phi-4-multimodal-instruct",
        filename=f,
        local_dir=".",
        local_dir_use_symlinks=False,
        force_download=True
    )
print("✅ Tokenizer files re-downloaded")
EOF
```

### Problem 4: Training extremely slow

**Check**:
1. GPU utilization: `nvidia-smi -l 1` (should be ~95%+)
2. Batch size too small → increase if VRAM allows
3. num_workers: Should be 4 (check training script)

---

## Expected Training Metrics

Based on paper results (Table 3, LoRA-only, Epoch 3):

| Metric | Target | Your Result |
|--------|--------|-------------|
| Accuracy PCC | 0.656 | TBD |
| Fluency PCC | 0.727 | TBD |
| Prosodic PCC | 0.711 | TBD |
| Total PCC | 0.675 | TBD |
| WER | 0.140 | TBD |
| PER | 0.114 | TBD |
| F1-score | 0.724 | TBD |

**Note**: Epoch 3 shows best results in paper. Epoch 4 leads to overfitting.

---

## Configuration Comparison

### Paper r=64 Configuration (Remote Training)

- **Speech LoRA**: r=64, alpha=128, dropout=0.05 ⭐
- **Trainable Params**: ~200M (3.5%)
- **Training Start**: Random initialization (from scratch)
- **Precision**: FP16 (TITAN RTX requirement)
- **Expected Time**: 3-4 hours
- **Output**: `src/output/paper_r64/`

### Pretrained r=320 Configuration (Mac Local)

- **Speech LoRA**: r=320, alpha=640, dropout=0.01
- **Trainable Params**: 830M (14.9%)
- **Training Start**: Pretrained LoRA weights
- **Precision**: BF16 (Apple MPS)
- **Expected Time**: 4-5 hours
- **Output**: `src/output/pretrained_r320/`
- **Status**: Currently running on Mac

---

## Complete Training Command Reference

### Minimal (Recommended for First Run)

```bash
python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

### Full Options

```bash
python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0 \
  --batch-size 8 \
  --gradient-accumulation 8 \
  --learning-rate 2e-5 \
  --epochs 3 \
  --fp16 \
  --output-dir src/output/paper_r64 \
  --logging-steps 10 \
  --save-strategy epoch
```

### All Available Flags

```
--config              paper_r64 | pretrained_r320
--gpus                GPU IDs (e.g., 0 or 0,1)
--batch-size          Per-device batch size (default: 8)
--gradient-accumulation  Steps before weight update (default: 8)
--learning-rate       Learning rate (default: 2e-5)
--epochs              Training epochs (default: 3)
--fp16                Use FP16 precision (REQUIRED for TITAN RTX)
--bf16                Use BF16 precision (NOT supported on TITAN RTX)
--output-dir          Output directory (default: based on config)
--logging-steps       Log every N steps (default: 10)
--save-strategy       Save checkpoints: epoch | steps (default: epoch)
```

---

## Files Created

### Configuration Files
- `src/model_utility_configs.py` - Dual LoRA configuration (needs path update)
- `src/model_utility_configs.py.backup` - Backup (created by update script)

### Training Scripts
- `src/train_single_config_remote.py` - **Main remote training script** ⭐
- `src/train_single_config.py` - Mac local training (currently running)
- `src/AudioDataCollator.py` - Audio batch collation

### Utility Scripts
- `update_model_path_remote.py` - **Run this first on remote** ⭐
- `fix_remote_tokenizer_v2.sh` - Fix tokenizer corruption
- `find_model_path.sh` - Find model location

### Documentation
- `REMOTE_FINAL_SETUP.md` - **This file** ⭐
- `REMOTE_TRAINING_QUICKSTART.md` - Quick reference
- `REMOTE_ERROR_QUICKFIX.md` - Common error solutions
- `REMOTE_SETUP_SIMPLEST.md` - Alternative setup methods
- `ONE_LINE_FIX.txt` - One-line model path fix

---

## Summary: What To Do Right Now

1. **Transfer update script to remote**:
   ```bash
   scp update_model_path_remote.py user@remote:/path/to/project/
   ```

2. **SSH to remote and update config**:
   ```bash
   ssh user@remote
   cd /path/to/project
   python update_model_path_remote.py
   ```

3. **Start training**:
   ```bash
   source venv/bin/activate
   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
   ```

4. **Monitor progress** (optional, in separate terminal):
   ```bash
   tensorboard --logdir src/output/paper_r64/logs/ --port 6007
   ```

**Expected timeline**:
- Setup: 2-3 minutes
- Training: 3-4 hours
- **Total to results**: ~4 hours

---

## Next Steps After Training Completes

1. **Evaluate model**:
   ```bash
   python src/estimate.py --model-dir src/output/paper_r64/checkpoint-XXX
   ```

2. **Compare with Mac training**:
   - Remote: paper_r64 (r=64 from scratch, FP16)
   - Mac: pretrained_r320 (r=320 pretrained, BF16)

3. **Analyze results**:
   - Which configuration achieves better PCC/WER/PER?
   - Does paper_r64 match paper benchmarks?
   - Is pretrained_r320 faster to converge?

---

**Good luck with training! 🚀**

Report any issues and I'll help troubleshoot.
