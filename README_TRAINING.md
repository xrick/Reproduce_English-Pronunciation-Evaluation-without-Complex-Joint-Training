# LoRA Training Documentation Index

Complete guide to dual-configuration LoRA training for English Pronunciation Evaluation.

---

## 🚀 Quick Start

**Choose your path**:

### Mac Local Training (Running Now ✅)
Already started with pretrained_r320 configuration.
- **Status**: Step 10/120 (8.3% complete)
- **See**: Training already in progress, no action needed

### Remote NVIDIA GPU Training (Next Step ⏳)
**→ Start here**: [REMOTE_QUICK_START.txt](REMOTE_QUICK_START.txt) (30 seconds)
- **Then**: [REMOTE_FINAL_SETUP.md](REMOTE_FINAL_SETUP.md) (complete guide)

---

## 📚 Documentation Structure

### Essential Guides (Start Here)

1. **[SESSION_SUMMARY.md](SESSION_SUMMARY.md)** ⭐
   - Complete overview of what we accomplished
   - Current status of Mac and Remote training
   - What you need to do next
   - All errors resolved and fixes applied

2. **[REMOTE_QUICK_START.txt](REMOTE_QUICK_START.txt)** ⭐
   - 3-step quick reference for remote training
   - Copy-paste commands
   - Essential troubleshooting
   - **READ THIS FIRST for remote setup**

3. **[REMOTE_FINAL_SETUP.md](REMOTE_FINAL_SETUP.md)** ⭐
   - Complete remote training guide
   - GPU optimization details (TITAN RTX)
   - All command options explained
   - Expected metrics and timeline

### Problem Solving

4. **[REMOTE_ERROR_QUICKFIX.md](REMOTE_ERROR_QUICKFIX.md)** 🔧
   - Common error solutions
   - Tokenizer corruption fix
   - Model path issues
   - Network connectivity problems

5. **[ONE_LINE_FIX.txt](ONE_LINE_FIX.txt)** 🚑
   - Emergency one-liner for model path
   - Use if everything else fails

6. **[BUGFIX_SUMMARY.md](BUGFIX_SUMMARY.md)**
   - All bugs encountered and fixed
   - Mac training issues resolved
   - API compatibility fixes

### Alternative Setup Methods

7. **[REMOTE_SETUP_SIMPLEST.md](REMOTE_SETUP_SIMPLEST.md)**
   - Simplest setup options
   - Online model vs local model
   - Finding model location

8. **[REMOTE_TRAINING_QUICKSTART.md](REMOTE_TRAINING_QUICKSTART.md)**
   - Alternative setup workflow
   - Detailed explanations
   - Multiple approaches

### Technical Deep Dives (claudedocs/)

9. **[claudedocs/dual_config_training_guide.md](claudedocs/dual_config_training_guide.md)**
   - Dual configuration system explained
   - r=320 vs r=64 comparison
   - When to use which config

10. **[claudedocs/peft_lora_incompatibility.md](claudedocs/peft_lora_incompatibility.md)**
    - PEFT/LoRA compatibility issues
    - Phi-4 specific problems
    - Patch implementation

11. **[claudedocs/lora_from_scratch_config.md](claudedocs/lora_from_scratch_config.md)**
    - Paper r=64 specification
    - Training from scratch details

12. **[claudedocs/remote_training_guide.md](claudedocs/remote_training_guide.md)**
    - Complete remote training theory
    - GPU architecture differences
    - CUDA optimization details

13. **[claudedocs/training_quick_reference.md](claudedocs/training_quick_reference.md)**
    - Command reference
    - All options explained

### Historical Bug Fixes (claudedocs/)

14. **[claudedocs/bugfix_sftconfig_max_length.md](claudedocs/bugfix_sftconfig_max_length.md)**
    - SFTConfig API changes
    - max_seq_length → max_length

15. **[claudedocs/audiodecoder_compatibility_fix.md](claudedocs/audiodecoder_compatibility_fix.md)**
    - Audio format issues
    - Numpy array conversion

16. **[claudedocs/remote_error_tokenizer_fix.md](claudedocs/remote_error_tokenizer_fix.md)**
    - Tokenizer corruption details
    - Detailed fix procedure

---

## 🛠️ Scripts and Tools

### Essential Scripts

- **[update_model_path_remote.py](update_model_path_remote.py)** ⭐
  - **Run this first on remote machine**
  - Auto-detects model location
  - Updates configuration files
  - Creates backup before modifying

### Training Scripts

- **[src/train_single_config_remote.py](src/train_single_config_remote.py)** ⭐
  - CUDA-optimized remote training
  - Auto-detects GPU architecture (FP16/BF16)
  - Multi-GPU support
  - **Use this for NVIDIA GPU training**

- **[src/train_single_config.py](src/train_single_config.py)** ⭐
  - Mac local training
  - Currently running pretrained_r320
  - Apple MPS optimized

- **[src/train_dual_configs.py](src/train_dual_configs.py)**
  - Interactive dual training
  - Train both configs sequentially

- **[train_both_configs.sh](train_both_configs.sh)**
  - Bash wrapper for dual training

### Utility Scripts

- **[fix_remote_tokenizer_v2.sh](fix_remote_tokenizer_v2.sh)**
  - Interactive tokenizer repair
  - Custom model path support

- **[find_model_path.sh](find_model_path.sh)**
  - Search for model location
  - Check common paths

### Configuration Files

- **[src/model_utility_configs.py](src/model_utility_configs.py)** ⭐
  - Dual LoRA configuration loader
  - pretrained_r320 and paper_r64
  - **Needs update on remote** (run update_model_path_remote.py)

- **[src/model_utility.py](src/model_utility.py)**
  - Original single-config loader
  - Deprecated (use model_utility_configs.py)

- **[src/AudioDataCollator.py](src/AudioDataCollator.py)** ⭐
  - Fixed audio batch collation
  - Numpy array + tuple format
  - Essential for multimodal training

---

## 📊 Configuration Comparison

### Pretrained r=320 (Mac - Running Now)

- **Speech LoRA**: r=320, alpha=640, dropout=0.01
- **Trainable**: 830M params (14.9%)
- **Start**: Pretrained LoRA weights
- **Precision**: BF16 (Apple MPS)
- **Time**: 4-5 hours
- **Output**: `src/output/pretrained_r320/`
- **Status**: ✅ Running (8.3% complete)

### Paper r=64 (Remote - Ready to Start)

- **Speech LoRA**: r=64, alpha=128, dropout=0.05 ⭐ (Paper spec)
- **Trainable**: 200M params (3.5%)
- **Start**: Random initialization (from scratch) ⭐
- **Precision**: FP16 (NVIDIA TITAN RTX)
- **Time**: 3-4 hours
- **Output**: `src/output/paper_r64/`
- **Status**: ⏳ Needs model path configuration

---

## 🎯 Current Status

### Mac Training ✅
```
Process:  33529
Config:   pretrained_r320
Progress: Step 10/120 (8.3%)
Loss:     6.9855
Elapsed:  ~26 minutes
Remaining: ~4 hours
```

### Remote Training ⏳
```
GPU:      NVIDIA TITAN RTX (24GB, Turing 7.5)
Config:   paper_r64
Model:    Downloaded (needs path configuration)
Next:     Run update_model_path_remote.py
Expected: 3-4 hours training time
```

---

## 🚦 What To Do Next

### On Mac (Already Running)
✅ Training in progress, no action needed
- Monitor: `ps aux | grep train_single_config`
- TensorBoard: `tensorboard --logdir src/output/pretrained_r320/logs/`

### On Remote (3 Steps)

1. **Transfer script** (on Mac):
   ```bash
   scp update_model_path_remote.py user@remote:/path/to/project/
   ```

2. **Update config** (on remote):
   ```bash
   ssh user@remote
   cd /path/to/project
   python update_model_path_remote.py
   ```

3. **Start training** (on remote):
   ```bash
   source venv/bin/activate
   python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
   ```

**See**: [REMOTE_QUICK_START.txt](REMOTE_QUICK_START.txt) for detailed commands

---

## 🎓 Expected Results

Based on paper Table 3 (LoRA-only, Epoch 3):

| Metric | Target |
|--------|--------|
| Accuracy PCC | 0.656 |
| Fluency PCC | 0.727 |
| Prosodic PCC | 0.711 |
| Total PCC | 0.675 |
| WER | 0.140 |
| PER | 0.114 |
| F1-score | 0.724 |

---

## ⚠️ Critical Notes

1. **TITAN RTX MUST use FP16** (no BF16 support)
2. **Mac uses BF16** (Apple MPS)
3. **Model path** must be updated on remote before training
4. **Epoch 3** shows best results (epoch 4 overfits)
5. **Backup config** created by update_model_path_remote.py

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | [REMOTE_ERROR_QUICKFIX.md](REMOTE_ERROR_QUICKFIX.md) |
| CUDA OOM | Use `--batch-size 4 --gradient-accumulation 16` |
| Tokenizer error | [fix_remote_tokenizer_v2.sh](fix_remote_tokenizer_v2.sh) |
| Mac path persisting | Run [update_model_path_remote.py](update_model_path_remote.py) |
| Emergency fix | [ONE_LINE_FIX.txt](ONE_LINE_FIX.txt) |

---

## 📁 File Organization

```
Project Root/
│
├── README_TRAINING.md          ⭐ This file - Documentation index
├── SESSION_SUMMARY.md          📊 Complete session overview
│
├── Remote Setup (Essential)
│   ├── REMOTE_QUICK_START.txt      ⭐ Start here (30 sec)
│   ├── REMOTE_FINAL_SETUP.md       📖 Complete guide
│   ├── REMOTE_ERROR_QUICKFIX.md    🔧 Troubleshooting
│   ├── ONE_LINE_FIX.txt            🚑 Emergency fix
│   ├── REMOTE_SETUP_SIMPLEST.md    💡 Alternative methods
│   └── REMOTE_TRAINING_QUICKSTART.md
│
├── Scripts (Run These)
│   ├── update_model_path_remote.py ⭐ Configure model path
│   ├── fix_remote_tokenizer_v2.sh  🔧 Fix tokenizer
│   ├── find_model_path.sh          🔍 Find model
│   └── train_both_configs.sh       🚀 Quick training
│
├── src/ (Training Code)
│   ├── train_single_config_remote.py ⭐ Remote training
│   ├── train_single_config.py       ⭐ Mac training
│   ├── model_utility_configs.py     ⭐ Dual configs
│   ├── AudioDataCollator.py         ✅ Fixed collator
│   └── output/
│       ├── pretrained_r320/         📊 Mac output
│       └── paper_r64/               📊 Remote output
│
└── claudedocs/ (Technical Details)
    ├── dual_config_training_guide.md
    ├── peft_lora_incompatibility.md
    ├── remote_training_guide.md
    └── ... (other technical docs)
```

---

## 🔗 Quick Links by Task

### I want to start remote training NOW
→ [REMOTE_QUICK_START.txt](REMOTE_QUICK_START.txt)

### I'm getting errors on remote
→ [REMOTE_ERROR_QUICKFIX.md](REMOTE_ERROR_QUICKFIX.md)

### I need complete understanding
→ [SESSION_SUMMARY.md](SESSION_SUMMARY.md)

### I want to understand configurations
→ [claudedocs/dual_config_training_guide.md](claudedocs/dual_config_training_guide.md)

### Emergency fix needed
→ [ONE_LINE_FIX.txt](ONE_LINE_FIX.txt)

### Monitor Mac training
→ `tensorboard --logdir src/output/pretrained_r320/logs/`

---

**Last Updated**: After resolving 10+ training errors and creating comprehensive remote setup

**Status**:
- Mac: ✅ Training (8.3% complete)
- Remote: ⏳ Ready (needs model path config)

**Next Action**: Run `update_model_path_remote.py` on remote machine

🚀 **Timeline to results**: ~4 hours (both trainings complete)
