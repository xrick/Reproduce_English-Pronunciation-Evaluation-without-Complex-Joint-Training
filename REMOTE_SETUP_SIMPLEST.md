# Remote Setup - Simplest Method (Use Online Model)

## 🎯 Problem
Model directory not found on remote machine at expected path.

## ✅ Simplest Solution (5 minutes)

### Option 1: Use Online Model (Recommended)

Edit **ONE** file to use HuggingFace online model:

#### Step 1: Edit model_utility_configs.py

```bash
# On remote machine
cd /path/to/project/src
nano model_utility_configs.py  # or vim, or any editor
```

#### Step 2: Find and Change Model Path

Look for this line (around line 14):
```python
model_path = "/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct"
```

Change to:
```python
model_path = "microsoft/phi-4-multimodal-instruct"
```

#### Step 3: Save and Test

```bash
# Save file (Ctrl+X then Y in nano)

# Test
python3 << 'EOF'
from model_utility_configs import CONFIGS
config = CONFIGS["paper_r64"]
model, processor, peft_config = config["loader"]()
print("✅ Success!")
EOF
```

#### Step 4: Start Training

```bash
python train_single_config_remote.py --config paper_r64 --gpus 0
```

**What happens**:
- ✅ First run: Downloads model from HuggingFace (one-time, ~15 mins)
- ✅ Cached at: `~/.cache/huggingface/hub/`
- ✅ Future runs: Uses cached model (fast)
- ✅ Auto-validates file integrity
- ✅ No manual file management

---

### Option 2: Find Your Actual Model Path

If you prefer to use local model:

#### Step 1: Find Model Location

```bash
# Method 1: Search by name
find ~ -type d -name "*phi*4*multimodal*" 2>/dev/null

# Method 2: Check common locations
ls -la ~/models/ 2>/dev/null
ls -la ~/.cache/huggingface/hub/ 2>/dev/null

# Method 3: Check where SFTTrainer.py expects
grep -r "model_path" src/
```

#### Step 2: Copy the Path You Found

For example, you might find:
```
/home/username/.cache/huggingface/hub/models--microsoft--phi-4-multimodal-instruct/snapshots/abc123/
```

#### Step 3: Update Config

Edit `src/model_utility_configs.py`:
```python
model_path = "/your/actual/path/here"
```

---

## 🔧 Quick Scripts

### Find Model Script

```bash
# Run this on remote machine to find model
chmod +x find_model_path.sh
./find_model_path.sh
```

### Interactive Fix Script

```bash
# Run this to fix tokenizer with custom path
chmod +x fix_remote_tokenizer_v2.sh
./fix_remote_tokenizer_v2.sh
```

---

## ⚡ Fastest Path Forward

**Just do this** (30 seconds):

```bash
# 1. SSH to remote
ssh user@remote
cd /path/to/project

# 2. Edit ONE line
sed -i 's|/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct|microsoft/phi-4-multimodal-instruct|g' src/model_utility_configs.py

# 3. Start training
source venv/bin/activate
python src/train_single_config_remote.py --config paper_r64 --gpus 0
```

Done! ✅

---

## 📊 Comparison

| Method | Setup Time | Download Time | Reliability |
|--------|-----------|---------------|-------------|
| **Online Model** | 30 sec | 15 min (first run only) | ⭐⭐⭐⭐⭐ |
| Find Local Path | 5 min | 0 (already exists) | ⭐⭐⭐ |
| Fix Tokenizer | 10 min | 2 min | ⭐⭐⭐⭐ |

**Recommendation**: Use **Online Model** method - it's the most reliable and requires minimal setup.

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'model_utility_configs'"

```bash
# Make sure you're in the right directory
cd /path/to/project/src
python train_single_config_remote.py ...
```

### "Network error" when downloading

```bash
# Check internet connection
ping huggingface.co

# Set proxy if needed
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### First download is slow

```bash
# Expected: 15-20 minutes for first download
# Model size: ~15GB

# Monitor progress in logs
# Look for: "Downloading..."
```

---

## ✅ Verification

After setup, verify everything works:

```bash
source venv/bin/activate
cd src

# Test 1: Import
python -c "from model_utility_configs import CONFIGS; print('✅ Import OK')"

# Test 2: Load processor
python -c "from model_utility_configs import CONFIGS; c=CONFIGS['paper_r64']; m,p,cfg=c['loader'](); print('✅ Model loaded')"

# Test 3: Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If all pass, start training!
python train_single_config_remote.py --config paper_r64 --gpus 0
```

---

**Time to train**: Setup (30 sec) + Download (15 min first time) + Training (2-6 hours)

**Total**: ~2-7 hours to results! 🚀
