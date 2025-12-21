#!/bin/bash
# Remote Training Launcher for NVIDIA TITAN RTX
# Ensures correct FP16 flags are set

set -e  # Exit on error

echo "================================================================================"
echo "NVIDIA TITAN RTX Training Launcher"
echo "================================================================================"
echo ""

# Check if venv is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating venv..."
    source venv/bin/activate
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""

# Check CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Get config choice
echo "Select configuration:"
echo "  1) paper_r64 (論文配置, r=64, 200M params) [RECOMMENDED]"
echo "  2) pretrained_r320 (預訓練配置, r=320, 830M params)"
echo ""
read -p "Enter choice (1 or 2, default=1): " CONFIG_CHOICE
CONFIG_CHOICE=${CONFIG_CHOICE:-1}

if [ "$CONFIG_CHOICE" == "1" ]; then
    CONFIG="paper_r64"
elif [ "$CONFIG_CHOICE" == "2" ]; then
    CONFIG="pretrained_r320"
else
    echo "❌ Invalid choice, using paper_r64"
    CONFIG="paper_r64"
fi

echo "Selected: $CONFIG"
echo ""

# Training parameters
BATCH_SIZE=8
GRAD_ACCUM=8
LEARNING_RATE=2e-5
EPOCHS=3

echo "Training parameters:"
echo "  Config: $CONFIG"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Precision: FP16 (TITAN RTX)"
echo ""

read -p "Start training? (y/n, default=y): " CONFIRM
CONFIRM=${CONFIRM:-y}

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "❌ Training cancelled"
    exit 0
fi

echo ""
echo "================================================================================"
echo "🚀 Starting Training..."
echo "================================================================================"
echo ""

# CRITICAL: Always use --fp16 for TITAN RTX
python src/train_single_config_remote.py \
  --config "$CONFIG" \
  --gpus 0 \
  --fp16 \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --epochs "$EPOCHS"

echo ""
echo "================================================================================"
echo "✅ Training Complete!"
echo "================================================================================"
