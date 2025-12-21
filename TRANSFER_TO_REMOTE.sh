#!/bin/bash
# Transfer fixed files from Mac to remote machine

# EDIT THIS: Your remote machine connection
REMOTE_HOST="user@remote-server"
REMOTE_PATH="/datas/store162/xrick/prjs/Reproduce_English_Pronunciation_Evaluation"

echo "=================================="
echo "Transferring fixed files to remote"
echo "=================================="
echo ""
echo "Remote: $REMOTE_HOST"
echo "Path:   $REMOTE_PATH"
echo ""

# Transfer fixed files
echo "📦 Transferring src/model_utility_configs.py..."
scp src/model_utility_configs.py $REMOTE_HOST:$REMOTE_PATH/src/

echo "📦 Transferring src/train_single_config_remote.py..."
scp src/train_single_config_remote.py $REMOTE_HOST:$REMOTE_PATH/src/

echo "📦 Transferring src/compat_trainer.py..."
scp src/compat_trainer.py $REMOTE_HOST:$REMOTE_PATH/src/

echo ""
echo "✅ All files transferred!"
echo ""
echo "=================================="
echo "Next: Run training on remote"
echo "=================================="
echo ""
echo "SSH to remote and run:"
echo ""
echo "  cd $REMOTE_PATH"
echo "  rm -rf src/output/paper_r64/"
echo "  source venv/bin/activate"
echo "  python src/train_single_config_remote.py --config paper_r64 --gpus 0"
echo ""
echo "Expected output:"
echo "  🔧 Applying LoRA configuration to model..."
echo "  ✅ LoRA configuration applied - parameters are now trainable"
echo "  可訓練參數: 200,000,000 (3.5%)  ← MUST BE ~200M"
echo "  loss: 6.98 → 6.42 → 5.89  ← MUST DECREASE"
echo ""
