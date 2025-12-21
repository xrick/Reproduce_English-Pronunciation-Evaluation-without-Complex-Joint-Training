# Remote NVIDIA GPU Training - Quick Start

## 🚀 快速開始（3 步驟）

### 1. 檢查環境
```bash
# 確認 CUDA 可用
nvidia-smi

# 確認 PyTorch CUDA 支持
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. 啟動訓練
```bash
# 激活虛擬環境（與本地相同）
source venv/bin/activate

# 論文配置 (r=64) - 單 GPU
python src/train_single_config_remote.py --config paper_r64 --gpus 0

# 預訓練配置 (r=320) - 單 GPU
python src/train_single_config_remote.py --config pretrained_r320 --gpus 0

# 多 GPU 訓練
python src/train_single_config_remote.py --config paper_r64 --gpus 0,1,2,3
```

### 3. 監控進度
```bash
# 實時監控 GPU
watch -n 1 nvidia-smi

# TensorBoard 可視化
tensorboard --logdir output/paper_r64/logs
```

## 📊 關鍵差異（vs Mac 本地訓練）

| 項目 | Mac 版本 | Remote 版本 |
|------|----------|-------------|
| 腳本文件 | `train_single_config.py` | `train_single_config_remote.py` |
| 設備 | MPS | CUDA |
| GPU 參數 | 無 | `--gpus 0,1,2,3` |
| Pin Memory | ❌ | ✅ |
| 多線程加載 | 單線程 | 4 workers |
| 多 GPU | ❌ | ✅ DDP/FSDP |

## ⚙️ 常用命令

### 基本訓練
```bash
# 論文配置（從零訓練 r=64）
python src/train_single_config_remote.py --config paper_r64 --gpus 0

# 預訓練配置（繼續訓練 r=320）
python src/train_single_config_remote.py --config pretrained_r320 --gpus 0
```

### 自定義參數
```bash
python src/train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0,1 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 3e-5
```

### FP16 模式（較舊 GPU）
```bash
# 如果 GPU 不支持 BF16（如 V100）
python src/train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

## 🔧 GPU 記憶體調整

| GPU 型號 | 內存 | 批次大小 | 梯度累積 |
|---------|------|---------|---------|
| RTX 3090 | 24GB | 4-8 | 8-16 |
| A100 (40GB) | 40GB | 8-16 | 4-8 |
| A100 (80GB) | 80GB | 16-32 | 2-4 |
| V100 | 32GB | 4-8 | 8-16 |

**記憶體不足時**:
```bash
# 減小批次大小，增加梯度累積
python src/train_single_config_remote.py --config paper_r64 --batch-size 4 --gradient-accumulation 16
```

## 📁 輸出目錄

```
output/
├── paper_r64/
│   ├── checkpoint-40/          # Epoch 1
│   ├── checkpoint-80/          # Epoch 2
│   ├── checkpoint-120/         # Epoch 3
│   ├── final_model/            # 最終模型
│   ├── logs/                   # TensorBoard
│   └── training_config_remote.json
```

## ⏱️ 預期訓練時間

基於 2500 樣本，3 epochs:

- **RTX 3090**: 8-10 小時
- **A100 (40GB)**: 4-6 小時
- **A100 (80GB)**: 2-3 小時
- **4×A100 (DDP)**: 1-2 小時

## 🐛 常見問題

### Q: CUDA out of memory
```bash
# 解決方案：減小批次大小
--batch-size 4 --gradient-accumulation 16
```

### Q: GPU 不支持 BF16
```bash
# 解決方案：使用 FP16（腳本自動檢測，或手動指定）
--fp16
```

### Q: 訓練中斷如何恢復？
```bash
# 從最後一個檢查點恢復
python src/train_single_config_remote.py --config paper_r64 --resume-from-checkpoint output/paper_r64/checkpoint-80
```

## 📚 詳細文檔

完整指南: [claudedocs/remote_training_guide.md](claudedocs/remote_training_guide.md)

包含:
- ✅ 平台差異詳解
- ✅ DeepSpeed/FSDP 配置
- ✅ 多節點訓練
- ✅ 性能優化建議
- ✅ 故障排除指南

## ✅ 遷移清單

從 Mac 遷移到 Remote:

- [ ] 虛擬環境已激活
- [ ] CUDA 可用 (`nvidia-smi`)
- [ ] 使用 `train_single_config_remote.py`
- [ ] 設置 GPU ID (`--gpus`)
- [ ] 確認 GPU 內存足夠
- [ ] 數據集路徑正確

完成後即可開始訓練！🎉
