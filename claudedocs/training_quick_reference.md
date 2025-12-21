# 訓練快速參考

## 一鍵訓練

```bash
# 訓練兩種配置（論文 r=64 + 預訓練 r=320）
./train_both_configs.sh

# 僅訓練論文配置 (r=64)
./train_both_configs.sh paper

# 僅訓練預訓練配置 (r=320)
./train_both_configs.sh pretrained
```

## 配置對比

| 配置 | LoRA Rank | 訓練參數 | 起始點 | 收斂速度 | 論文符合度 |
|------|-----------|----------|--------|----------|------------|
| **pretrained_r320** | r=320 | 830M (14.9%) | 預訓練權重 | 快 | 部分 |
| **paper_r64** | r=64 | ~200M (3.5%) | 隨機初始化 | 慢 | 完全 ⭐ |

## 輸出位置

```
output/
├── pretrained_r320/final_model/  # 預訓練配置模型
└── paper_r64/final_model/        # 論文配置模型
```

## 監控訓練

```bash
# 查看訓練日誌
tensorboard --logdir output/
```

## 訓練參數（論文規格）

- Epochs: 3
- Batch size: 8
- Gradient accumulation: 8 (有效批次 = 64)
- Learning rate: 2e-5
- Precision: bfloat16

## 預期性能（論文 Table 3）

| 指標 | 目標值 |
|------|--------|
| Accuracy PCC | 0.656 |
| Fluency PCC | 0.727 |
| Prosodic PCC | 0.711 |
| Total PCC | 0.675 |
| WER | 0.140 |
| PER | 0.114 |
| F1-score | 0.724 |

## 硬體需求

- **VRAM**: 40-45GB
- **訓練時間**: 12-24 小時 (3 epochs)
- **建議硬體**: NVIDIA A100 (80GB) 或 A6000 (48GB)

## 詳細指南

完整文檔請參閱 [dual_config_training_guide.md](dual_config_training_guide.md)
