# 雙配置訓練指南

## 概述

本專案現在支援兩種 LoRA 配置的訓練：

1. **預訓練配置** (r=320)：使用預訓練 LoRA 權重，收斂更快
2. **論文配置** (r=64)：從零訓練，嚴格復現論文規格

## 配置對照

| 項目 | 預訓練配置 (r=320) | 論文配置 (r=64) |
|------|-------------------|----------------|
| Speech LoRA rank | 320 | 64 ⭐ |
| Speech LoRA alpha | 640 | 128 ⭐ |
| Speech dropout | 0.01 | 0.05 ⭐ |
| Vision LoRA rank | 256 | 256 |
| Vision LoRA alpha | 512 | 512 |
| 可訓練參數 | 830M (14.9%) | ~200M (3.5%) |
| 訓練起點 | 預訓練 LoRA 權重 | 隨機初始化 ⭐ |
| 輸出目錄 | output/pretrained_r320/ | output/paper_r64/ |
| 預期收斂速度 | 較快 | 較慢（需更多 epoch） |
| 論文符合度 | 部分 | 完全符合 ⭐ |

⭐ = 論文原始規格

## 訓練方式

### 方式 1：交互式訓練（訓練兩種配置）

```bash
source run_env.sh
cd src
python train_dual_configs.py
```

**交互式選項**：
1. 先訓練預訓練配置，再訓練論文配置
2. 先訓練論文配置，再訓練預訓練配置
3. 僅訓練預訓練配置
4. 僅訓練論文配置

### 方式 2：命令行訓練（單一配置）

**訓練預訓練配置**：
```bash
source run_env.sh
cd src
python train_single_config.py --config pretrained_r320
```

**訓練論文配置**：
```bash
source run_env.sh
cd src
python train_single_config.py --config paper_r64
```

**自定義參數**：
```bash
python train_single_config.py \
    --config paper_r64 \
    --epochs 4 \
    --batch-size 4 \
    --gradient-accumulation 16 \
    --learning-rate 2e-5
```

## 訓練參數（論文規格）

根據論文 Table 3 的最佳配置：

```python
num_train_epochs = 3                    # 論文最佳結果在 epoch 3
per_device_train_batch_size = 8         # 論文設定
gradient_accumulation_steps = 8         # 有效批次大小 = 64
learning_rate = 2e-5                    # 論文設定（2×10⁻⁵）
optimizer = "adamw_torch"               # Adam 優化器
bf16 = True                             # bfloat16 精度
max_length = 2048                       # 容納音訊 token (SFTConfig 使用 max_length)
```

## 輸出結構

訓練後的目錄結構：

```
output/
├── pretrained_r320/
│   ├── checkpoint-epoch-1/
│   ├── checkpoint-epoch-2/
│   ├── checkpoint-epoch-3/
│   ├── final_model/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── ...
│   ├── logs/
│   │   └── events.out.tfevents.*
│   └── training_config.json
│
└── paper_r64/
    ├── checkpoint-epoch-1/
    ├── checkpoint-epoch-2/
    ├── checkpoint-epoch-3/
    ├── final_model/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer_config.json
    │   └── ...
    ├── logs/
    │   └── events.out.tfevents.*
    └── training_config.json
```

## 監控訓練

### TensorBoard

**查看預訓練配置的訓練過程**：
```bash
tensorboard --logdir output/pretrained_r320/logs
```

**查看論文配置的訓練過程**：
```bash
tensorboard --logdir output/paper_r64/logs
```

**同時查看兩種配置**：
```bash
tensorboard --logdir output/
```

然後在瀏覽器中打開 http://localhost:6006

### 訓練配置檔案

每個訓練運行都會生成 `training_config.json`，包含：
- 配置名稱和描述
- Speech LoRA 和 Vision LoRA 參數
- 訓練超參數
- 可訓練參數統計

示例：
```json
{
  "config_name": "paper_r64",
  "description": "論文 LoRA 配置（r=64），Speech LoRA 從零訓練",
  "speech_lora": {"r": 64, "alpha": 128, "dp": 0.05},
  "vision_lora": {"r": 256, "alpha": 512, "dp": 0.0},
  "trainable_params": "~200M (3.5%)",
  "training_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-05,
    "effective_batch_size": 64
  }
}
```

## 硬體需求

### VRAM 需求
- **bfloat16 無量化**：約 40-45GB
- **建議硬體**：
  - NVIDIA A100 (80GB)
  - NVIDIA A6000 (48GB)
  - 多張 RTX 4090 (24GB × 2)

### 訓練時間估計
- **預訓練配置 (r=320)**：約 6-8 小時 / epoch（預訓練 LoRA，收斂快）
- **論文配置 (r=64)**：約 4-6 小時 / epoch（參數較少，但從零訓練）
- **總時間（3 epochs）**：約 12-24 小時

## 評估訓練結果

訓練完成後，使用 `src/estimate.py` 評估模型性能：

```bash
# 評估預訓練配置
python estimate.py \
    --model-path ../output/pretrained_r320/final_model \
    --test-data ../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/test/

# 評估論文配置
python estimate.py \
    --model-path ../output/paper_r64/final_model \
    --test-data ../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/test/
```

### 論文基準性能（Paper Table 3, LoRA-only, Epoch 3）

| 指標 | 目標值 |
|------|--------|
| Accuracy PCC | 0.656 |
| Fluency PCC | 0.727 |
| Prosodic PCC | 0.711 |
| Total PCC | 0.675 |
| WER | 0.140 |
| PER | 0.114 |
| F1-score | 0.724 |

## 配置選擇建議

### 選擇預訓練配置 (r=320) 如果：
- ✅ 需要快速驗證訓練流程
- ✅ 想要更快的收斂速度
- ✅ 專案目標是實用性，不需要嚴格復現論文
- ✅ 有足夠的 VRAM（830M 可訓練參數）

### 選擇論文配置 (r=64) 如果：
- ✅ 需要嚴格復現論文結果
- ✅ 想要與論文基準進行公平比較
- ✅ 研究目標是驗證論文方法論
- ✅ 可以接受較慢的收斂速度和更多訓練時間

### 建議：訓練兩種配置
- 先訓練**論文配置**驗證論文復現能力
- 再訓練**預訓練配置**探索性能上限
- 比較兩種配置的性能差異，分析預訓練 LoRA 的價值

## 常見問題

### Q: 為什麼論文配置參數更少但訓練時間可能更長？
A: 論文配置的 LoRA 從零開始隨機初始化，需要更多訓練步驟才能收斂。預訓練配置從已經訓練好的 LoRA 權重開始，可以更快達到良好性能。

### Q: 兩種配置可以同時訓練嗎？
A: 不建議。每個配置需要約 40-45GB VRAM，同時訓練需要 80-90GB VRAM。建議依序訓練。

### Q: 如何選擇訓練順序？
A: 建議先訓練**論文配置** (r=64)，因為：
1. 驗證論文復現能力
2. 訓練時間較短（參數較少）
3. 可以作為基準與預訓練配置比較

### Q: 訓練失敗如何恢復？
A: 訓練會在每個 epoch 結束時保存 checkpoint，可以從最近的 checkpoint 繼續訓練。修改訓練腳本使用 `trainer.train(resume_from_checkpoint="output/xxx/checkpoint-epoch-N")` 恢復。

### Q: 如何調整訓練參數？
A: 使用 `train_single_config.py` 的命令行參數調整：
```bash
python train_single_config.py \
    --config paper_r64 \
    --epochs 4 \              # 增加訓練輪數
    --batch-size 4 \          # 減少批次大小（VRAM 不足時）
    --gradient-accumulation 16 \  # 增加梯度累積（保持有效批次大小）
    --learning-rate 1e-5      # 調整學習率
```

## 實作細節

### 關鍵檔案

1. **[src/model_utility_configs.py](../src/model_utility_configs.py)**
   - 兩種配置的模型載入函數
   - 配置對照表 `CONFIGS` 字典

2. **[src/train_dual_configs.py](../src/train_dual_configs.py)**
   - 交互式雙配置訓練腳本
   - 支援選擇訓練順序

3. **[src/train_single_config.py](../src/train_single_config.py)**
   - 命令行單一配置訓練腳本
   - 支援自定義訓練參數

4. **[src/AudioDataCollator.py](../src/AudioDataCollator.py)**
   - 音訊數據批次處理器
   - 處理填充和標籤遮罩

5. **[src/data_utility.py](../src/data_utility.py)**
   - SpeechOcean762 數據集格式化
   - 支援 TorchCodec AudioDecoder

### 核心差異

**預訓練配置載入**（model_utility_configs.py:78-98）：
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,  # r=320 配置
    torch_dtype=torch.bfloat16,
    # 不使用 ignore_mismatched_sizes
)
# LoRA 權重從 checkpoint 載入（r=320）
```

**論文配置載入**（model_utility_configs.py:165-174）：
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,  # r=64 配置
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True,  # 🔑 關鍵
)
# LoRA 權重被重新初始化（r=64）
```

### PEFT 補丁

兩種配置都使用相同的 PEFT 補丁（model_utility_configs.py:14-30）：
```python
def _patched_peft_init(self, model, peft_config, adapter_name="default", **kwargs):
    if not hasattr(model, 'prepare_inputs_for_generation'):
        def prepare_inputs_for_generation(*args, **kwargs):
            return {}
        model.prepare_inputs_for_generation = prepare_inputs_for_generation
    _original_peft_init(self, model, peft_config, adapter_name, **kwargs)
```

這解決了 Phi-4-multimodal 的架構不兼容問題。

## 參考資料

- [PEFT/LoRA 不兼容問題文檔](peft_lora_incompatibility.md)
- [從零訓練 LoRA 配置指南](lora_from_scratch_config.md)
- [論文原文](../paper/)
- [專案 CLAUDE.md](../CLAUDE.md)
