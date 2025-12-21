# 雙配置訓練系統實作總結

**日期**: 2025-12-20
**狀態**: ✅ 完成實作，準備訓練

---

## 實作概述

成功實作了雙 LoRA 配置訓練系統，支援：

1. **預訓練配置** (r=320)：使用 Phi-4-multimodal 的預訓練 LoRA 權重
2. **論文配置** (r=64)：從零訓練，嚴格復現論文規格

兩種配置可以獨立訓練，模型保存到不同的輸出目錄，便於性能比較和分析。

---

## 核心檔案

### 1. 模型配置載入器

**檔案**: [src/model_utility_configs.py](../src/model_utility_configs.py)

**功能**:
- `get_model_and_processor_pretrained()`: 載入預訓練配置 (r=320)
- `get_model_and_processor_paper()`: 載入論文配置 (r=64)
- `CONFIGS`: 配置字典，映射配置名稱到載入函數和元數據
- `print_config_comparison()`: 打印配置對照表

**關鍵實作細節**:

```python
# 預訓練配置：直接載入預訓練 LoRA 權重
config.speech_lora = {'r': 320, 'lora_alpha': 640, 'dp': 0.01, ...}
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    # 不使用 ignore_mismatched_sizes
)

# 論文配置：允許形狀不匹配，重新初始化 Speech LoRA
config.speech_lora = {'r': 64, 'lora_alpha': 128, 'dp': 0.05, ...}
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True,  # 🔑 關鍵參數
)
```

**PEFT 補丁** (兩種配置共用):
```python
def _patched_peft_init(self, model, peft_config, adapter_name="default", **kwargs):
    if not hasattr(model, 'prepare_inputs_for_generation'):
        def prepare_inputs_for_generation(*args, **kwargs):
            return {}
        model.prepare_inputs_for_generation = prepare_inputs_for_generation
    _original_peft_init(self, model, peft_config, adapter_name, **kwargs)
```

### 2. 單一配置訓練腳本

**檔案**: [src/train_single_config.py](../src/train_single_config.py)

**功能**:
- 命令行介面訓練單一配置
- 支援自定義訓練超參數
- 自動保存訓練配置 JSON
- TensorBoard 日誌記錄

**使用方式**:
```bash
python train_single_config.py --config pretrained_r320
python train_single_config.py --config paper_r64

# 自定義參數
python train_single_config.py \
    --config paper_r64 \
    --epochs 4 \
    --batch-size 4 \
    --gradient-accumulation 16
```

### 3. 雙配置交互式訓練腳本

**檔案**: [src/train_dual_configs.py](../src/train_dual_configs.py)

**功能**:
- 交互式選擇訓練順序
- 支援訓練單一或兩種配置
- 顯示配置對照表
- 自動保存訓練配置和元數據

**訓練選項**:
1. 先訓練預訓練配置，再訓練論文配置
2. 先訓練論文配置，再訓練預訓練配置
3. 僅訓練預訓練配置
4. 僅訓練論文配置

### 4. 快速啟動 Shell 腳本

**檔案**: [train_both_configs.sh](../train_both_configs.sh)

**功能**:
- 一鍵啟動訓練
- 支援訓練單一或兩種配置
- 自動激活虛擬環境

**使用方式**:
```bash
./train_both_configs.sh              # 訓練兩種配置
./train_both_configs.sh pretrained   # 僅訓練預訓練配置
./train_both_configs.sh paper        # 僅訓練論文配置
```

### 5. 配置驗證腳本

**檔案**: [src/verify_configs.py](../src/verify_configs.py)

**功能**:
- 驗證兩種配置是否正確載入
- 檢查 LoRA 參數可訓練性
- 顯示參數統計
- 驗證配置設置

**使用方式**:
```bash
source run_env.sh
cd src
python verify_configs.py
```

---

## 配置規格

### 預訓練配置 (pretrained_r320)

| 參數 | 值 |
|------|-----|
| Speech LoRA rank | 320 |
| Speech LoRA alpha | 640 |
| Speech LoRA dropout | 0.01 |
| Vision LoRA rank | 256 |
| Vision LoRA alpha | 512 |
| Vision LoRA dropout | 0.0 |
| 可訓練參數 | 830M (14.9%) |
| 訓練起點 | 預訓練 LoRA 權重 |
| 輸出目錄 | output/pretrained_r320/ |

### 論文配置 (paper_r64)

| 參數 | 值 |
|------|-----|
| Speech LoRA rank | 64 ⭐ |
| Speech LoRA alpha | 128 ⭐ |
| Speech LoRA dropout | 0.05 ⭐ |
| Vision LoRA rank | 256 |
| Vision LoRA alpha | 512 |
| Vision LoRA dropout | 0.0 |
| 可訓練參數 | ~200M (3.5%) |
| 訓練起點 | 隨機初始化 ⭐ |
| 輸出目錄 | output/paper_r64/ |

⭐ = 論文原始規格

---

## 訓練超參數（論文規格）

基於論文 Table 3 的最佳配置：

```python
num_train_epochs = 3                    # 論文最佳結果在 epoch 3
per_device_train_batch_size = 8         # 論文設定
gradient_accumulation_steps = 8         # 有效批次大小 = 64
learning_rate = 2e-5                    # 論文設定 (2×10⁻⁵)
optimizer = "adamw_torch"               # Adam 優化器
bf16 = True                             # bfloat16 精度
max_length = 2048                       # 音訊 token 容量 (SFTConfig 使用 max_length)
gradient_checkpointing = True           # 內存優化
```

---

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
│   │   ├── processor_config.json
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
    │   ├── processor_config.json
    │   ├── tokenizer_config.json
    │   └── ...
    ├── logs/
    │   └── events.out.tfevents.*
    └── training_config.json
```

---

## 技術細節

### 關鍵差異：ignore_mismatched_sizes

**預訓練配置**:
- 模型載入時不使用 `ignore_mismatched_sizes`
- Speech LoRA 權重從 checkpoint 載入 (r=320)
- Vision LoRA 權重從 checkpoint 載入 (r=256)
- **結果**: 使用預訓練的 LoRA 權重

**論文配置**:
- 模型載入時使用 `ignore_mismatched_sizes=True`
- Speech LoRA 權重因形狀不匹配被重新初始化 (r=64)
- Vision LoRA 權重從 checkpoint 載入 (r=256，形狀相同)
- **結果**: Speech LoRA 從零訓練，Vision LoRA 使用預訓練權重

### PEFT 補丁

兩種配置都使用相同的 PEFT 補丁來解決 Phi-4-multimodal 的架構不兼容問題：

**問題**: `Phi4MMModel` 缺少 `prepare_inputs_for_generation` 方法
**解決方案**: 在 PEFT 初始化前動態添加該方法
**實作位置**: [src/model_utility_configs.py:14-30](../src/model_utility_configs.py)

### 精度選擇：bfloat16 無量化

**為什麼不使用量化？**
- 4-bit/8-bit 量化會將 LoRA 參數也量化為 uint8/int8
- 量化後的 LoRA 參數無法設置 `requires_grad=True`
- 導致 LoRA 層無法訓練

**解決方案**:
- 使用 `torch_dtype=torch.bfloat16`
- 不使用 `quantization_config`
- LoRA 參數保持為 bfloat16，可以正常訓練

**VRAM 影響**:
- 無量化 bfloat16: ~40-45GB
- 建議硬體: NVIDIA A100 (80GB) 或 A6000 (48GB)

---

## 文檔系統

### 用戶文檔

1. **[claudedocs/dual_config_training_guide.md](dual_config_training_guide.md)**
   - 完整的訓練指南
   - 配置對照表
   - 訓練參數說明
   - 常見問題解答

2. **[claudedocs/training_quick_reference.md](training_quick_reference.md)**
   - 快速參考卡
   - 一頁總結所有關鍵信息

3. **[CLAUDE.md](../CLAUDE.md)** (已更新)
   - 專案總覽
   - 新增雙配置系統說明
   - 快速啟動指南

### 技術文檔

1. **[claudedocs/peft_lora_incompatibility.md](peft_lora_incompatibility.md)**
   - PEFT/LoRA 不兼容問題詳細文檔
   - 嘗試的解決方案歷史
   - 最終解決方案說明

2. **[claudedocs/lora_from_scratch_config.md](lora_from_scratch_config.md)**
   - 論文規格 (r=64) 從零訓練指南
   - 三種實作選項比較
   - 推薦方案和實作代碼

3. **本文檔** (dual_config_implementation_summary.md)
   - 實作總結
   - 技術細節記錄
   - 設計決策說明

---

## 驗證清單

在開始訓練前，請確認：

- [ ] 虛擬環境已激活 (`source run_env.sh`)
- [ ] 訓練數據集已準備 (`../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/train/`)
- [ ] 測試數據集已準備 (`../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/test/`)
- [ ] 模型權重已下載 (`/Users/xrickliao/WorkSpaces/LLM_Repo/models/Phi-4-multimodal-instruct/`)
- [ ] 配置驗證通過 (`python src/verify_configs.py`)
- [ ] 足夠的 VRAM (≥40GB)
- [ ] 足夠的磁碟空間 (每個配置約 5-10GB)

---

## 預期訓練時間

基於 3 epochs 訓練：

- **預訓練配置** (r=320): 約 18-24 小時
  - 更多可訓練參數 (830M)
  - 但從預訓練權重開始，收斂較快

- **論文配置** (r=64): 約 12-18 小時
  - 較少可訓練參數 (~200M)
  - 但從零訓練，可能需要更多 epoch 才能達到最佳性能

**總計**（訓練兩種配置）: 約 30-42 小時

---

## 下一步

### 1. 驗證配置

```bash
source run_env.sh
cd src
python verify_configs.py
```

### 2. 開始訓練

```bash
# 方式 1: 一鍵訓練兩種配置
./train_both_configs.sh

# 方式 2: 分別訓練
./train_both_configs.sh paper      # 先訓練論文配置
./train_both_configs.sh pretrained # 再訓練預訓練配置
```

### 3. 監控訓練

```bash
# 查看訓練日誌
tensorboard --logdir output/

# 查看特定配置
tensorboard --logdir output/paper_r64/logs
```

### 4. 評估模型

```bash
# 評估論文配置
python estimate.py \
    --model-path ../output/paper_r64/final_model \
    --test-data ../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/test/

# 評估預訓練配置
python estimate.py \
    --model-path ../output/pretrained_r320/final_model \
    --test-data ../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/test/
```

### 5. 比較性能

比較兩種配置的：
- PCC (Accuracy, Fluency, Prosodic, Total)
- WER, PER, F1-score
- 訓練時間和收斂速度
- 與論文基準的差距

---

## 論文基準性能 (Paper Table 3)

LoRA-only 配置，Epoch 3：

| 指標 | 目標值 |
|------|--------|
| Accuracy PCC | 0.656 |
| Fluency PCC | 0.727 |
| Prosodic PCC | 0.711 |
| Total PCC | 0.675 |
| WER | 0.140 |
| PER | 0.114 |
| F1-score | 0.724 |

---

## 設計決策

### 為什麼需要兩種配置？

1. **科學驗證**: 論文配置用於嚴格復現論文結果
2. **性能探索**: 預訓練配置探索預訓練 LoRA 的性能上限
3. **比較分析**: 評估預訓練 LoRA 的實際價值
4. **靈活性**: 用戶可以根據需求選擇合適的配置

### 為什麼保留 Vision LoRA r=256？

論文主要關注 Speech LoRA（語音發音評估任務），Vision LoRA 影響較小：

- **保守策略**: 保留 Vision LoRA 預訓練值 (r=256)，降低風險
- **主要任務**: 語音評估為主，視覺模態為輔
- **穩定性**: 避免同時從零訓練兩個 LoRA，增加訓練穩定性

### 為什麼使用 bfloat16 而非量化？

- **LoRA 可訓練性**: 量化會導致 LoRA 參數無法訓練
- **精度要求**: 發音評估需要較高精度
- **硬體可用性**: A100/A6000 提供足夠的 VRAM

---

## 已知限制

1. **VRAM 需求高**: 需要 40-45GB VRAM，限制可用硬體
2. **訓練時間長**: 單個配置需要 12-24 小時
3. **數據集限制**: 僅在 SpeechOcean762 上測試
4. **缺少控制 token**: 當前實作尚未加入 `<|APA|>` 和 `<|MDD|>` 控制 token
5. **提示工程不完整**: 尚未加入論文的詳細評分標準 (133 行)
6. **標籤遮罩未實作**: 需要實作 prompt masking（僅訓練 assistant 回應部分）

---

## 未來改進

### 短期（訓練前必須完成）

1. **加入控制 token**: 實作 `<|APA|>` 和 `<|MDD|>` token
2. **詳細 prompt**: 加入論文附錄 7.1 的完整評分標準
3. **Prompt masking**: 實作標籤遮罩，只訓練 assistant 回應

### 中期（提升性能）

4. **完整評估指標**: 實作所有 PCC、WER、PER、F1 指標
5. **Checkpoint 恢復**: 支援從 checkpoint 繼續訓練
6. **學習率調度**: 實作 warmup 和 decay

### 長期（研究擴展）

7. **多數據集驗證**: 在其他發音評估數據集上測試
8. **混合精度優化**: 探索 FP8 或其他混合精度方案
9. **分佈式訓練**: 支援多 GPU 訓練，縮短訓練時間

---

## 貢獻者

- **實作者**: Claude (Anthropic)
- **指導**: xrickliao
- **日期**: 2025-12-20

---

## 參考資料

- [論文原文](../paper/)
- [專案 CLAUDE.md](../CLAUDE.md)
- [PEFT 不兼容問題文檔](peft_lora_incompatibility.md)
- [從零訓練配置指南](lora_from_scratch_config.md)
- [完整訓練指南](dual_config_training_guide.md)
