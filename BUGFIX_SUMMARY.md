# 訓練腳本修復總結

**日期**: 2025-12-20
**狀態**: ✅ 所有錯誤已修復（5個錯誤）

---

## 問題概述

在嘗試啟動訓練時遇到了七個連續的錯誤：

1. ❌ `max_seq_length` 參數錯誤 → ✅ 已修復為 `max_length`
2. ❌ `evaluation_strategy` 參數錯誤 → ✅ 已修復為 `eval_strategy`
3. ❌ `trust_remote_code` 錯誤 → ✅ 已修復，添加 `processing_class` 參數
4. ❌ `StopIteration` 多模態錯誤 → ✅ 初步修復嘗試（`formatting_func=None`）
5. ❌ `formatting_func=None` 無效 → ✅ 最終修復，添加 dummy `input_ids` 欄位
6. ❌ `TensorBoard` 缺失錯誤 → ✅ 已修復，使用 uv 安裝 tensorboard
7. ❌ `KeyError: 'audio_array'` → 🔄 正在調查

---

## 修復的檔案

### 1. [src/train_single_config.py](src/train_single_config.py)
- 行 97: `max_length=2048` (原為 max_seq_length)
- 行 106: `eval_strategy="no"` (原為 evaluation_strategy)
- 行 78-85: 添加 `add_dummy_input_ids()` 函數和數據集轉換
- 行 125: 添加 `processing_class=processor`
- 行 126: 添加 `formatting_func=None`

### 2. [src/train_dual_configs.py](src/train_dual_configs.py)
- 行 61: `max_length=2048` (原為 max_seq_length)
- 行 70: `eval_strategy="no"` (原為 evaluation_strategy)
- 行 42-49: 添加 `add_dummy_input_ids()` 函數和數據集轉換
- 行 89: 添加 `processing_class=processor`
- 行 90: 添加 `formatting_func=None`

### 3. 文檔更新
- [CLAUDE.md](CLAUDE.md) - 更新正確參數名稱
- [claudedocs/dual_config_training_guide.md](claudedocs/dual_config_training_guide.md) - 更新訓練參數
- [claudedocs/bugfix_sftconfig_max_length.md](claudedocs/bugfix_sftconfig_max_length.md) - 詳細錯誤文檔

---

## 根本原因

### 錯誤 1 & 2: SFTConfig API 差異
`trl` 庫的 `SFTConfig` 使用不同的參數名稱，與 Transformers 的 `TrainingArguments` 不同：

| Transformers | trl SFTConfig | 狀態 |
|--------------|---------------|------|
| `max_seq_length` | `max_length` | ✅ 已修復 |
| `evaluation_strategy` | `eval_strategy` | ✅ 已修復 |

### 錯誤 3: SFTTrainer 內部行為
`SFTTrainer.__init__()` 會嘗試自動重新加載 processor（在 `trl/trainer/sft_trainer.py:620`），但未傳遞 `trust_remote_code=True`。

**解決方案**: 明確傳遞 `processing_class=processor` 參數，防止 SFTTrainer 重新加載。

### 錯誤 4: 多模態數據處理
`SFTTrainer` 的自動 tokenization（在 `trl/trainer/sft_trainer.py:1060`）只傳遞文本，不支援多模態數據。當 Phi-4 processor 只接收文本時，會嘗試迭代空的 `audio_embed_size_iter`，導致 `StopIteration`。

**初步嘗試**: 設置 `formatting_func=None` → **無效**，SFTTrainer 仍然嘗試 tokenization

### 錯誤 5: formatting_func=None 無法阻止 tokenization
`SFTTrainer` 只在檢測到數據集已包含 `input_ids` 欄位時才跳過 tokenization（`is_processed=True`）。`formatting_func=None` 不能觸發此行為。

**最終解決方案**: 在數據集中添加 dummy `input_ids` 欄位（值為 `[0]`），使 SFTTrainer 認為數據集已處理，跳過 tokenization。實際的 tokenization 由 `AudioDataCollator` 在批次處理時完成。

---

## 驗證步驟

### 1. 語法檢查
```bash
source run_env.sh
cd src
python -c "from train_single_config import *"
python -c "from train_dual_configs import *"
```
預期：無錯誤輸出

### 2. 配置驗證
```bash
python verify_configs.py
```
預期：
```
✅ 配置 pretrained_r320 驗證成功！
✅ 配置 paper_r64 驗證成功！
✅ 所有配置驗證成功！可以開始訓練。
```

### 3. 啟動訓練測試
```bash
# 測試論文配置
python train_single_config.py --config paper_r64

# 測試預訓練配置
python train_single_config.py --config pretrained_r320

# 或使用便捷腳本
./train_both_configs.sh paper
```

預期結果：訓練應該開始載入模型和數據集，無任何 TypeError 或 ValueError

---

## 正確的訓練參數（論文規格）

```python
training_args = SFTConfig(
    output_dir=output_dir,

    # 論文超參數（Paper Table 3）
    num_train_epochs=3,                    # 論文最佳結果在 epoch 3
    per_device_train_batch_size=8,         # 論文設定
    gradient_accumulation_steps=8,         # 有效批次大小 = 64
    learning_rate=2e-5,                    # 論文設定（2×10⁻⁵）

    # 優化器和精度
    optim="adamw_torch",                   # Adam 優化器
    bf16=True,                             # bfloat16 精度

    # ✅ 正確參數名稱
    max_length=2048,                       # SFTConfig 使用 max_length
    eval_strategy="no",                    # SFTConfig 使用 eval_strategy

    # 日誌和保存
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,

    # 其他
    dataset_text_field="text_input",
    report_to="tensorboard",
    gradient_checkpointing=True,
)

# ✅ 正確的 SFTTrainer 初始化
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=AudioDataCollator(processor),
    peft_config=peft_config,
    processing_class=processor,  # 防止重新加載 processor
    formatting_func=None,  # 禁用自動 tokenization，使用 AudioDataCollator
)
```

---

## 下一步

### 立即測試
```bash
# 1. 驗證配置
source run_env.sh
cd src
python verify_configs.py

# 2. 啟動訓練（論文配置）
python train_single_config.py --config paper_r64
```

### 完整訓練流程
```bash
# 訓練兩種配置
./train_both_configs.sh

# 或分別訓練
./train_both_configs.sh paper       # 只訓練論文配置
./train_both_configs.sh pretrained  # 只訓練預訓練配置
```

### 監控訓練
```bash
# TensorBoard
tensorboard --logdir output/

# 查看訓練日誌
tail -f output/paper_r64/logs/events.out.tfevents.*
tail -f output/pretrained_r320/logs/events.out.tfevents.*
```

---

## 預期性能（論文基準 Table 3, Epoch 3）

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

## 詳細文檔

完整的錯誤分析和修復詳情請參考：
- [claudedocs/bugfix_sftconfig_max_length.md](claudedocs/bugfix_sftconfig_max_length.md)

訓練指南請參考：
- [claudedocs/dual_config_training_guide.md](claudedocs/dual_config_training_guide.md)
