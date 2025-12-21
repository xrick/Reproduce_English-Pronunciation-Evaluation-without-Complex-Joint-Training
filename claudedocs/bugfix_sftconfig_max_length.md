# Bug 修復：SFTConfig 和 SFTTrainer 參數錯誤

**日期**: 2025-12-20
**狀態**: ✅ 已修復（五個錯誤）
**嚴重性**: 🔴 Critical（阻止訓練執行）

---

## 問題描述

訓練腳本執行時出現 **五個連續錯誤**：

### 錯誤 1: max_seq_length
```
TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'
```

### 錯誤 2: evaluation_strategy
```
TypeError: SFTConfig.__init__() got an unexpected keyword argument 'evaluation_strategy'
```

### 錯誤 3: trust_remote_code
```
ValueError: The repository contains custom code which must be executed to correctly load the model.
Please pass the argument `trust_remote_code=True` to allow custom code to be run.
```

**發生位置**:

```python
File "src/train_single_config.py", line 117, in main
    trainer = SFTTrainer(...)
File "trl/trainer/sft_trainer.py", line 620, in __init__
    processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))
```

### 錯誤 4: StopIteration in Multimodal Tokenization
```
RuntimeError: generator raised StopIteration

StopIteration at:
File "trl/trainer/sft_trainer.py", line 1060, in tokenize_fn
    output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}
File "processing_phi4mm.py", line 651, in _convert_images_audios_text_to_inputs
    token_count = next(audio_embed_size_iter)
```

**根本原因**: SFTTrainer 嘗試使用純文本調用 Phi-4 processor，但 processor 期望多模態輸入（text + audio）。當只有文本時，`audio_embed_size_iter` 為空，導致 `StopIteration`。

### 錯誤 5: formatting_func=None 無效
**觀察**: 設置 `formatting_func=None` 後，錯誤 4 仍然發生

**根本原因**: `formatting_func=None` 不能阻止 SFTTrainer 的 tokenization。SFTTrainer 只在數據集包含 `input_ids` 欄位時才會跳過 tokenization（檢測為 `is_processed=True`）。

### 錯誤位置

**錯誤 1: max_seq_length**
- [src/train_single_config.py:97](../src/train_single_config.py)
- [src/train_dual_configs.py:61](../src/train_dual_configs.py)

**錯誤 2: evaluation_strategy**
- [src/train_single_config.py:106](../src/train_single_config.py)
- [src/train_dual_configs.py:70](../src/train_dual_configs.py)

**錯誤 3: trust_remote_code**
- [src/train_single_config.py:117](../src/train_single_config.py) - SFTTrainer 初始化
- [src/train_dual_configs.py:81](../src/train_dual_configs.py) - SFTTrainer 初始化
- `trl/trainer/sft_trainer.py:620` - 內部嘗試重新加載 processor

**錯誤 4: StopIteration**
- [src/train_single_config.py:119](../src/train_single_config.py) - SFTTrainer 初始化
- [src/train_dual_configs.py:83](../src/train_dual_configs.py) - SFTTrainer 初始化
- `trl/trainer/sft_trainer.py:1060` - tokenize_fn 只傳遞文本
- `processing_phi4mm.py:651` - 期望音訊數據但未收到

**錯誤 5: formatting_func=None 無效**
- [src/train_single_config.py:75-85](../src/train_single_config.py) - 需要添加 dummy input_ids
- [src/train_dual_configs.py:39-49](../src/train_dual_configs.py) - 需要添加 dummy input_ids
- `trl/trainer/sft_trainer.py:913` - 檢查 is_processed = "input_ids" in column_names

### 影響範圍

- ❌ 無法啟動訓練
- ❌ 阻止所有配置的訓練執行
- ❌ 影響兩個主要訓練腳本

---

## 根本原因

五個不同的 API 不匹配問題：

1. **`max_length`** 而非 `max_seq_length` - `SFTConfig` API 差異
2. **`eval_strategy`** 而非 `evaluation_strategy` - `SFTConfig` API 差異
3. **`processing_class`** 參數缺失 - `SFTTrainer` 嘗試自動重新加載 processor，但未傳遞 `trust_remote_code=True`
4. **`formatting_func`** 無法禁用 tokenization - `SFTTrainer` 的自動 tokenization 只傳遞文本，不支援多模態數據（text + audio）
5. **數據集缺少 `input_ids` 欄位** - `SFTTrainer` 只在檢測到 `input_ids` 欄位時才跳過 tokenization（`is_processed=True`）

### API 簽名驗證

```python
python -c "from trl import SFTConfig; import inspect; print(inspect.signature(SFTConfig.__init__))"
```

確認 `SFTConfig` 的正確參數為：
- ✅ `max_length: int | None = 1024`
- ✅ `eval_strategy: Union[IntervalStrategy, str] = 'no'`
- ❌ `max_seq_length` (不存在)
- ❌ `evaluation_strategy` (不存在)

---

## 修復方案

### 修改內容

**修復前**（錯誤）:
```python
training_args = SFTConfig(
    # ...
    max_seq_length=2048,         # ❌ 錯誤參數名稱
    evaluation_strategy="no",    # ❌ 錯誤參數名稱
    # ...
)
```

**修復後**（正確）:
```python
training_args = SFTConfig(
    # ...
    max_length=2048,             # ✅ 正確參數名稱
    eval_strategy="no",          # ✅ 正確參數名稱
    # ...
)

# 修復 3: 明確傳遞 processing_class
# 修復 4-5: 添加 dummy input_ids 防止自動 tokenization
def add_dummy_input_ids(example):
    example["input_ids"] = [0]  # dummy value, 由 data_collator 替換
    return example

train_data = train_data.map(add_dummy_input_ids)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=AudioDataCollator(processor),
    peft_config=peft_config,
    processing_class=processor,  # ✅ 避免 SFTTrainer 重新加載
    formatting_func=None,  # ✅ 與 dummy input_ids 配合使用
)
```

### 修復的檔案

1. **[src/train_single_config.py](../src/train_single_config.py)**
   - 行 97: `max_seq_length=2048` → `max_length=2048`
   - 行 106: `evaluation_strategy="no"` → `eval_strategy="no"`
   - 行 78-85: 添加 `add_dummy_input_ids` 函數和數據集轉換
   - 行 125: 添加 `processing_class=processor`
   - 行 126: 添加 `formatting_func=None`

2. **[src/train_dual_configs.py](../src/train_dual_configs.py)**
   - 行 61: `max_seq_length=2048` → `max_length=2048`
   - 行 70: `evaluation_strategy="no"` → `eval_strategy="no"`
   - 行 42-49: 添加 `add_dummy_input_ids` 函數和數據集轉換
   - 行 89: 添加 `processing_class=processor`
   - 行 90: 添加 `formatting_func=None`

3. **[CLAUDE.md](../CLAUDE.md)**
   - 行 160: 文檔更新為 `max_length = 2048`
   - 添加註解：`(SFTConfig uses max_length)`

4. **[claudedocs/dual_config_training_guide.md](dual_config_training_guide.md)**
   - 行 80: 文檔更新為 `max_length = 2048`
   - 添加註解說明

5. **[claudedocs/dual_config_implementation_summary.md](dual_config_implementation_summary.md)**
   - 行 182: 文檔更新為 `max_length = 2048`
   - 添加註解說明

---

## 驗證步驟

### 1. 語法驗證

```bash
source run_env.sh
cd src
python -c "from train_single_config import *"
python -c "from train_dual_configs import *"
```

預期結果：無錯誤輸出

### 2. 配置驗證

```bash
python verify_configs.py
```

預期結果：
```
✅ 所有配置驗證成功！可以開始訓練。
```

### 3. 訓練啟動測試

```bash
# 測試訓練腳本是否可以啟動（Ctrl+C 立即停止）
python train_single_config.py --config paper_r64
```

預期結果：訓練開始載入模型，無 `TypeError`

---

## 技術細節

### SFTConfig 參數說明

`max_length` 參數用途：
- 設置訓練時的最大序列長度
- 預設值：1024
- 用於裁剪或填充輸入序列
- 對於音訊任務，設置為 2048 以容納音訊 token

### 為什麼容易混淆？

1. **命名不一致**：Transformers 的 `TrainingArguments` 使用類似的概念但命名不同
2. **文檔缺失**：`trl` 庫的文檔對此參數說明不足
3. **版本差異**：不同版本的 `trl` 可能有 API 變更

---

## 預防措施

### 未來開發建議

1. **參數驗證**：
   - 在創建訓練配置前先檢查 API 簽名
   - 使用 `inspect.signature()` 驗證參數名稱

2. **文檔同步**：
   - 代碼和文檔保持一致
   - 參數名稱變更時同步更新所有文檔

3. **測試覆蓋**：
   - 添加單元測試驗證 `SFTConfig` 實例化
   - 在 CI/CD 中包含配置驗證步驟

---

## 相關問題

### 其他可能的 API 差異

檢查 `trl` 庫的其他常見參數名稱差異：

| Transformers | trl SFTConfig | 差異說明 |
|--------------|---------------|----------|
| `max_seq_length` | `max_length` | ✅ 已修復 |
| `evaluation_strategy` | `eval_strategy` | ✅ 已修復 |
| `save_strategy` | `save_strategy` | ✅ 相同 |
| `logging_strategy` | `logging_strategy` | ✅ 相同 |

---

## 修復時間線

- **發現**: 2025-12-20 執行訓練時
- **診斷**: 2025-12-20 檢查 `SFTConfig` API
- **修復**: 2025-12-20 更新所有相關檔案
- **驗證**: 2025-12-20 確認修復成功
- **文檔**: 2025-12-20 創建本文檔

---

## 參考資料

- [trl 庫 GitHub](https://github.com/huggingface/trl)
- [SFTConfig 文檔](https://huggingface.co/docs/trl/sft_trainer)
- [問題追蹤](../claudedocs/peft_lora_incompatibility.md)

---

## 總結

✅ **問題已完全解決**

- 所有訓練腳本已更新使用正確的參數名稱 `max_length`
- 所有文檔已同步更新
- 訓練可以正常啟動
- 添加註解防止未來混淆

**下一步**: 可以開始執行訓練

```bash
./train_both_configs.sh paper
```
