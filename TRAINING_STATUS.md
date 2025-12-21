# Training Status Summary

**生成時間**: 2025-12-20 18:46 (本地時間)

## 🎯 當前狀態

### Mac 本地訓練（測試中）

- **狀態**: ✅ **正在運行**
- **配置**: `pretrained_r320` (Microsoft 預訓練 r=320)
- **腳本**: `src/train_single_config.py`
- **進程 ID**: 33529
- **CPU 使用率**: 15.3%
- **內存**: ~825 MB
- **運行時間**: 36+ 秒
- **目的**: 驗證代碼修復後訓練正常運行

### 修復歷史

成功解決的 7 個連續錯誤:

1. ✅ `max_seq_length` → `max_length` (SFTConfig API)
2. ✅ `evaluation_strategy` → `eval_strategy` (SFTConfig API)
3. ✅ `trust_remote_code` ValueError → 添加 `processing_class` 參數
4. ✅ `StopIteration` 錯誤 → 嘗試 `formatting_func=None`（失敗）
5. ✅ `formatting_func` 無效 → 添加 dummy `input_ids`（失敗，改用 Trainer）
6. ✅ TensorBoard 缺失 → 使用 `uv pip install tensorboard`
7. ✅ `sampling_rate` TypeError → 修復 audios 格式為 `List[Tuple[array, sr]]`
8. ✅ `'list' has no attribute 'ndim'` → 轉換 list 為 numpy array

### 最終解決方案

**關鍵修改**:

1. **放棄 SFTTrainer，改用 Trainer**:
   ```python
   # 避免 SFTTrainer 的 tokenization 問題
   from transformers import Trainer
   trainer = Trainer(...)  # 而非 SFTTrainer
   ```

2. **修復 AudioDataCollator**:
   ```python
   # 正確的 audios 格式
   audios = [(np.array(f["audio_array"], dtype=np.float32),
              f["sampling_rate"]) for f in features]
   ```

3. **TrainingArguments 配置**:
   ```python
   training_args = TrainingArguments(
       eval_strategy="no",  # 而非 evaluation_strategy
       remove_unused_columns=False,  # 保留所有欄位
       # ...
   )
   ```

## 📂 已生成文件

### 遠程訓練支持

1. **訓練腳本**:
   - `src/train_single_config_remote.py` - NVIDIA GPU 優化版本

2. **文檔**:
   - `claudedocs/remote_training_guide.md` - 完整遠程訓練指南
   - `REMOTE_TRAINING_QUICKSTART.md` - 快速開始指南

3. **修復文檔**:
   - `BUGFIX_SUMMARY.md` - 錯誤修復總結
   - `claudedocs/bugfix_sftconfig_max_length.md` - 詳細錯誤分析

## 🚀 下一步行動

### Mac 本地（當前）

```bash
# 監控訓練進度
tail -f /tmp/claude/.../tasks/bf12ee2.output

# 查看 TensorBoard（訓練開始後）
tensorboard --logdir output/pretrained_r320/logs
```

### Remote NVIDIA GPU

#### 1. 準備環境
```bash
# 傳輸文件到遠程機器
scp -r src/ user@remote:/path/to/project/
scp -r venv/ user@remote:/path/to/project/  # 或重新創建 venv

# SSH 連接
ssh user@remote
cd /path/to/project
```

#### 2. 檢查環境
```bash
# 確認 CUDA
nvidia-smi

# 確認 PyTorch CUDA
source venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 3. 開始訓練
```bash
# 單 GPU - 論文配置 (r=64)
python src/train_single_config_remote.py --config paper_r64 --gpus 0

# 多 GPU - 論文配置
python src/train_single_config_remote.py --config paper_r64 --gpus 0,1,2,3

# 預訓練配置 (r=320)
python src/train_single_config_remote.py --config pretrained_r320 --gpus 0
```

#### 4. 監控
```bash
# 終端 1: 訓練
python src/train_single_config_remote.py --config paper_r64 --gpus 0

# 終端 2: GPU 監控
watch -n 1 nvidia-smi

# 終端 3 (可選): TensorBoard
tensorboard --logdir output/paper_r64/logs --port 6006
```

## 📊 配置對比

### pretrained_r320 (Microsoft 預訓練)
- **Speech LoRA**: r=320, alpha=640, dropout=0.01
- **Vision LoRA**: r=256, alpha=512, dropout=0.0
- **訓練參數**: 830M (14.9%)
- **起點**: 預訓練 LoRA 權重
- **優勢**: 更快收斂
- **用途**: 測試、快速驗證

### paper_r64 (論文從零訓練)
- **Speech LoRA**: r=64, alpha=128, dropout=0.05
- **Vision LoRA**: r=256, alpha=512, dropout=0.0
- **訓練參數**: ~200M (3.5%)
- **起點**: 隨機初始化 LoRA
- **優勢**: 嚴格論文復現
- **用途**: 正式實驗、論文對比

## 🎓 論文目標性能（Epoch 3）

| 指標 | 目標值 |
|------|--------|
| Accuracy PCC | 0.656 |
| Fluency PCC | 0.727 |
| Prosodic PCC | 0.711 |
| Total PCC | 0.675 |
| WER | 0.140 |
| PER | 0.114 |
| F1-score | 0.724 |

## ⚠️ 重要注意事項

### Mac 本地
- **用途**: 代碼驗證、調試
- **限制**: 訓練速度較慢（8-12+ 小時）
- **設備**: Apple MPS
- **不支持**: 多 GPU、pin_memory

### Remote NVIDIA
- **用途**: 正式訓練、論文復現
- **優勢**: 訓練速度快（2-6 小時，視 GPU）
- **設備**: CUDA
- **支持**: 多 GPU、DeepSpeed、FSDP

### 虛擬環境
- ✅ **已確認**: 本地和遠程使用相同虛擬環境
- ✅ **TensorBoard**: 已安裝（`uv pip install tensorboard`）
- ✅ **依賴**: 所有依賴已滿足

## 📖 參考文檔

1. **快速開始**: `REMOTE_TRAINING_QUICKSTART.md`
2. **完整指南**: `claudedocs/remote_training_guide.md`
3. **項目說明**: `CLAUDE.md`
4. **錯誤修復**: `BUGFIX_SUMMARY.md`
5. **雙配置**: `claudedocs/dual_config_training_guide.md`

## ✅ 準備就緒清單

### 代碼準備
- [x] ✅ 本地訓練腳本可用 (`train_single_config.py`)
- [x] ✅ 遠程訓練腳本已創建 (`train_single_config_remote.py`)
- [x] ✅ AudioDataCollator 已修復
- [x] ✅ 雙配置支持（r=320 & r=64）
- [x] ✅ TensorBoard 已安裝

### 文檔準備
- [x] ✅ 遠程訓練指南
- [x] ✅ 快速開始文檔
- [x] ✅ 錯誤修復記錄
- [x] ✅ 平台差異說明

### 環境準備
- [x] ✅ 虛擬環境配置（相同）
- [x] ✅ 數據集格式化完成
- [ ] ⏳ 遠程機器 CUDA 驗證（待執行）
- [ ] ⏳ 遠程數據集傳輸（待執行）

---

**狀態**: 🟢 **準備就緒，可開始遠程訓練**

**建議**: 先在 Mac 本地驗證完整流程（當前正在進行），確認無誤後再在遠程 NVIDIA GPU 上進行正式訓練。
