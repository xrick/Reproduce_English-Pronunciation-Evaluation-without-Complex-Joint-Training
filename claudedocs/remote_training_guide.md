# Remote NVIDIA GPU Training Guide

完整的遠程 NVIDIA GPU 訓練指南

## 平台差異總結

| 特性 | Mac (Apple Silicon) | Remote (NVIDIA GPU) |
|------|---------------------|---------------------|
| 設備 | `mps` | `cuda` |
| Pin Memory | ❌ 不支持 | ✅ 支持 |
| 混合精度 | BF16 | BF16/FP16（視 GPU 架構） |
| 多 GPU | ❌ 單設備 | ✅ DDP/FSDP/DeepSpeed |
| 數據加載 | 單線程 | 多線程（4 workers） |
| 內存優化 | MPS 自動管理 | CUDA 緩存管理 |

## 文件對比

### 本地訓練（Mac）
```bash
python train_single_config.py --config paper_r64
```
**文件**: `src/train_single_config.py`

### 遠程訓練（NVIDIA）
```bash
python train_single_config_remote.py --config paper_r64
```
**文件**: `src/train_single_config_remote.py`

## 關鍵差異

### 1. 設備檢測和分配

**Mac 版本**:
```python
# 自動使用 MPS（Apple Metal Performance Shaders）
# 無需手動設置設備
```

**Remote 版本**:
```python
# 明確設置 CUDA 設備
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# 檢查 CUDA 可用性
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
```

### 2. 混合精度支持

**Mac 版本**:
```python
training_args = TrainingArguments(
    bf16=True,  # MPS 支持 BF16
)
```

**Remote 版本**:
```python
# 自動檢測 GPU 架構
compute_capability = torch.cuda.get_device_capability()
if compute_capability[0] < 8:  # Ampere 之前的架構
    use_fp16 = True  # 使用 FP16
else:
    use_bf16 = True  # Ampere+ 使用 BF16

training_args = TrainingArguments(
    bf16=use_bf16,
    fp16=use_fp16,
)
```

### 3. 數據加載優化

**Mac 版本**:
```python
training_args = TrainingArguments(
    # MPS 不支持 pin_memory
    # 數據加載器使用默認設置
)
```

**Remote 版本**:
```python
training_args = TrainingArguments(
    dataloader_pin_memory=True,     # CUDA 優化
    dataloader_num_workers=4,       # 多線程加載
)
```

### 4. 多 GPU 支持

**Remote 版本新增功能**:

#### 單 GPU
```bash
python train_single_config_remote.py --config paper_r64 --gpus 0
```

#### 多 GPU (DDP - Distributed Data Parallel)
```bash
python train_single_config_remote.py --config paper_r64 --gpus 0,1,2,3
```

#### FSDP (Fully Sharded Data Parallel)
```bash
python train_single_config_remote.py --config paper_r64 --gpus 0,1,2,3 --fsdp
```

#### DeepSpeed (內存效率最高)
```bash
python train_single_config_remote.py --config paper_r64 --gpus 0,1,2,3 --deepspeed ds_config.json
```

## 使用方式

### 基本訓練（單 GPU）

```bash
# 論文配置 (r=64)
python train_single_config_remote.py --config paper_r64 --gpus 0

# 預訓練配置 (r=320)
python train_single_config_remote.py --config pretrained_r320 --gpus 0
```

### 多 GPU 訓練

```bash
# 使用 4 個 GPU
python train_single_config_remote.py --config paper_r64 --gpus 0,1,2,3
```

### 自定義超參數

```bash
python train_single_config_remote.py \
  --config paper_r64 \
  --gpus 0,1 \
  --epochs 5 \
  --batch-size 16 \
  --gradient-accumulation 4 \
  --learning-rate 3e-5
```

### 使用 FP16（較舊 GPU）

```bash
# 如果 GPU 不支持 BF16（如 V100）
python train_single_config_remote.py --config paper_r64 --gpus 0 --fp16
```

## GPU 架構支持

### BF16 支持（推薦）
- ✅ A100 (Compute Capability 8.0)
- ✅ A6000 (Compute Capability 8.6)
- ✅ RTX 3090/4090 (Compute Capability 8.6/8.9)
- ✅ H100 (Compute Capability 9.0)

### FP16 支持（備選）
- ✅ V100 (Compute Capability 7.0)
- ✅ P100 (Compute Capability 6.0)
- ✅ 所有 NVIDIA GPU

**檢查方法**:
```python
import torch
print(torch.cuda.get_device_capability())
# (8, 0) = A100 → 支持 BF16
# (7, 0) = V100 → 僅支持 FP16
```

## DeepSpeed 配置（可選）

如果需要最大化內存效率，創建 `ds_config.json`:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  }
}
```

使用方式:
```bash
python train_single_config_remote.py --config paper_r64 --gpus 0,1,2,3 --deepspeed ds_config.json
```

## 性能優化建議

### 批次大小調整

根據 GPU 內存調整:

| GPU 型號 | 內存 | 建議批次大小 | 梯度累積 |
|---------|------|------------|---------|
| RTX 3090 | 24GB | 4-8 | 8-16 |
| A100 (40GB) | 40GB | 8-16 | 4-8 |
| A100 (80GB) | 80GB | 16-32 | 2-4 |
| V100 | 32GB | 4-8 | 8-16 |

**計算公式**:
```
有效批次大小 = 批次大小 × 梯度累積 × GPU 數量
論文設定 = 8 × 8 × 1 = 64
```

### 數據加載優化

```python
# 根據 CPU 核心數調整
dataloader_num_workers = min(4, os.cpu_count())
```

### 內存不足（OOM）解決方案

1. **減小批次大小**:
```bash
python train_single_config_remote.py --config paper_r64 --batch-size 4 --gradient-accumulation 16
```

2. **啟用梯度檢查點**（已默認啟用）:
```python
gradient_checkpointing=True
```

3. **使用 DeepSpeed ZeRO**:
```bash
python train_single_config_remote.py --config paper_r64 --deepspeed ds_config.json
```

## 監控和調試

### 實時監控

```bash
# 終端 1: 運行訓練
python train_single_config_remote.py --config paper_r64 --gpus 0

# 終端 2: 監控 GPU
watch -n 1 nvidia-smi

# 終端 3: TensorBoard
tensorboard --logdir output/paper_r64/logs
```

### 常見警告和解決方案

#### 警告: "pin_memory not supported on MPS"
- **Mac 本地訓練**: 忽略（正常現象）
- **Remote NVIDIA**: 不應出現（已啟用 pin_memory）

#### 錯誤: "CUDA out of memory"
```bash
# 解決方案 1: 減小批次大小
--batch-size 4 --gradient-accumulation 16

# 解決方案 2: 使用 DeepSpeed
--deepspeed ds_config.json
```

#### 警告: "GPU 不支持 BF16"
```bash
# 自動切換到 FP16（腳本已處理）
# 或手動指定:
--fp16
```

## 輸出結構

訓練完成後的文件結構:

```
output/
├── paper_r64/
│   ├── checkpoint-40/          # Epoch 1 檢查點
│   ├── checkpoint-80/          # Epoch 2 檢查點
│   ├── checkpoint-120/         # Epoch 3 檢查點
│   ├── final_model/            # 最終模型
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── ...
│   ├── logs/                   # TensorBoard 日誌
│   │   └── events.out.tfevents.*
│   └── training_config_remote.json  # 訓練配置記錄
```

## 遷移清單

從 Mac 遷移到 Remote NVIDIA GPU:

- [x] ✅ 使用 `train_single_config_remote.py` 而非 `train_single_config.py`
- [x] ✅ 檢查 CUDA 可用性: `nvidia-smi`
- [x] ✅ 設置正確的 GPU ID: `--gpus 0` 或 `--gpus 0,1,2,3`
- [x] ✅ 確認 GPU 架構（BF16 vs FP16）
- [x] ✅ 根據 GPU 內存調整批次大小
- [x] ✅ 虛擬環境相同（已確認）
- [x] ✅ 數據集路徑正確
- [x] ✅ 輸出目錄可寫

## 預期訓練時間

基於論文設定（2500 樣本，3 epochs）:

| GPU 型號 | 批次大小 | 預估時間 |
|---------|---------|---------|
| RTX 3090 | 8 | 8-10 小時 |
| A100 (40GB) | 8 | 4-6 小時 |
| A100 (80GB) | 16 | 2-3 小時 |
| 4×A100 (DDP) | 8 | 1-2 小時 |

## 故障排除

### 1. 導入錯誤
```python
ModuleNotFoundError: No module named 'torch'
```
**解決**: 確認虛擬環境已激活
```bash
source venv/bin/activate
pip list | grep torch
```

### 2. CUDA 不可用
```python
torch.cuda.is_available() = False
```
**檢查**:
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

### 3. 多 GPU 訓練失敗
```bash
# 使用 torchrun（推薦）
torchrun --nproc_per_node=4 train_single_config_remote.py --config paper_r64
```

## 進階用法

### 斷點續訓

```bash
# 訓練會自動在每個 epoch 保存檢查點
# 從檢查點恢復:
python train_single_config_remote.py --config paper_r64 --resume-from-checkpoint output/paper_r64/checkpoint-80
```

### 混合使用多台機器

使用 `torchrun` 的多節點訓練（需要 SSH 配置）:

```bash
# 主節點（Rank 0）
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 train_single_config_remote.py --config paper_r64

# 從節點（Rank 1）
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=29500 train_single_config_remote.py --config paper_r64
```

## 總結

**關鍵優勢**:
1. ✅ 自動檢測 GPU 架構和混合精度支持
2. ✅ 多 GPU 並行訓練（DDP/FSDP/DeepSpeed）
3. ✅ 優化的 CUDA 數據加載（pin_memory + 多線程）
4. ✅ 靈活的批次大小和內存管理
5. ✅ 完整的訓練監控和日誌記錄

**推薦工作流**:
```bash
# 1. 檢查環境
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 2. 開始訓練
python train_single_config_remote.py --config paper_r64 --gpus 0

# 3. 監控進度
tensorboard --logdir output/paper_r64/logs
```

祝訓練順利！🚀
