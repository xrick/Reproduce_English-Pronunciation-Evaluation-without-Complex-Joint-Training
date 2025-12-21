# 使用論文原始 LoRA 規格從零訓練

## 配置方案

如果要使用論文原始規格（r=64, alpha=128, dropout=0.05），需要**不載入預訓練 LoRA 權重**，改為從零開始訓練。

## 實作方法

### 選項 A：修改模型載入邏輯（跳過 LoRA 權重）

```python
# 在 model_utility.py 中修改

# Speech LoRA: 使用論文規格
config.speech_lora = {
    'r': 64,              # 論文規格
    'lora_alpha': 128,    # 論文規格
    'layer': '((layers.*self_attn\\.(qkv|o)_proj)|(layers.*mlp\\.(gate_up|down)_proj))',
    'dp': 0.05            # 論文規格
}

config.vision_lora = {
    'r': 64,              # 論文規格（或保持 256）
    'lora_alpha': 128,    # 論文規格（或保持 512）
    'layer': 'layers.*((self_attn\\.(qkv_proj|o_proj))|(mlp\\.(gate_up|down)_proj))',
    'dp': 0.05            # 論文規格
}

# 載入模型時跳過 LoRA 權重
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    local_files_only=True,
    quantization_config=None,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True,  # 🔑 關鍵：忽略形狀不匹配
)
```

### 選項 B：不載入預訓練模型（完全從零）

```python
from transformers import AutoConfig, AutoModelForCausalLM

# 只載入配置，不載入權重
config = AutoConfig.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True
)

# 設定 LoRA 為論文規格
config.speech_lora = {'r': 64, 'lora_alpha': 128, 'dp': 0.05, ...}
config.vision_lora = {'r': 64, 'lora_alpha': 128, 'dp': 0.05, ...}

# 從配置建立新模型（隨機初始化）
model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
```

### 選項 C：載入基礎模型，重新初始化 LoRA

```python
# 1. 先載入不含 LoRA 的基礎模型
config_no_lora = AutoConfig.from_pretrained(model_path, ...)
# 暫時禁用 LoRA（設為最小值）
config_no_lora.speech_lora = {'r': 1, ...}  # 最小化 LoRA
config_no_lora.vision_lora = {'r': 1, ...}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config_no_lora,
    ...
)

# 2. 手動重新初始化為論文規格的 LoRA
# （需要修改模型內部結構，較複雜）
```

## 優缺點比較

| 方案 | 優點 | 缺點 | VRAM | 訓練時間 |
|------|------|------|------|---------|
| **A: ignore_mismatched_sizes** | 簡單，一行代碼 | LoRA 隨機初始化，失去預訓練優勢 | ~40GB | 長（從零訓練）|
| **B: from_config** | 完全控制，清晰明確 | 整個模型從零訓練（包括基礎層） | ~40GB | 最長 |
| **C: 重新初始化** | 保留基礎模型權重 | 實作複雜，需修改內部結構 | ~40GB | 長（LoRA 從零）|

## 推薦方案：選項 A

### 為什麼推薦選項 A？

1. **實作簡單**：只需加一個參數 `ignore_mismatched_sizes=True`
2. **保留基礎模型**：Phi-4 的主體權重（LLM、視覺、音訊編碼器）仍使用預訓練
3. **LoRA 從零訓練**：符合論文設定，公平比較

### 完整實作代碼

```python
def get_model_and_processor(
    model_id: str = "microsoft/Phi-4-multimodal-instruct",
    lora_rank: int = 64,        # 論文規格
    lora_alpha: int = 128       # 論文規格
):
    bnb_config = None  # bfloat16，無量化

    model_path = "/Users/xrickliao/WorkSpaces/LLM_Repo/models/Phi-4-multimodal-instruct"

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    config._attn_implementation = "eager"

    # 🔑 使用論文規格
    config.speech_lora = {
        'r': lora_rank,           # 64
        'lora_alpha': lora_alpha, # 128
        'layer': '((layers.*self_attn\\.(qkv|o)_proj)|(layers.*mlp\\.(gate_up|down)_proj))',
        'dp': 0.05                # 論文規格
    }

    config.vision_lora = {
        'r': lora_rank,           # 64（或 256 保持原樣）
        'lora_alpha': lora_alpha, # 128（或 512 保持原樣）
        'layer': 'layers.*((self_attn\\.(qkv_proj|o_proj))|(mlp\\.(gate_up|down)_proj))',
        'dp': 0.05
    }

    # 🔑 關鍵：忽略形狀不匹配，LoRA 權重會被重新初始化
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,  # 🔑 允許形狀不匹配
    )

    model.gradient_checkpointing_enable()

    # LoRA 參數會被隨機初始化為論文規格
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return model, processor, peft_config
```

## 預期結果

執行後應該看到：

```
⚠️  Some weights were not initialized from checkpoint: ...lora_A...lora_B...
✅ 可訓練 LoRA 層: 512 個
✅ LoRA rank: 64 (論文規格)
✅ 可訓練參數: ~200M (約 3.5%)  # 比 r=320 少很多
```

## 訓練考量

### 優點
- ✅ 完全符合論文實驗設定
- ✅ 可公平比較論文結果
- ✅ 參數更少，訓練更快

### 缺點
- ⚠️ 失去預訓練 LoRA 的優勢
- ⚠️ 可能需要更多訓練 epoch 才能收斂
- ⚠️ 初期性能會較差（需要從零學習）

### 建議
1. **先試試預訓練配置**（r=320）訓練幾個 epoch，看看效果
2. **如果要嚴格復現論文**，使用 r=64 從零訓練
3. **記錄兩種配置的性能**，比較差異

## Vision LoRA 配置建議

論文主要關注 **Speech LoRA**（語音發音評估任務），Vision LoRA 可能影響不大。建議：

**選項 1**（保守）：
```python
config.vision_lora = {
    'r': 256,      # 保持預訓練值
    'lora_alpha': 512,
    'dp': 0.0
}
```

**選項 2**（嚴格復現）：
```python
config.vision_lora = {
    'r': 64,       # 論文規格
    'lora_alpha': 128,
    'dp': 0.05
}
```

推薦**選項 1**，因為視覺模態不是主要任務，保留預訓練優勢較安全。
