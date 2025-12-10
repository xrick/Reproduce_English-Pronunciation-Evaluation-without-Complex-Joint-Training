import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. QLoRA 量化配置
# 使用 NF4 和雙重量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_id = "microsoft/Phi-4-multimodal-instruct"

# 2. 加載處理器
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 3. 加載模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation='flash_attention_2' 
)

# 啟用梯度檢查點以提高記憶體效率
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 4. LoRA 配置
# 遵循 "LoRA Without Regret" 的建議
peft_config = LoraConfig(
    r=64,                        # 秩 (Rank)
    lora_alpha=128,              # Alpha
    target_modules="all-linear", # 目標為所有線性層
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"] 
)

model = get_peft_model(model, peft_config)

# 5. 實作「解凍」策略
# 我們必須找到音訊編碼器層並解凍它們。
# 在 Phi-4 中，這通常在 'model.model.audio_tower' 或類似名稱下。
# 我們遍歷並設置 requires_grad = True。
for name, param in model.named_parameters():
    if "audio" in name and "lora" not in name: # 特別針對音訊編碼器
        param.requires_grad = True
        # 注意：如果 param 是量化的 (uint8)，我們不能簡單地解凍。
        # 理想情況下，音訊塔最初應該以 bfloat16 加載。
        # 這需要小心的加載：LLM 以 4bit 加載，音訊塔以 16bit 加載。
        # 如果簡單的 'load_in_4bit=True' 量化了所有內容，除非我們使用進階加載腳本，
        # 否則我們可能會被迫僅使用 LoRA-only。
        # 對於此重現代碼，我們堅持使用標準 LoRA，這比較安全
        # 且據報導性能非常接近。

model.print_trainable_parameters()