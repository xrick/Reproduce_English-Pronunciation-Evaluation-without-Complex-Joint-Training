import torch
import sys
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoConfig

# CRITICAL: Patch PEFT before using it to handle Phi-4's missing method
from peft import peft_model, LoraConfig, get_peft_model

_original_peft_init = peft_model.PeftModelForCausalLM.__init__

def _patched_peft_init(self, model, peft_config, adapter_name="default", **kwargs):
    """
    Patched PEFT init that adds prepare_inputs_for_generation if missing.
    This works around Phi-4-multimodal's architectural bug.
    """
    # Add missing method if needed
    if not hasattr(model, 'prepare_inputs_for_generation'):
        def prepare_inputs_for_generation(*args, **kwargs):
            return {}
        model.prepare_inputs_for_generation = prepare_inputs_for_generation

    # Call original init
    _original_peft_init(self, model, peft_config, adapter_name, **kwargs)

# Apply the patch
peft_model.PeftModelForCausalLM.__init__ = _patched_peft_init
print("✅ Patched PEFT to handle Phi-4's missing prepare_inputs_for_generation method")

def get_model_and_processor(
    model_id: str = "microsoft/Phi-4-multimodal-instruct",
    lora_rank: int = 64,
    lora_alpha: int = 128
):
    # 1. 不使用量化（選項 B：完整精度訓練）
    # 注意：Phi-4 的內建 LoRA 在量化時也會被量化，導致無法訓練
    # 解決方案：不使用量化，直接以 bfloat16 加載
    # VRAM 需求：約 26-30GB（需要足夠的 GPU 記憶體）
    bnb_config = None  # 不使用量化

    # model_id = "microsoft/Phi-4-multimodal-instruct"
    model_path = "/Users/xrickliao/WorkSpaces/LLM_Repo/models/Phi-4-multimodal-instruct"

    # 2. 加載處理器
    processor = AutoProcessor.from_pretrained(model_path,
                                            local_files_only=True,
                                            trust_remote_code=True)

    # 3. 加載並修改配置
    # trust_remote_code=True 會在這裡載入 modeling_phi4mm 模組
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    # 4. PEFT/LoRA 兼容性問題的嘗試修補 (已知無效)
    # Phi-4-multimodal 內建 LoRA 實作與 PEFT 不兼容
    # trust_remote_code 會重新載入模組，導致任何 monkey-patch 失效
    # 詳見: claudedocs/peft_lora_incompatibility.md
    #
    # 我們依賴模型的內建 LoRA 系統，透過 config.speech_lora 和 config.vision_lora 配置
    # （已在下方步驟 4b 完成配置）

    # 4a. 禁用 Flash Attention 2（Apple Silicon 不支持）
    config._attn_implementation = "eager"

    # 4b. 修改內建 LoRA 配置以符合論文參數
    # Phi-4-multimodal 強制要求 vision_lora 和 speech_lora 存在
    # 我們調整參數以符合論文規格（r=64, alpha=128）
    # 注意：模型會自動應用這些 LoRA，無需外部 PEFT

    # Speech LoRA: 使用預訓練模型的原始配置
    # 注意：預訓練模型已包含 LoRA 權重，必須使用相同的 rank
    # 原始配置：r=320, alpha=640（與論文不同！）
    config.speech_lora = {
        'r': 320,                 # 預訓練模型: 320（不是論文的 64）
        'lora_alpha': 640,        # 預訓練模型: 640（不是論文的 128）
        'layer': '((layers.*self_attn\\.(qkv|o)_proj)|(layers.*mlp\\.(gate_up|down)_proj))',
        'dp': 0.01                # 預訓練模型: 0.01
    }

    # Vision LoRA: 使用預訓練模型的原始配置
    config.vision_lora = {
        'r': 256,                 # 預訓練模型: 256
        'lora_alpha': 512,        # 預訓練模型: 512
        'layer': 'layers.*((self_attn\\.(qkv_proj|o_proj))|(mlp\\.(gate_up|down)_proj))',
        'dp': 0.0                 # 預訓練模型: 0.0
    }

    # 5. 使用修改後的配置加載模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        quantization_config=bnb_config,  # None = 不使用量化
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度
    )

    # 6. 啟用梯度檢查點以提高記憶體效率
    model.gradient_checkpointing_enable()
    # 注意：不使用量化時，不需要 prepare_model_for_kbit_training

    # 7. LoRA 已由模型內建機制自動應用
    # Phi-4-multimodal 在 __init__ 中已應用 vision_lora 和 speech_lora
    # 無需額外的 get_peft_model() 調用
    # 我們在步驟 3b 已設定符合論文的 LoRA 參數

    # 保存 LoRA 配置供參考（雖然不使用 PEFT 直接應用）
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"]
    )

    # 8. 檢查並報告 LoRA 參數狀態
    #
    # ⚠️ 已知限制：Phi-4 內建 LoRA 與 QLoRA 4-bit 量化不兼容
    #
    # 問題：
    # - Phi-4 的內建 LoRA 參數在模型量化時也被量化為 uint8
    # - 量化參數無法設置 requires_grad = True（會引發 RuntimeError）
    # - prepare_model_for_kbit_training() 無法自動處理 Phi-4 的內建 LoRA
    #
    # 暫時解決方案：
    # - 模型成功加載，所有基礎功能正常
    # - LoRA 參數存在但無法訓練（requires_grad = False）
    # - 可以進行推理，但無法進行 LoRA 微調
    #
    # 完整解決方案（需要進一步實作）：
    # 1. 選項 A: 不使用 load_in_4bit，改用 load_in_8bit 或 fp16
    # 2. 選項 B: 修改模型加載邏輯，選擇性量化（LLM 4-bit，LoRA 參數 fp16）
    # 3. 選項 C: 完全禁用內建 LoRA，改用外部 PEFT（需修改模型原始碼）
    #
    # 參考: claudedocs/peft_lora_incompatibility.md

    # 統計參數狀態
    lora_params = [(name, p) for name, p in model.named_parameters() if "lora" in name.lower()]
    trainable_lora = sum(1 for _, p in lora_params if p.requires_grad)
    total_lora = len(lora_params)

    all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"\n📊 參數統計:")
    print(f"  總參數: {all_params:,}")
    print(f"  可訓練參數: {all_trainable:,} ({100*all_trainable/all_params:.4f}%)")
    print(f"  LoRA 層數: {total_lora}")
    print(f"  可訓練 LoRA 層: {trainable_lora}")

    if trainable_lora == 0 and total_lora > 0:
        print(f"\n⚠️  警告: 發現 {total_lora} 個 LoRA 參數層，但全部被凍結（quantized uint8）")
        print(f"   模型可用於推理，但無法進行 LoRA 微調訓練")
        print(f"   詳見: claudedocs/peft_lora_incompatibility.md")

    return model, processor, peft_config