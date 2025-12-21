#!/usr/bin/env python3
"""
驗證雙配置系統是否正確設置
快速測試模型載入和 LoRA 參數配置
"""

import sys
from model_utility_configs import CONFIGS, print_config_comparison

def verify_config(config_name: str):
    """驗證單一配置"""
    print(f"\n{'='*80}")
    print(f"驗證配置: {config_name}")
    print(f"{'='*80}")

    config = CONFIGS[config_name]

    print(f"\n描述: {config['description']}")
    print(f"輸出目錄: {config['output_dir']}")

    try:
        print("\n載入模型...")
        model, processor, peft_config = config["loader"]()

        print("\n✅ 模型載入成功！")

        # 驗證 LoRA 參數
        lora_params = [(name, p) for name, p in model.named_parameters() if "lora" in name.lower()]
        trainable_lora = sum(1 for _, p in lora_params if p.requires_grad)
        total_lora = len(lora_params)
        all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        print(f"\n參數統計:")
        print(f"  總參數: {all_params:,}")
        print(f"  可訓練參數: {all_trainable:,} ({100*all_trainable/all_params:.2f}%)")
        print(f"  LoRA 層數: {total_lora}")
        print(f"  可訓練 LoRA 層: {trainable_lora}")

        # 驗證配置
        print(f"\nLoRA 配置:")
        print(f"  Speech LoRA: {config['speech_lora']}")
        print(f"  Vision LoRA: {config['vision_lora']}")

        # 檢查是否所有 LoRA 層都可訓練
        if trainable_lora == total_lora and trainable_lora > 0:
            print(f"\n✅ 所有 {total_lora} 個 LoRA 層都可訓練")
        else:
            print(f"\n⚠️  警告: 只有 {trainable_lora}/{total_lora} LoRA 層可訓練")

        return True

    except Exception as e:
        print(f"\n❌ 配置驗證失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("雙配置系統驗證")
    print("="*80)

    # 顯示配置對照表
    print_config_comparison()

    # 驗證兩種配置
    results = {}

    print("\n" + "="*80)
    print("開始驗證配置...")
    print("="*80)

    for config_name in ["pretrained_r320", "paper_r64"]:
        results[config_name] = verify_config(config_name)

    # 總結
    print("\n" + "="*80)
    print("驗證結果總結")
    print("="*80)

    for config_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{config_name}: {status}")

    if all(results.values()):
        print("\n✅ 所有配置驗證成功！可以開始訓練。")
        print("\n建議下一步:")
        print("  ./train_both_configs.sh")
        return 0
    else:
        print("\n❌ 部分配置驗證失敗，請檢查錯誤訊息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
