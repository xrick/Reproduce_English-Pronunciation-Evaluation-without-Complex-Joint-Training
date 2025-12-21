#!/usr/bin/env python3
"""
雙配置訓練腳本
訓練兩種 LoRA 配置：
1. 預訓練配置 (r=320) → output/pretrained_r320/
2. 論文配置 (r=64) → output/paper_r64/
"""

import json
import os
import sys
from trl import SFTTrainer, SFTConfig
from data_utility import get_processed_dataset
from model_utility_configs import CONFIGS, print_config_comparison
from AudioDataCollator import AudioDataCollator


def train_configuration(config_name: str):
    """
    訓練指定配置的模型

    Args:
        config_name: "pretrained_r320" 或 "paper_r64"
    """
    print("\n" + "="*80)
    print(f"開始訓練配置: {config_name}")
    print("="*80)

    # 獲取配置
    config = CONFIGS[config_name]

    # 載入模型
    print(f"\n載入模型 ({config['description']})...")
    model, processor, peft_config = config["loader"]()

    # 載入數據集
    print("\n載入訓練數據集...")
    train_data_path = "../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/train/"
    train_data = get_processed_dataset(train_data_path)
    print(f"訓練樣本數: {len(train_data)}")

    # 添加 dummy input_ids 以防止 SFTTrainer 自動 tokenization
    # SFTTrainer 檢測到 "input_ids" 欄位後會跳過 tokenization
    # 實際的 tokenization 由 AudioDataCollator 在批次處理時完成
    def add_dummy_input_ids(example):
        example["input_ids"] = [0]  # dummy value, will be replaced by data_collator
        return example

    train_data = train_data.map(add_dummy_input_ids)

    # 設置輸出目錄
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 訓練參數（根據論文 Table 3 設置）
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

        # 序列長度 (SFTConfig 使用 max_length 而非 max_seq_length)
        max_length=2048,                       # 容納音訊 token

        # 日誌和保存
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",                 # 每個 epoch 保存
        save_total_limit=3,                    # 保留最近 3 個 checkpoint

        # 評估 (SFTConfig 使用 eval_strategy 而非 evaluation_strategy)
        eval_strategy="no",                    # 離線評估

        # 其他
        dataset_text_field="text_input",       # 佔位符
        report_to="tensorboard",               # TensorBoard 日誌

        # 內存優化
        gradient_checkpointing=True,
    )

    # 創建 Trainer（明確傳遞 processing_class 避免 SFTTrainer 重新加載）
    # 設置 formatting_func=None 禁用 SFTTrainer 的自動 tokenization
    # 我們使用 AudioDataCollator 處理多模態數據
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=AudioDataCollator(processor),
        peft_config=peft_config,
        processing_class=processor,  # 使用已加載的 processor
        formatting_func=None,  # 禁用自動格式化，使用 data_collator
    )

    # 保存配置信息
    config_info = {
        "config_name": config_name,
        "description": config["description"],
        "speech_lora": config["speech_lora"],
        "vision_lora": config["vision_lora"],
        "trainable_params": config["trainable_params"],
        "training_args": {
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        }
    }

    config_info_path = os.path.join(output_dir, "training_config.json")
    with open(config_info_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"\n訓練配置已保存至: {config_info_path}")

    # 開始訓練
    print(f"\n開始訓練 {config_name}...")
    print(f"輸出目錄: {output_dir}")
    print(f"訓練參數: {config['trainable_params']}")
    print(f"預期訓練時間: {'較短（預訓練 LoRA）' if config_name == 'pretrained_r320' else '較長（從零訓練）'}")

    trainer.train()

    # 保存最終模型
    print(f"\n保存最終模型至: {output_dir}/final_model")
    trainer.save_model(f"{output_dir}/final_model")

    # 保存處理器
    processor.save_pretrained(f"{output_dir}/final_model")

    print(f"\n✅ 配置 {config_name} 訓練完成！")
    print(f"模型保存位置: {output_dir}/final_model")
    print("="*80 + "\n")


def main():
    """主函數：訓練兩種配置"""

    # 顯示配置對照表
    print_config_comparison()

    # 確認訓練
    print("\n" + "="*80)
    print("準備訓練兩種配置:")
    print("1. 預訓練配置 (r=320) → output/pretrained_r320/")
    print("2. 論文配置 (r=64) → output/paper_r64/")
    print("="*80)

    response = input("\n是否繼續？(y/n): ")
    if response.lower() != 'y':
        print("訓練已取消")
        return

    # 訓練順序選擇
    print("\n選擇訓練順序:")
    print("1. 先訓練預訓練配置，再訓練論文配置")
    print("2. 先訓練論文配置，再訓練預訓練配置")
    print("3. 僅訓練預訓練配置")
    print("4. 僅訓練論文配置")

    choice = input("請選擇 (1-4): ")

    try:
        if choice == "1":
            train_configuration("pretrained_r320")
            train_configuration("paper_r64")
        elif choice == "2":
            train_configuration("paper_r64")
            train_configuration("pretrained_r320")
        elif choice == "3":
            train_configuration("pretrained_r320")
        elif choice == "4":
            train_configuration("paper_r64")
        else:
            print("無效選擇")
            return
    except KeyboardInterrupt:
        print("\n\n訓練被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n訓練過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("✅ 所有訓練任務完成！")
    print("="*80)
    print("\n訓練結果:")
    if choice in ["1", "2", "3"]:
        print(f"  預訓練配置: output/pretrained_r320/final_model")
    if choice in ["1", "2", "4"]:
        print(f"  論文配置: output/paper_r64/final_model")
    print("\n建議下一步:")
    print("  1. 使用 src/estimate.py 評估模型性能")
    print("  2. 比較兩種配置的 PCC、WER、PER、F1 指標")
    print("  3. 分析 TensorBoard 日誌: tensorboard --logdir output/")


if __name__ == "__main__":
    main()
