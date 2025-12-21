#!/usr/bin/env python3
"""
單一配置訓練腳本
使用方式：
    python train_single_config.py --config pretrained_r320
    python train_single_config.py --config paper_r64
"""

import argparse
import json
import os
import sys
from transformers import TrainingArguments
from data_utility import get_processed_dataset
from model_utility_configs import CONFIGS
from AudioDataCollator import AudioDataCollator


def main():
    parser = argparse.ArgumentParser(description="訓練單一 LoRA 配置")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["pretrained_r320", "paper_r64"],
        help="選擇配置: pretrained_r320 或 paper_r64"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/train/",
        help="訓練數據集路徑"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="訓練 epoch 數（論文建議 3）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="每個設備的批次大小（論文設定 8）"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="梯度累積步數（論文設定 8）"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="學習率（論文設定 2e-5）"
    )

    args = parser.parse_args()

    # 獲取配置
    config = CONFIGS[args.config]

    print("\n" + "="*80)
    print(f"訓練配置: {args.config}")
    print(f"描述: {config['description']}")
    print("="*80)

    # 載入模型
    print(f"\n載入模型...")
    model, processor, peft_config = config["loader"]()

    # 載入數據集
    print(f"\n載入訓練數據集: {args.train_data}")
    train_data = get_processed_dataset(args.train_data)
    print(f"訓練樣本數: {len(train_data)}")

    # 設置輸出目錄
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 訓練參數（根據論文 Table 3 設置）
    training_args = TrainingArguments(
        output_dir=output_dir,

        # 論文超參數
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,

        # 優化器和精度
        optim="adamw_torch",
        bf16=True,

        # 日誌和保存
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",
        save_total_limit=3,

        # 評估
        eval_strategy="no",

        # 其他
        report_to="tensorboard",

        # 內存優化
        gradient_checkpointing=True,

        # 數據處理 - 保留所有欄位給 AudioDataCollator
        remove_unused_columns=False,
    )

    # 創建 Trainer
    # 不使用 SFTTrainer 的自動 tokenization（會導致多模態數據問題）
    # 直接使用 Trainer + AudioDataCollator 處理多模態數據
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=AudioDataCollator(processor),
    )

    # 保存配置信息
    config_info = {
        "config_name": args.config,
        "description": config["description"],
        "speech_lora": config["speech_lora"],
        "vision_lora": config["vision_lora"],
        "trainable_params": config["trainable_params"],
        "training_args": {
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
        }
    }

    config_info_path = os.path.join(output_dir, "training_config.json")
    with open(config_info_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    print(f"\n訓練配置已保存至: {config_info_path}")
    print(f"輸出目錄: {output_dir}")
    print(f"訓練參數: {config['trainable_params']}")
    print(f"有效批次大小: {args.batch_size * args.gradient_accumulation}")

    # 開始訓練
    print(f"\n開始訓練...")
    trainer.train()

    # 保存最終模型
    final_model_dir = f"{output_dir}/final_model"
    print(f"\n保存最終模型至: {final_model_dir}")
    trainer.save_model(final_model_dir)

    # 保存處理器
    processor.save_pretrained(final_model_dir)

    print(f"\n✅ 訓練完成！")
    print(f"模型保存位置: {final_model_dir}")
    print(f"TensorBoard 日誌: {output_dir}/logs")
    print("\n建議下一步:")
    print(f"  tensorboard --logdir {output_dir}/logs")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
