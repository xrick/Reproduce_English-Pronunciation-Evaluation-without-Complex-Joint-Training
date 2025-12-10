```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./phi4-capt-reproduction",
    num_train_epochs=4,
    per_device_train_batch_size=4, # 根據 VRAM 調整。24GB 下 4 是安全的。
    gradient_accumulation_steps=16, # 有效批次大小 = 64
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="no", # CAPT 通常離線進行評估
    bf16=True,
    max_seq_length=2048, # 容納音訊 token
    dataset_text_field="text_input", # 佔位符
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=AudioDataCollator(processor),
    peft_config=peft_config
)

# 開始訓練
trainer.train()