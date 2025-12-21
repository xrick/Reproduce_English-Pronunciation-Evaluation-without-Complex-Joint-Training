
import json
import os
from data_utility import get_processed_dataset
from model_utility import get_model_and_processor
from trl import SFTTrainer, SFTConfig
from AudioDataCollator import AudioDataCollator
# 1. 呼叫函式取得資料物件 (資料處理邏輯在 data_utils.py 跑)
# 1. 呼叫函式取得資料物件 (資料處理邏輯在 data_utils.py 跑)
train_data_path = "../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/train/"
test_data_path = "../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/test/"

train_data = get_processed_dataset(train_data_path)
# test_data = get_processed_dataset(test_data_path)

# 2. 呼叫函式取得模型物件 (模型載入邏輯在 model_utils.py 跑)
model, processor, peft_config = get_model_and_processor()
_output_dir = "../phi4-capt-reproduction/"
training_args = SFTConfig(
    output_dir=_output_dir,
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
    train_dataset=train_data,
    data_collator=AudioDataCollator(processor),
    peft_config=peft_config
)

# 開始訓練
trainer.train()

trainer.save_model()