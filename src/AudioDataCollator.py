from transformers import DataCollatorForSeq2Seq
import numpy as np

class AudioDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # features 是來自資料集的字典列表
        # Phi4MMProcessor expects audios as List[Tuple[numpy_array, sampling_rate]]
        # Convert audio_array from list to numpy array if needed
        audios = []
        for f in features:
            audio_array = f["audio_array"]
            if isinstance(audio_array, list):
                audio_array = np.array(audio_array, dtype=np.float32)
            audios.append((audio_array, f["sampling_rate"]))

        text_inputs = [f["text_input"] for f in features]

        # 處理器處理複雜的多模態填充和 token 化
        batch = self.processor(
            text=text_inputs,
            audios=audios,
            return_tensors="pt",
            padding=True
        )
        
        # 建立標籤 (Labels) 的邏輯
        # 我們希望遮罩 Prompt 的損失，以便我們只訓練 JSON 輸出。
        # 簡化版：
        batch["labels"] = batch["input_ids"].clone()
        
        # (選用但推薦) 遮罩填充 token
        if self.processor.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100
            
        return batch