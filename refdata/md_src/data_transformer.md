```python

import json
import os
from datasets import load_dataset, Dataset

class SpeechOceanFormatter:
    def __init__(self, data_root):
        self.data_root = data_root

    def format_sample(self, sample):
        # 提取純量分數
        scores = {
            "accuracy": sample['accuracy'],
            "fluency": sample['fluency'],
            "prosodic": sample['prosodic'],
            "total": sample['total']
        }
        
        # 建構音素轉錄 (Phoneme Transcript)
        # 我們將每個單字的音素列表連接起來
        phonemes = []
        for word in sample['words']:
            # 在真實的 MDD 場景中，我們可能會在此處使用錯讀元數據
            # 對於此重現，我們使用標準音素作為基本轉錄目標
            phonemes.extend(word['phones'])
        
        # 目標 JSON 輸出
        target_output = {
            "accuracy": scores['accuracy'],
            "prosodic": scores['prosodic'],
            "fluency": scores['fluency'],
            "total": scores['total'],
            "word transcript": sample['text'],
            "phoneme transcript": " ".join(phonemes)
        }
        
        # 建構提示 (Prompt)
        # <|audio_1|> 是 Phi-4 用於關注音訊特徵的特定 token
        user_prompt = "<|user|><|audio_1|>Analyze the pronunciation of the audio. Provide accuracy, fluency, prosodic, and total scores, along with word and phoneme transcripts in JSON format.<|end|>"
        assistant_response = f"<|assistant|>{json.dumps(target_output)}<|end|>"
        
        return {
            "audio_path": sample['audio']['path'], 
            "audio_array": sample['audio']['array'],
            "sampling_rate": sample['audio']['sampling_rate'],
            "text_input": user_prompt + assistant_response, # 用於訓練的完整序列
            "prompt_only": user_prompt # 用於推論遮罩
        }

# 加載資料集 (假設使用 huggingface 結構)
raw_dataset = load_dataset("mispeech/speechocean762", split="train")
formatter = SpeechOceanFormatter(data_root=None)
processed_dataset = raw_dataset.map(formatter.format_sample)