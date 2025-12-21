import json
import os
from datasets import load_dataset, load_from_disk, Dataset

class SpeechOceanFormatter:
    def __init__(self, data_root):
        self.data_root = data_root

    def format_sample(self, sample):
        # 處理 TorchCodec AudioDecoder 對象
        # datasets 庫 4.x 版本在可用時使用 TorchCodec
        audio_obj = sample['audio']

        # 檢查是否為 AudioDecoder 對象（TorchCodec）
        if hasattr(audio_obj, 'get_all_samples'):
            # TorchCodec AudioDecoder 對象
            audio_samples = audio_obj.get_all_samples()
            audio_array = audio_samples.data.squeeze(0).numpy()  # 將 torch.Tensor 轉換為 numpy
            sampling_rate = audio_obj.metadata.sample_rate
            audio_path = None  # AudioDecoder 不直接提供路徑
        else:
            # 標準字典格式（舊版 datasets 或無 TorchCodec）
            audio_array = audio_obj['array']
            sampling_rate = audio_obj['sampling_rate']
            audio_path = audio_obj.get('path')

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

        result = {
            "audio_array": audio_array,
            "sampling_rate": sampling_rate,
            "text_input": user_prompt + assistant_response, # 用於訓練的完整序列
            "prompt_only": user_prompt # 用於推論遮罩
        }

        # 僅在可用時添加 audio_path
        if audio_path is not None:
            result["audio_path"] = audio_path

        return result

def get_processed_dataset(data_path):
    # 加載原始資料集
    processed_dataset = load_from_disk(data_path)
    return processed_dataset

if __name__ == "__main__":
    # 加載資料集 (假設使用 huggingface 結構)
    # Option 1: Load from HuggingFace Hub
    raw_dataset = load_dataset("mispeech/speechocean762", split="test")

    # Option 2: Load from saved disk dataset
    # raw_dataset = load_from_disk("../../DataSets/Reproduce_English_Pronunciation/speechocean762_raw_train")

    # raw_dataset = raw_dataset.save_to_disk("../../DataSets/Reproduce_English_Pronunciation/speechocean762_raw_train")
    formatter = SpeechOceanFormatter(data_root=None)
    processed_dataset = raw_dataset.map(formatter.format_sample)
    processed_dataset.save_to_disk("../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/test/")