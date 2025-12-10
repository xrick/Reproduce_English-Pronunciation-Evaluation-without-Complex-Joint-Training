```python

from scipy.stats import pearsonr
import jiwer

def evaluate_model(model, test_dataset):
    model.eval()
    predictions = []
    references = []
    
    for sample in test_dataset:
        # 準備輸入
        inputs = processor(
            text=f"<|user|><|audio_1|>Analyze the pronunciation...<|end|><|assistant|>",
            audios=[sample['audio']['array']],
            return_tensors="pt",
            sampling_rate=16000
        ).to(model.device)
        
        # 生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 解析 JSON
        try:
            # 提取 JSON 部分（啟發式分割）
            json_str = generated_text.split("<|assistant|>")[-1]
            pred_json = json.loads(json_str)
            predictions.append(pred_json)
            references.append(sample)
        except:
            print("Failed to parse JSON")
            
    # 計算 PCC
    pred_acc = [p.get('accuracy', 0) for p in predictions]
    ref_acc = [r['accuracy'] for r in references]
    pcc = pearsonr(pred_acc, ref_acc)
    
    print(f"Accuracy PCC: {pcc}")