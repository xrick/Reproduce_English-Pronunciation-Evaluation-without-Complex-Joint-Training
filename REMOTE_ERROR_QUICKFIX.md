# Remote Training Error - Quick Fix Guide

## 🔴 Error Summary

```
Exception: expected value at line 1 column 1
```

**問題**: tokenizer.json 文件損壞
**影響**: 無法加載模型處理器
**嚴重性**: 🔴 Critical（阻止訓練）

## ⚡ 快速修復（3 分鐘）

### 方法 1: 自動修復腳本（推薦）

```bash
# 1. 傳輸修復腳本到遠程機器
scp fix_remote_tokenizer.sh user@remote:/path/to/project/

# 2. SSH 到遠程機器
ssh user@remote
cd /path/to/project

# 3. 運行修復腳本
chmod +x fix_remote_tokenizer.sh
./fix_remote_tokenizer.sh
```

**腳本功能**:
- ✅ 自動檢測 tokenizer.json 是否損壞
- ✅ 備份舊文件
- ✅ 重新下載正確文件
- ✅ 驗證修復成功

### 方法 2: 手動修復

```bash
# 1. SSH 到遠程機器
ssh user@remote

# 2. 進入模型目錄
cd /datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct

# 3. 檢查文件
ls -lh tokenizer.json
python3 -c "import json; json.load(open('tokenizer.json'))" || echo "Corrupted!"

# 4. 重新下載
pip install -U huggingface_hub
python3 << 'EOF'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="microsoft/phi-4-multimodal-instruct",
    filename="tokenizer.json",
    local_dir=".",
    local_dir_use_symlinks=False,
    force_download=True
)
print("✅ Downloaded!")
EOF

# 5. 驗證
python3 << 'EOF'
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
print(f"✅ Success! Vocab size: {processor.tokenizer.vocab_size:,}")
EOF
```

### 方法 3: 使用在線模型（最簡單）

修改 `src/model_utility_configs.py`:

```python
# 找到這行:
model_path = "/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct"

# 改為:
model_path = "microsoft/phi-4-multimodal-instruct"
```

**優點**:
- ✅ 無需手動修復
- ✅ 自動驗證完整性
- ✅ 首次運行後會緩存

**缺點**:
- ⚠️ 首次需要網絡連接
- ⚠️ 初次下載較慢

## 🔍 診斷步驟

### 1. 檢查文件完整性

```bash
cd /datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct

# 檢查文件大小（應該 > 0）
ls -lh tokenizer.json

# 檢查 JSON 有效性
python3 -c "import json; json.load(open('tokenizer.json'))" && echo "✅ Valid" || echo "❌ Corrupted"
```

### 2. 檢查權限

```bash
# 檢查文件權限
ls -l tokenizer.json

# 如果權限有問題
chmod 644 tokenizer.json
```

### 3. 檢查磁盤空間

```bash
# 檢查可用空間
df -h /datas/store162/xrick/LLM_Repo/models/
```

## ⚠️ 關於警告訊息

這些警告 **可以忽略**（不影響訓練）:

```
The module name  (originally ) is not a valid Python identifier.
Please rename the original module to avoid import issues.
```

- **來源**: transformers 模塊名稱檢查
- **影響**: 無（只是警告）
- **處理**: 不需要處理

**真正的錯誤** 是 tokenizer 加載失敗。

## 📋 完整檢查清單

- [ ] ✅ tokenizer.json 文件存在
- [ ] ✅ tokenizer.json 不為空（檢查大小 > 0）
- [ ] ✅ tokenizer.json 是有效的 JSON
- [ ] ✅ 文件權限正確（644 或 644）
- [ ] ✅ 可以成功加載 AutoProcessor
- [ ] ✅ 磁盤空間充足

## 🚀 修復後測試

```bash
# 激活環境
source venv/bin/activate
cd src

# 測試模型加載
python3 << 'EOF'
from model_utility_configs import CONFIGS

config = CONFIGS["paper_r64"]
model, processor, peft_config = config["loader"]()
print("✅ Model loaded successfully!")
print(f"✅ Trainable params: {config['trainable_params']}")
EOF

# 如果成功，開始訓練
python train_single_config_remote.py --config paper_r64 --gpus 0
```

## 📖 相關文檔

- **詳細說明**: `claudedocs/remote_error_tokenizer_fix.md`
- **遠程訓練指南**: `REMOTE_TRAINING_QUICKSTART.md`
- **完整文檔**: `claudedocs/remote_training_guide.md`

## 💡 預防措施

### 傳輸完整模型目錄

如果要從 Mac 傳輸到 Remote:

```bash
# 在 Mac 上壓縮
cd /Users/xrickliao/WorkSpaces/LLM_Repo/models
tar czf phi4-model.tar.gz Phi-4-multimodal-instruct/

# 計算校驗和
md5 phi4-model.tar.gz > phi4-model.tar.gz.md5

# 傳輸
scp phi4-model.tar.gz phi4-model.tar.gz.md5 user@remote:/path/

# 在 Remote 上
cd /datas/store162/xrick/LLM_Repo/models/
md5sum -c phi4-model.tar.gz.md5  # 驗證完整性
tar xzf phi4-model.tar.gz
```

### 使用 rsync（更安全）

```bash
# 同步整個模型目錄
rsync -avz --progress \
  /Users/xrickliao/WorkSpaces/LLM_Repo/models/Phi-4-multimodal-instruct/ \
  user@remote:/datas/store162/xrick/LLM_Repo/models/Phi-4-multimodal-instruct/

# rsync 會自動驗證文件完整性
```

## ✅ 預期結果

修復成功後，您應該看到:

```
✅ Tokenizer loaded successfully!
✅ Vocab size: 51,200
✅ Model loaded successfully!
✅ Trainable params: ~200M (3.5%)
```

然後可以正常開始訓練。

---

**時間估計**: 2-5 分鐘修復 + 1 分鐘驗證 = **總計 3-6 分鐘**
