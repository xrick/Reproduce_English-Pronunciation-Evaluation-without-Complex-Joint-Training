# AudioDecoder Compatibility Fix - SpeechOcean762 Dataset

**Date**: 2025-12-20
**System**: macOS 14.5, Python 3.11.6
**Issue**: TypeError when accessing audio data due to TorchCodec AudioDecoder object

---

## Problem Description

### Error Message

```
TypeError: 'torchcodec.decoders.AudioDecoder' object is not subscriptable
```

**Error Location**: `SpeechOceanFormatter.py`, line 42
```python
"audio_path": sample['audio']['path']  # ❌ Attempting dict access on AudioDecoder
```

### Context

After successfully fixing TorchCodec RPATH issues (see `torchcodec_so_files_fix.md`), the script attempted to process the SpeechOcean762 dataset but failed when trying to access audio data using dictionary syntax.

---

## Root Cause Analysis

### Investigation Process

**Step 1: Check Dataset Audio Feature**

```bash
$ python -c "
from datasets import load_dataset
dataset = load_dataset('mispeech/speechocean762', split='train[:1]')
print('Audio feature:', dataset.features['audio'])
"

Audio feature: Audio(sampling_rate=None, decode=True, num_channels=None, stream_index=None)
```

**Finding**: Audio feature has `decode=True` but doesn't indicate which backend is used.

**Step 2: Inspect Actual Audio Object Type**

```bash
$ python -c "
from datasets import load_dataset
dataset = load_dataset('mispeech/speechocean762', split='train[:1]')
sample = dataset[0]
print('Type:', type(sample['audio']))
"

Type: <class 'datasets.features._torchcodec.AudioDecoder'>
```

**Finding**: ❌ The `datasets` library (v4.4.2) automatically uses TorchCodec's `AudioDecoder` when TorchCodec is available, instead of returning a dictionary.

**Step 3: Examine AudioDecoder API**

```bash
$ python -c "
from datasets import load_dataset
dataset = load_dataset('mispeech/speechocean762', split='train[:1]')
audio_decoder = dataset[0]['audio']

print('Attributes:', [attr for attr in dir(audio_decoder) if not attr.startswith('_')])
print('Metadata:', audio_decoder.metadata)
"

Attributes: ['get_all_samples', 'get_samples_played_in_range', 'metadata', 'stream_index']

Metadata: AudioStreamMetadata:
  duration_seconds: 2.58
  sample_rate: 16000
  num_channels: 1
  codec: pcm_s16le
```

**Finding**: AudioDecoder provides:
- `metadata` - Audio stream information (sample_rate, duration, etc.)
- `get_all_samples()` - Method to retrieve audio data
- No dictionary-style access (`['path']`, `['array']`, etc.)

**Step 4: Extract Audio Data**

```bash
$ python -c "
from datasets import load_dataset
dataset = load_dataset('mispeech/speechocean762', split='train[:1]')
audio_decoder = dataset[0]['audio']

audio_samples = audio_decoder.get_all_samples()
print('AudioSamples type:', type(audio_samples))
print('Attributes:', [attr for attr in dir(audio_samples) if not attr.startswith('_')])
print('Data type:', type(audio_samples.data))
print('Data shape:', audio_samples.data.shape)
print('Sample rate:', audio_samples.sample_rate)
"

AudioSamples type: <class 'torchcodec._frame.AudioSamples'>
Attributes: ['data', 'duration_seconds', 'pts_seconds', 'sample_rate']
Data type: <class 'torch.Tensor'>
Data shape: torch.Size([1, 41280])
Sample rate: 16000
```

**Finding**: `get_all_samples()` returns an `AudioSamples` object with:
- `data` - PyTorch tensor with shape `[channels, samples]`
- `sample_rate` - Sampling rate (16000 Hz)
- Audio data is in tensor format, needs conversion to numpy

---

## Root Cause Summary

**Behavior Change in datasets v4.x**:
- **Old behavior** (datasets < 4.0 OR TorchCodec unavailable):
  ```python
  sample['audio'] = {
      'path': '/path/to/audio.wav',
      'array': numpy.ndarray([...]),
      'sampling_rate': 16000
  }
  ```

- **New behavior** (datasets >= 4.0 AND TorchCodec available):
  ```python
  sample['audio'] = AudioDecoder(...)
  # Access via:
  # - audio_decoder.metadata.sample_rate
  # - audio_decoder.get_all_samples().data  # torch.Tensor
  ```

**Impact**: Code assuming dictionary access (`sample['audio']['path']`) fails with `TypeError`.

---

## Solution Implementation

### Strategy

Update `format_sample()` method to handle both formats:
1. **Detect object type** - Check if AudioDecoder (has `get_all_samples` method)
2. **Extract audio data** - Use appropriate API for each format
3. **Convert to numpy** - Ensure consistent output format
4. **Maintain backward compatibility** - Support old dict format

### Code Changes

**File**: `src/SpeechOceanFormatter.py`

**Before** (assumes dictionary format):
```python
def format_sample(self, sample):
    # ... score extraction ...

    return {
        "audio_path": sample['audio']['path'],        # ❌ Fails on AudioDecoder
        "audio_array": sample['audio']['array'],      # ❌ Fails on AudioDecoder
        "sampling_rate": sample['audio']['sampling_rate'],  # ❌ Fails on AudioDecoder
        "text_input": user_prompt + assistant_response,
        "prompt_only": user_prompt
    }
```

**After** (handles both formats):
```python
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
    phonemes = []
    for word in sample['words']:
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
    user_prompt = "<|user|><|audio_1|>Analyze the pronunciation of the audio. Provide accuracy, fluency, prosodic, and total scores, along with word and phoneme transcripts in JSON format.<|end|>"
    assistant_response = f"<|assistant|>{json.dumps(target_output)}<|end|>"

    result = {
        "audio_array": audio_array,
        "sampling_rate": sampling_rate,
        "text_input": user_prompt + assistant_response,
        "prompt_only": user_prompt
    }

    # 僅在可用時添加 audio_path
    if audio_path is not None:
        result["audio_path"] = audio_path

    return result
```

### Key Implementation Details

**1. Type Detection**
```python
if hasattr(audio_obj, 'get_all_samples'):
    # TorchCodec AudioDecoder
```
- Uses duck typing to detect AudioDecoder
- More robust than `isinstance()` checks
- Works across different TorchCodec versions

**2. Audio Data Extraction**
```python
audio_samples = audio_obj.get_all_samples()
audio_array = audio_samples.data.squeeze(0).numpy()
```
- `get_all_samples()` returns AudioSamples object
- `data` is PyTorch tensor with shape `[channels, samples]`
- `squeeze(0)` removes channel dimension (mono audio: `[1, N]` → `[N]`)
- `.numpy()` converts to numpy array for consistency

**3. Metadata Access**
```python
sampling_rate = audio_obj.metadata.sample_rate
```
- Access via `metadata` attribute
- Returns integer (16000 Hz for SpeechOcean762)

**4. Path Handling**
```python
audio_path = None  # AudioDecoder doesn't provide path
```
- AudioDecoder doesn't expose file path
- Set to `None` for TorchCodec format
- Optional field in output (only added if not None)

**5. Backward Compatibility**
```python
else:
    # Standard dict format
    audio_array = audio_obj['array']
    sampling_rate = audio_obj['sampling_rate']
    audio_path = audio_obj.get('path')
```
- Maintains support for old dictionary format
- Uses `.get('path')` to handle missing keys gracefully
- No changes needed for older datasets library versions

---

## Verification

### Test Execution

```bash
$ source venv/bin/activate
$ python src/SpeechOceanFormatter.py

Map: 100%|██████████| 2500/2500 [00:12<00:00, 206.89 examples/s]
Saving the dataset (2/2 shards): 100%|██████████| 2500/2500 [00:00<00:00, 5932.59 examples/s]
```

**Result**: ✅ **Success!** All 2500 samples processed without errors

### Output Verification

```bash
$ ls -lh ../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted/

total 1947904
-rw-r--r--  1 xrickliao  staff   413M Dec 20 13:30 data-00000-of-00002.arrow
-rw-r--r--  1 xrickliao  staff   538M Dec 20 13:30 data-00001-of-00002.arrow
-rw-r--r--  1 xrickliao  staff   3.2K Dec 20 13:30 dataset_info.json
-rw-r--r--  1 xrickliao  staff   309B Dec 20 13:30 state.json
```

**Output**: 951 MB formatted dataset successfully created

### Sample Inspection

```bash
$ python -c "
from datasets import load_from_disk
dataset = load_from_disk('../../DataSets/Reproduce_English_Pronunciation/speechocean762_formatted')
sample = dataset[0]

print('Keys:', list(sample.keys()))
print('Audio array shape:', sample['audio_array'].shape)
print('Audio array type:', type(sample['audio_array']))
print('Sampling rate:', sample['sampling_rate'])
print('Has audio_path:', 'audio_path' in sample)
print('Text input length:', len(sample['text_input']))
"

Keys: ['audio_array', 'sampling_rate', 'text_input', 'prompt_only']
Audio array shape: (41280,)
Audio array type: <class 'numpy.ndarray'>
Sampling rate: 16000
Has audio_path: False
Text input length: 312
```

**Validation**:
- ✅ Audio data correctly converted to numpy array
- ✅ Proper shape (mono audio, 1D array)
- ✅ Correct sampling rate (16000 Hz)
- ✅ Training prompts properly formatted
- ✅ Optional `audio_path` field excluded (as expected for AudioDecoder)

---

## Additional Fix: load_from_disk

The user also encountered a secondary issue when trying to load a saved dataset:

### Error
```python
raw_dataset = load_dataset("../../DataSets/.../speechocean762_raw_train", split="train")
# ValueError: You are trying to load a dataset that was saved using `save_to_disk`.
# Please use `load_from_disk` instead.
```

### Solution
```python
# Import load_from_disk
from datasets import load_dataset, load_from_disk, Dataset

# Use correct loading method
raw_dataset = load_from_disk("../../DataSets/Reproduce_English_Pronunciation/speechocean762_raw_train")
```

---

## Lessons Learned

### Technical Insights

1. **Library Version Dependencies**:
   - The `datasets` library v4.x automatically uses TorchCodec when available
   - No configuration option to disable TorchCodec in current version
   - Code must handle both old and new formats for compatibility

2. **Duck Typing for Detection**:
   - Using `hasattr(obj, 'method_name')` is more robust than `isinstance()`
   - Works across library versions without importing internal classes
   - Prevents ImportError if TorchCodec unavailable

3. **Tensor to Numpy Conversion**:
   - TorchCodec returns `torch.Tensor` with shape `[channels, samples]`
   - `.squeeze(0)` removes channel dimension for mono audio
   - `.numpy()` conversion maintains data type and value precision

4. **Backward Compatibility**:
   - Always support multiple data formats when external dependencies change
   - Use graceful fallbacks (`.get()` instead of `[]` for optional keys)
   - Document which format is expected vs. supported

### Code Quality

1. **Defensive Programming**:
   - Check object type before accessing attributes
   - Handle missing optional fields gracefully
   - Provide clear comments explaining format differences

2. **Maintainability**:
   - Centralize format detection logic
   - Use descriptive variable names (`audio_obj`, `audio_samples`)
   - Comment in native language (Chinese) for local team consistency

3. **Testing Strategy**:
   - Test with actual dataset samples, not synthetic data
   - Verify output shapes and types match expectations
   - Check performance with full dataset (2500 samples in 12 seconds = 208 samples/sec)

---

## Alternative Solutions Considered

### Option 1: Disable TorchCodec (Not Possible)

**Attempted**:
```python
import os
os.environ['HF_DATASETS_DISABLE_TORCHCODEC'] = '1'
```

**Result**: Environment variable not recognized by datasets library

**Reason**: No official way to disable TorchCodec in datasets v4.x

### Option 2: Force Old Decoder via cast_column (Failed)

**Attempted**:
```python
from datasets import load_dataset, Audio
dataset = load_dataset('mispeech/speechocean762', split='train[:1]')
dataset = dataset.cast_column('audio', Audio(decode=True))
```

**Result**: Still returns AudioDecoder object

**Reason**: `cast_column` doesn't override TorchCodec backend selection

### Option 3: Downgrade datasets Library (Not Recommended)

**Option**: Install datasets < 4.0

**Pros**: Would use old dictionary format automatically

**Cons**:
- Loses new features and bug fixes
- May have compatibility issues with other dependencies
- Not sustainable long-term solution

**Decision**: Rejected in favor of forward-compatible code

### Option 4: Handle Both Formats (SELECTED)

**Pros**:
- Works with any datasets library version
- No dependency version constraints
- Future-proof for datasets library updates
- Minimal code complexity

**Cons**:
- Slightly more code than format-specific implementation
- Requires understanding both APIs

**Decision**: Selected as optimal solution

---

## Impact on Project

### Training Pipeline

**Status**: ✅ **Ready for Training**

The formatted dataset now contains:
- **audio_array**: NumPy array (float32, mono, 16kHz)
- **sampling_rate**: Integer (16000)
- **text_input**: Full training sequence with prompts and JSON responses
- **prompt_only**: User prompt for inference-time masking

**Next Steps**:
1. Implement data collator ([src/data_collator.py](../src/data_collator.py))
2. Set up LoRA training configuration ([src/SFTTrainer.py](../src/SFTTrainer.py))
3. Fix critical issues identified in [CLAUDE.md](../CLAUDE.md):
   - Add control tokens (`<|APA|>`, `<|MDD|>`)
   - Include detailed scoring rubrics in prompts
   - Implement prompt masking in data collator
   - Correct hyperparameters (learning rate, batch size, epochs)

### Performance

**Processing Speed**: 208 samples/second on Apple Silicon
**Memory Usage**: Reasonable (951 MB output for 2500 samples)
**Compatibility**: Works with both TorchCodec and legacy formats

---

## Status

**AudioDecoder Compatibility**: ✅ **FULLY RESOLVED**
- Code handles both TorchCodec AudioDecoder and legacy dict formats
- All 2500 training samples successfully processed
- Dataset ready for training pipeline integration

**Related Issues**: ✅ **RESOLVED**
- TorchCodec dylib loading: Fixed in `torchcodec_dylib_fix.md`
- TorchCodec .so files loading: Fixed in `torchcodec_so_files_fix.md`
- load_from_disk usage: Fixed in this update

---

## References

- TorchCodec Documentation: https://github.com/pytorch/torchcodec
- Datasets Library: https://huggingface.co/docs/datasets
- SpeechOcean762 Dataset: https://huggingface.co/datasets/mispeech/speechocean762
- Related Fix: `torchcodec_so_files_fix.md`
