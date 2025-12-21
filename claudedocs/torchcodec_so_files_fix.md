# TorchCodec .so Files RPATH Issue - Additional Fix Required

**Date**: 2025-12-20
**System**: macOS 14.5 (Darwin 23.5.0), Apple Silicon (arm64)
**Issue**: TorchCodec error persisted after fixing .dylib files due to missing .so file RPATH entries

---

## Problem Description

### Context

After successfully fixing RPATH entries for all TorchCodec `.dylib` files (see `torchcodec_dylib_fix.md`), the TorchCodec loading error **reappeared** when running `SpeechOceanFormatter.py`.

### Error Message

```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed in your environment.
  2. The PyTorch version (2.9.1) is not compatible with TorchCodec.
  3. Another runtime dependency; see exceptions below.

[start of libtorchcodec loading traceback]
FFmpeg version 8: Could not load this library: .../libtorchcodec_core8.dylib
FFmpeg version 7: dlopen(.../libtorchcodec_pybind_ops7.so, 0x0002):
  Library not loaded: @rpath/libavutil.59.dylib
  Referenced from: .../libtorchcodec_pybind_ops7.so
  Reason: no LC_RPATH's found
FFmpeg version 6: Could not load this library: .../libtorchcodec_core6.dylib
...
[end of libtorchcodec loading traceback]
```

### Key Observation

The error now shows a **different file type**:
- ❌ **Before**: `libtorchcodec_core7.dylib` (dynamic library)
- ❌ **Now**: `libtorchcodec_pybind_ops7.so` (Python extension module)

---

## Root Cause Analysis

### Investigation

**Step 1: Identify All TorchCodec Shared Libraries**

```bash
$ ls venv/lib/python3.11/site-packages/torchcodec/*.{dylib,so}
```

**Result**: TorchCodec ships with **two types** of shared libraries:

**Type 1: Dynamic Libraries (.dylib)** - 10 files
```
libtorchcodec_core4.dylib          # FFmpeg 4 support
libtorchcodec_core5.dylib          # FFmpeg 5 support
libtorchcodec_core6.dylib          # FFmpeg 6 support
libtorchcodec_core7.dylib          # FFmpeg 7 support ✅ FIXED
libtorchcodec_core8.dylib          # FFmpeg 8 support
libtorchcodec_custom_ops4.dylib
libtorchcodec_custom_ops5.dylib
libtorchcodec_custom_ops6.dylib
libtorchcodec_custom_ops7.dylib
libtorchcodec_custom_ops8.dylib
```

**Type 2: Python Extension Modules (.so)** - 5 files
```
libtorchcodec_pybind_ops4.so       # Python bindings for FFmpeg 4
libtorchcodec_pybind_ops5.so       # Python bindings for FFmpeg 5
libtorchcodec_pybind_ops6.so       # Python bindings for FFmpeg 6
libtorchcodec_pybind_ops7.so       # Python bindings for FFmpeg 7 ❌ NOT FIXED
libtorchcodec_pybind_ops8.so       # Python bindings for FFmpeg 8
```

**Step 2: Check .so File Dependencies**

```bash
$ otool -L venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops7.so

libtorchcodec_pybind_ops7.so:
	@rpath/libtorchcodec_core7.dylib      # ← TorchCodec dylib
	@rpath/libavutil.59.dylib             # ← FFmpeg library
	@rpath/libavcodec.61.dylib            # ← FFmpeg library
	@rpath/libavformat.61.dylib           # ← FFmpeg library
	@rpath/libavdevice.61.dylib           # ← FFmpeg library
	@rpath/libavfilter.10.dylib           # ← FFmpeg library
	@rpath/libswscale.8.dylib             # ← FFmpeg library
	@rpath/libswresample.5.dylib          # ← FFmpeg library
	@rpath/libtorch.dylib                 # ← PyTorch library
	@rpath/libtorch_cpu.dylib             # ← PyTorch library
	@rpath/libc10.dylib                   # ← PyTorch library
	...
```

**Finding**: `.so` files depend on:
1. TorchCodec `.dylib` files in the same directory
2. FFmpeg libraries in `/opt/homebrew/lib`
3. PyTorch libraries in `../../torch/lib`

**Step 3: Check RPATH Configuration**

```bash
$ otool -l venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops7.so | grep -A2 LC_RPATH
# No output
```

**Finding**: ❌ **NO RPATH entries** in `.so` files

---

## Root Cause Summary

**Problem**: The initial fix only addressed `.dylib` files (10 files), but **completely missed** the `.so` files (5 files).

**Why .so Files Matter**: Python imports TorchCodec by loading the `.so` extension modules first, which then load the `.dylib` files. Without RPATH in `.so` files, the entire chain fails.

**Dependency Chain**:
```
Python
  └─> libtorchcodec_pybind_ops7.so  ❌ Missing RPATH
       ├─> libtorchcodec_core7.dylib      ✅ Has RPATH
       ├─> libavutil.59.dylib              (needs path)
       ├─> libavcodec.61.dylib             (needs path)
       └─> libc10.dylib                    (needs path)
```

---

## Solution Implementation

### Strategy

Add **three RPATH entries** to each `.so` file:

1. **`@loader_path/../../torch/lib`** - For PyTorch libraries
2. **`/opt/homebrew/lib`** - For FFmpeg libraries
3. **`@loader_path`** - For TorchCodec dylibs in same directory

### Execution Commands

#### Phase 1: Add PyTorch Library RPATH

```bash
install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops4.so

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops5.so

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops6.so

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops7.so

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops8.so
```

#### Phase 2: Add FFmpeg Library RPATH

```bash
install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops4.so

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops5.so

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops6.so

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops7.so

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops8.so
```

#### Phase 3: Add TorchCodec Dylib RPATH

```bash
install_name_tool -add_rpath "@loader_path" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops4.so

install_name_tool -add_rpath "@loader_path" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops5.so

install_name_tool -add_rpath "@loader_path" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops6.so

install_name_tool -add_rpath "@loader_path" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops7.so

install_name_tool -add_rpath "@loader_path" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops8.so
```

**Note**: `@loader_path` (without `/../../`) resolves to the directory containing the `.so` file, which is where the `.dylib` files are located.

#### Phase 4: Re-sign Modified Libraries

```bash
cd venv/lib/python3.11/site-packages/torchcodec

codesign --force --sign - libtorchcodec_pybind_ops4.so
codesign --force --sign - libtorchcodec_pybind_ops5.so
codesign --force --sign - libtorchcodec_pybind_ops6.so
codesign --force --sign - libtorchcodec_pybind_ops7.so
codesign --force --sign - libtorchcodec_pybind_ops8.so
```

**Output**: "replacing existing signature" for each file

---

## Verification

### Verify RPATH Addition

```bash
$ otool -l venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_pybind_ops7.so | grep -A2 LC_RPATH

          cmd LC_RPATH
      cmdsize 48
         path @loader_path/../../torch/lib (offset 12)
--
          cmd LC_RPATH
      cmdsize 32
         path /opt/homebrew/lib (offset 12)
--
          cmd LC_RPATH
      cmdsize 32
         path @loader_path (offset 12)
```

**Result**: ✅ All three RPATH entries present

### Test TorchCodec Import

```bash
$ source venv/bin/activate
$ python -c "import torchcodec; print('✅ TorchCodec imported successfully')"

✅ TorchCodec imported successfully
```

**Result**: ✅ **Import successful!**

### Test Script Execution

```bash
$ python src/SpeechOceanFormatter.py

Map:   0%|          | 0/2500 [00:00<?, ? examples/s]
Traceback (most recent call last):
  ...
  File "src/SpeechOceanFormatter.py", line 42, in format_sample
    "audio_path": sample['audio']['path'],
TypeError: 'torchcodec.decoders.AudioDecoder' object is not subscriptable
```

**Result**: ✅ **TorchCodec loads successfully** - script now fails at a different point (data format issue, not library loading)

---

## Complete Fix Summary

### Total Files Modified

**15 shared library files total**:
- 10 `.dylib` files (from initial fix)
- 5 `.so` files (from this additional fix)

### RPATH Configuration Per File Type

**For .dylib files** (2 RPATH entries):
1. `@loader_path/../../torch/lib` - PyTorch libraries
2. `/opt/homebrew/lib` - FFmpeg libraries

**For .so files** (3 RPATH entries):
1. `@loader_path/../../torch/lib` - PyTorch libraries
2. `/opt/homebrew/lib` - FFmpeg libraries
3. `@loader_path` - TorchCodec dylibs in same directory

### File List with Modifications

```
venv/lib/python3.11/site-packages/torchcodec/
├── libtorchcodec_core4.dylib                  ✅ Fixed (2 RPATHs)
├── libtorchcodec_core5.dylib                  ✅ Fixed (2 RPATHs)
├── libtorchcodec_core6.dylib                  ✅ Fixed (2 RPATHs)
├── libtorchcodec_core7.dylib                  ✅ Fixed (2 RPATHs)
├── libtorchcodec_core8.dylib                  ✅ Fixed (2 RPATHs)
├── libtorchcodec_custom_ops4.dylib            ✅ Fixed (2 RPATHs)
├── libtorchcodec_custom_ops5.dylib            ✅ Fixed (2 RPATHs)
├── libtorchcodec_custom_ops6.dylib            ✅ Fixed (2 RPATHs)
├── libtorchcodec_custom_ops7.dylib            ✅ Fixed (2 RPATHs)
├── libtorchcodec_custom_ops8.dylib            ✅ Fixed (2 RPATHs)
├── libtorchcodec_pybind_ops4.so               ✅ Fixed (3 RPATHs)
├── libtorchcodec_pybind_ops5.so               ✅ Fixed (3 RPATHs)
├── libtorchcodec_pybind_ops6.so               ✅ Fixed (3 RPATHs)
├── libtorchcodec_pybind_ops7.so               ✅ Fixed (3 RPATHs)
└── libtorchcodec_pybind_ops8.so               ✅ Fixed (3 RPATHs)
```

---

## Updated Automation Script

```bash
#!/bin/bash
# fix_torchcodec_rpath_complete.sh
# Complete fix for TorchCodec RPATH configuration (both .dylib and .so files)

set -e

VENV_PATH="${1:-venv}"
TORCHCODEC_DIR="$VENV_PATH/lib/python3.11/site-packages/torchcodec"
PYTORCH_RPATH="@loader_path/../../torch/lib"
FFMPEG_RPATH="/opt/homebrew/lib"
LOCAL_RPATH="@loader_path"

echo "🔧 Fixing TorchCodec RPATH configuration (complete fix)..."

# Check if torchcodec directory exists
if [ ! -d "$TORCHCODEC_DIR" ]; then
    echo "❌ Error: TorchCodec directory not found at $TORCHCODEC_DIR"
    exit 1
fi

# Fix .dylib files (2 RPATH entries each)
echo "📦 Processing .dylib files..."
for dylib in "$TORCHCODEC_DIR"/*.dylib; do
    [ -f "$dylib" ] || continue
    echo "  → $(basename "$dylib")"

    install_name_tool -add_rpath "$PYTORCH_RPATH" "$dylib" 2>/dev/null || true
    install_name_tool -add_rpath "$FFMPEG_RPATH" "$dylib" 2>/dev/null || true
    codesign --force --sign - "$dylib" 2>/dev/null
done

# Fix .so files (3 RPATH entries each)
echo "🐍 Processing .so files..."
for sofile in "$TORCHCODEC_DIR"/*.so; do
    [ -f "$sofile" ] || continue
    echo "  → $(basename "$sofile")"

    install_name_tool -add_rpath "$PYTORCH_RPATH" "$sofile" 2>/dev/null || true
    install_name_tool -add_rpath "$FFMPEG_RPATH" "$sofile" 2>/dev/null || true
    install_name_tool -add_rpath "$LOCAL_RPATH" "$sofile" 2>/dev/null || true
    codesign --force --sign - "$sofile" 2>/dev/null
done

echo "✅ RPATH fix completed successfully!"
echo ""
echo "Files modified:"
echo "  - 10 .dylib files (2 RPATH entries each)"
echo "  - 5 .so files (3 RPATH entries each)"
echo ""
echo "Test with: source $VENV_PATH/bin/activate && python -c 'import torchcodec'"
```

**Usage**:
```bash
chmod +x fix_torchcodec_rpath_complete.sh
./fix_torchcodec_rpath_complete.sh venv
```

---

## Lessons Learned

### Technical Insights

1. **Check All Shared Library Types**: On macOS, Python packages can include:
   - `.dylib` - Standard dynamic libraries
   - `.so` - Python extension modules (also Mach-O format on macOS)
   - Both types need RPATH configuration

2. **Understand Loading Chain**: Python extension modules (`.so`) often load regular libraries (`.dylib`). RPATH must be configured at the **entry point** of the chain.

3. **@loader_path Context Matters**:
   - In `.so` files: `@loader_path` = directory containing the `.so` file
   - In `.dylib` files: `@loader_path` = directory containing the `.dylib` file
   - Both resolve at load time based on the **loading library**, not the loaded library

### Process Improvements

1. **Complete Inventory First**: Before fixing RPATH issues, inventory **all** shared library files:
   ```bash
   find venv -name "*.dylib" -o -name "*.so" | grep torchcodec
   ```

2. **Verify Loading Order**: Use `DYLD_PRINT_LIBRARIES=1` to see which library fails to load first

3. **Test Incrementally**: After fixing each file type, test imports to catch issues early

---

## Next Issue: AudioDecoder Subscript Error

The script now successfully loads TorchCodec but encounters a **data format issue**:

```python
File "src/SpeechOceanFormatter.py", line 42, in format_sample
    "audio_path": sample['audio']['path'],
TypeError: 'torchcodec.decoders.AudioDecoder' object is not subscriptable
```

### Problem

When the `datasets` library loads audio data with TorchCodec enabled, it returns an `AudioDecoder` object instead of a dictionary.

**Old Format** (without TorchCodec):
```python
sample['audio'] = {
    'path': '/path/to/file.wav',
    'array': numpy.ndarray,
    'sampling_rate': 16000
}
```

**New Format** (with TorchCodec):
```python
sample['audio'] = AudioDecoder(...)  # Not subscriptable
```

### Solution Options

**Option 1: Disable TorchCodec for Dataset Loading**
```python
raw_dataset = load_dataset(
    "mispeech/speechocean762",
    split="train",
    # Disable TorchCodec decoder
    audio_backend=None  # or "soundfile", "librosa"
)
```

**Option 2: Update format_sample() to Handle AudioDecoder**
```python
def format_sample(self, sample):
    # Check if audio is AudioDecoder object
    if hasattr(sample['audio'], 'decode'):
        # TorchCodec AudioDecoder - need to decode first
        audio_data = sample['audio'].decode()
        audio_path = sample['audio'].path
        audio_array = audio_data['array']
        sampling_rate = audio_data['sampling_rate']
    else:
        # Standard dict format
        audio_path = sample['audio']['path']
        audio_array = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']

    # Rest of formatting logic...
```

**Option 3: Set Audio Format Explicitly**
```python
raw_dataset = load_dataset("mispeech/speechocean762", split="train")
# Ensure audio is decoded to standard format
raw_dataset = raw_dataset.cast_column("audio", Audio(decode=True))
```

**Recommended**: **Option 1** - Disable TorchCodec for dataset loading since it's only needed for model inference, not data preprocessing.

---

## Status

**TorchCodec Library Loading**: ✅ **FULLY RESOLVED**
- All 15 shared library files (.dylib + .so) have proper RPATH configuration
- TorchCodec imports successfully
- All dependencies (PyTorch, FFmpeg) load correctly

**Script Execution**: ⚠️ **New Issue** (unrelated to library loading)
- Data format incompatibility with TorchCodec AudioDecoder
- Requires code update in `SpeechOceanFormatter.py`

---

## References

- Initial fix: `torchcodec_dylib_fix.md`
- TorchCodec GitHub: https://github.com/pytorch/torchcodec
- Datasets library documentation: https://huggingface.co/docs/datasets/audio_load
- macOS dyld documentation: https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/DynamicLibraries/
