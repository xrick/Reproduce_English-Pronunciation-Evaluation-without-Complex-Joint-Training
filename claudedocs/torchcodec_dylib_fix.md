# TorchCodec Dynamic Library Loading Issue - Resolution Report

**Date**: 2025-12-20
**System**: macOS 14.5 (Darwin 23.5.0), Apple Silicon (arm64)
**Python**: 3.11.6
**Issue**: RuntimeError when importing TorchCodec due to missing RPATH entries

---

## Problem Description

### Initial Error

When executing `src/SpeechOceanFormatter.py`, the following runtime error occurred:

```
RuntimeError: Could not load libtorchcodec. Likely causes:
  1. FFmpeg is not properly installed in your environment.
  2. The PyTorch version (2.9.1) is not compatible with TorchCodec.
  3. Another runtime dependency; see exceptions below.

[start of libtorchcodec loading traceback]
FFmpeg version 8: Could not load this library: .../libtorchcodec_core8.dylib
FFmpeg version 7: Could not load this library: .../libtorchcodec_core7.dylib
FFmpeg version 6: Could not load this library: .../libtorchcodec_core6.dylib
FFmpeg version 5: Could not load this library: .../libtorchcodec_core5.dylib
FFmpeg version 4: Could not load this library: .../libtorchcodec_core4.dylib
[end of libtorchcodec loading traceback]
```

### User Context

- **Task**: Running data transformation script for SpeechOcean762 dataset
- **Environment**: Python virtual environment (`venv`) with PyTorch 2.9.0
- **FFmpeg**: Version 7.1.1 installed via Homebrew at `/opt/homebrew/bin/ffmpeg`

---

## Root Cause Analysis

### Investigation Process

#### Step 1: Verify System Dependencies

```bash
# Check FFmpeg installation
$ which ffmpeg
/opt/homebrew/bin/ffmpeg

$ ffmpeg -version | head -5
ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers
...
libavutil      59. 39.100 / 59. 39.100
libavcodec     61. 19.101 / 61. 19.101
```

**Finding**: ✅ FFmpeg 7.1.1 properly installed via Homebrew

#### Step 2: Verify Python Environment

```bash
$ source venv/bin/activate
$ python -c "import torch; print(f'PyTorch: {torch.__version__}')"
PyTorch: 2.9.0
```

**Finding**: ✅ PyTorch 2.9.0 installed correctly in virtual environment

#### Step 3: Check TorchCodec Installation

```bash
$ ls venv/lib/python3.11/site-packages/torchcodec/*.dylib
libtorchcodec_core4.dylib
libtorchcodec_core5.dylib
libtorchcodec_core6.dylib
libtorchcodec_core7.dylib
libtorchcodec_core8.dylib
libtorchcodec_custom_ops4.dylib
libtorchcodec_custom_ops5.dylib
libtorchcodec_custom_ops6.dylib
libtorchcodec_custom_ops7.dylib
libtorchcodec_custom_ops8.dylib
```

**Finding**: ✅ All TorchCodec dylib files present

#### Step 4: Inspect Dynamic Library Dependencies

```bash
$ otool -L venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib:
	@rpath/libtorchcodec_core7.dylib
	@rpath/libc10.dylib                    # ← PyTorch library
	@rpath/libavutil.59.dylib              # ← FFmpeg library
	@rpath/libavcodec.61.dylib             # ← FFmpeg library
	@rpath/libavformat.61.dylib            # ← FFmpeg library
	...
	@rpath/libtorch.dylib                  # ← PyTorch library
	@rpath/libtorch_cpu.dylib              # ← PyTorch library
```

**Finding**: Libraries reference `@rpath/` for dependencies

#### Step 5: Check RPATH Configuration

```bash
$ otool -l venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib | grep -A2 LC_RPATH
# No output
```

**Finding**: ❌ **No RPATH entries found** - this is the root cause!

#### Step 6: Confirm Root Cause with Direct Loading

```bash
$ python -c "import ctypes; ctypes.CDLL('venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib')"
OSError: dlopen(...): Library not loaded: @rpath/libc10.dylib
  Referenced from: .../libtorchcodec_core7.dylib
  Reason: no LC_RPATH's found
```

**Finding**: ✅ Confirmed - missing RPATH prevents dylib from finding dependencies

#### Step 7: Locate Required Dependencies

```bash
# Find PyTorch libraries
$ find venv -name "libc10.dylib"
venv/lib/python3.11/site-packages/torch/lib/libc10.dylib

# Find FFmpeg libraries
$ ls /opt/homebrew/lib/libav*.dylib | head -5
/opt/homebrew/lib/libavcodec.61.19.101.dylib
/opt/homebrew/lib/libavcodec.61.dylib
/opt/homebrew/lib/libavdevice.61.3.100.dylib
...
```

**Finding**: Dependencies exist at known locations:
- PyTorch libs: `venv/lib/python3.11/site-packages/torch/lib/`
- FFmpeg libs: `/opt/homebrew/lib/`

---

## Root Cause Summary

**Issue**: TorchCodec dylibs shipped with **zero RPATH entries** in their Mach-O headers.

**Impact**: The dynamic linker (`dyld`) cannot resolve `@rpath/` references to:
1. PyTorch libraries (`libc10.dylib`, `libtorch.dylib`, `libtorch_cpu.dylib`)
2. FFmpeg libraries (`libavutil.59.dylib`, `libavcodec.61.dylib`, etc.)

**Why This Happened**: TorchCodec pip package was likely built without proper RPATH configuration in its build system, expecting system-wide library installations or conda environments with automatic RPATH handling.

---

## Solution Implementation

### Strategy

Add RPATH entries to all TorchCodec dylibs using macOS `install_name_tool` utility:

1. **PyTorch RPATH**: `@loader_path/../../torch/lib` (relative path from torchcodec/ to torch/lib/)
2. **FFmpeg RPATH**: `/opt/homebrew/lib` (absolute path to Homebrew libraries)

### Execution Commands

#### Phase 1: Add PyTorch Library RPATH

```bash
# Core libraries (FFmpeg version-specific)
install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core4.dylib

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core5.dylib

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core6.dylib

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core8.dylib

# Custom ops libraries
install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops4.dylib

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops5.dylib

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops6.dylib

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops7.dylib

install_name_tool -add_rpath "@loader_path/../../torch/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops8.dylib
```

**Output**: Warning about code signature invalidation (expected, addressed in Phase 3)

#### Phase 2: Add FFmpeg Library RPATH

```bash
# Core libraries
install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core4.dylib

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core5.dylib

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core6.dylib

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core8.dylib

# Custom ops libraries
install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops4.dylib

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops5.dylib

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops6.dylib

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops7.dylib

install_name_tool -add_rpath "/opt/homebrew/lib" \
  venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops8.dylib
```

#### Phase 3: Re-sign Modified Libraries

Code signature invalidation warnings require re-signing with ad-hoc signature:

```bash
cd venv/lib/python3.11/site-packages/torchcodec

codesign --force --sign - libtorchcodec_core4.dylib
codesign --force --sign - libtorchcodec_core5.dylib
codesign --force --sign - libtorchcodec_core6.dylib
codesign --force --sign - libtorchcodec_core7.dylib
codesign --force --sign - libtorchcodec_core8.dylib
codesign --force --sign - libtorchcodec_custom_ops4.dylib
codesign --force --sign - libtorchcodec_custom_ops5.dylib
codesign --force --sign - libtorchcodec_custom_ops6.dylib
codesign --force --sign - libtorchcodec_custom_ops7.dylib
codesign --force --sign - libtorchcodec_custom_ops8.dylib
```

**Output**: "replacing existing signature" (successful re-signing)

---

## Verification

### Verify RPATH Addition

```bash
$ otool -l venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib | grep -A2 LC_RPATH
          cmd LC_RPATH
      cmdsize 48
         path @loader_path/../../torch/lib (offset 12)
--
          cmd LC_RPATH
      cmdsize 32
         path /opt/homebrew/lib (offset 12)
```

**Result**: ✅ Both RPATH entries successfully added

### Test Script Execution

```bash
$ source venv/bin/activate
$ python src/SpeechOceanFormatter.py

Saving the dataset (1/1 shards): 100%|██████████| 2500/2500 [00:00<00:00, 5041.45 examples/s]
Traceback (most recent call last):
  File "src/SpeechOceanFormatter.py", line 53, in <module>
    processed_dataset = raw_dataset.map(formatter.format_sample)
AttributeError: 'NoneType' object has no attribute 'map'
```

**Result**: ✅ **TorchCodec import successful** - script progressed past import stage

The new error is unrelated to TorchCodec (it's a script logic bug where `save_to_disk()` returns `None`).

---

## Alternative Solutions Considered

### Option 1: Environment Variable (Rejected - Temporary)

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:venv/lib/python3.11/site-packages/torch/lib:$DYLD_LIBRARY_PATH
```

**Pros**: No file modifications
**Cons**: Must be set in every session, fragile, not persistent
**Decision**: Rejected in favor of permanent fix

### Option 2: Reinstall via Conda (Not Pursued - Environment Change)

```bash
conda create -n speech python=3.11
conda install -c pytorch -c conda-forge torchcodec pytorch
```

**Pros**: Conda handles RPATH configuration automatically
**Cons**: Requires changing entire environment setup, may affect other dependencies
**Decision**: Not pursued to preserve existing pip-based environment

### Option 3: RPATH Fix (Selected - Permanent Solution)

**Pros**:
- Permanent fix
- No environment changes required
- Preserves existing pip installation
- Minimal invasiveness

**Cons**:
- Invalidates code signatures (mitigated by re-signing)
- Needs reapplication if TorchCodec is reinstalled

**Decision**: Selected as optimal solution

---

## Technical Background

### Understanding @rpath on macOS

**Definition**: `@rpath` is a placeholder in Mach-O dynamic library references that gets resolved at runtime using RPATH entries in the library's LC_RPATH load commands.

**Resolution Process**:
1. `dyld` encounters `@rpath/libc10.dylib` reference
2. Iterates through LC_RPATH entries in order
3. Substitutes each RPATH for `@rpath` and attempts to load
4. First successful match is used

**Example**:
```
Dependency: @rpath/libc10.dylib
RPATH[0]:   @loader_path/../../torch/lib
RPATH[1]:   /opt/homebrew/lib

Resolution attempt 1: @loader_path/../../torch/lib/libc10.dylib
  → Resolves to: venv/lib/python3.11/site-packages/torch/lib/libc10.dylib ✅
```

### @loader_path vs @rpath

- **@loader_path**: Expands to directory containing the Mach-O binary being loaded
- **@rpath**: Placeholder resolved using LC_RPATH entries
- **@executable_path**: Expands to directory containing the main executable

**Why @loader_path/../../torch/lib?**
```
Library location: venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib
@loader_path:     venv/lib/python3.11/site-packages/torchcodec/
../               venv/lib/python3.11/site-packages/
../../            venv/lib/python3.11/site-packages/
../../torch/lib   venv/lib/python3.11/site-packages/torch/lib/ ✅
```

---

## Automation Script

For future installations or other users encountering this issue:

```bash
#!/bin/bash
# fix_torchcodec_rpath.sh
# Fix TorchCodec RPATH configuration for pip-installed packages on macOS

set -e

VENV_PATH="${1:-venv}"
TORCHCODEC_DIR="$VENV_PATH/lib/python3.11/site-packages/torchcodec"
PYTORCH_RPATH="@loader_path/../../torch/lib"
FFMPEG_RPATH="/opt/homebrew/lib"

echo "Fixing TorchCodec RPATH configuration..."

# Check if torchcodec directory exists
if [ ! -d "$TORCHCODEC_DIR" ]; then
    echo "Error: TorchCodec directory not found at $TORCHCODEC_DIR"
    exit 1
fi

# Add RPATH entries to all dylibs
for dylib in "$TORCHCODEC_DIR"/*.dylib; do
    echo "Processing $(basename "$dylib")..."

    # Add PyTorch RPATH
    install_name_tool -add_rpath "$PYTORCH_RPATH" "$dylib" 2>/dev/null || true

    # Add FFmpeg RPATH
    install_name_tool -add_rpath "$FFMPEG_RPATH" "$dylib" 2>/dev/null || true

    # Re-sign library
    codesign --force --sign - "$dylib" 2>/dev/null
done

echo "✅ RPATH fix completed successfully!"
echo "Test with: source $VENV_PATH/bin/activate && python -c 'import torchcodec'"
```

**Usage**:
```bash
chmod +x fix_torchcodec_rpath.sh
./fix_torchcodec_rpath.sh venv
```

---

## Lessons Learned

### For Package Maintainers

1. **Always include RPATH entries** in distributed dylibs for pip packages
2. Use relative paths (`@loader_path`) for same-package dependencies
3. Document RPATH requirements in installation guides
4. Consider using `delocate` tool to bundle dependencies for pip packages

### For Users

1. **Investigate dylib dependencies** with `otool -L` when encountering loading errors
2. **Check RPATH configuration** with `otool -l | grep -A2 LC_RPATH`
3. **Verify dependency locations** before modifying RPATH
4. **Re-sign modified dylibs** to maintain system security compliance

### For This Project

1. Document this fix in project README for other users
2. Consider switching to conda-forge TorchCodec for production deployment
3. Add RPATH verification to CI/CD environment setup scripts
4. Pin TorchCodec version to prevent regression on updates

---

## Related Issues

- **TorchCodec GitHub**: https://github.com/pytorch/torchcodec/issues
- **Similar Issue Pattern**: pip packages with native dependencies on macOS often lack proper RPATH configuration
- **FFmpeg Compatibility**: TorchCodec supports FFmpeg versions 4-8, our Homebrew version 7.1.1 is compatible

---

## Appendix: Complete System State

### Environment Details
```
OS: macOS 14.5 (Darwin Kernel 23.5.0)
Architecture: arm64 (Apple Silicon)
Python: 3.11.6
PyTorch: 2.9.0
TorchCodec: (version from pip, compatible with PyTorch 2.9.x)
FFmpeg: 7.1.1 (Homebrew)
```

### Package Locations
```
Virtual env: /Users/xrickliao/WorkSpaces/.../venv
PyTorch libs: venv/lib/python3.11/site-packages/torch/lib/
TorchCodec libs: venv/lib/python3.11/site-packages/torchcodec/
FFmpeg libs: /opt/homebrew/lib/
```

### Files Modified
```
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core4.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core5.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core6.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core7.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_core8.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops4.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops5.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops6.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops7.dylib
venv/lib/python3.11/site-packages/torchcodec/libtorchcodec_custom_ops8.dylib
```

**Total files modified**: 10 dylib files

---

## Conclusion

The TorchCodec loading issue was successfully resolved by adding proper RPATH entries to all TorchCodec dynamic libraries. The fix is permanent and requires no changes to user code or environment variables. The solution demonstrates the importance of proper RPATH configuration in distributed native libraries for Python packages on macOS.

**Status**: ✅ **RESOLVED**
**Time to Resolution**: ~30 minutes of systematic diagnosis and implementation
**Persistence**: Permanent until TorchCodec package reinstallation
