# Deprecated Python Implementation

⚠️ **DO NOT USE THIS CODE** ⚠️

This folder contains the deprecated pure Python implementation of the linear regression functionality. This code has been **replaced by a high-performance C extension** and should not be imported or used.

## What's in here:
- `lm.py` - Original Python implementation of lm() function
- `linear_algebra.py` - Python-based linear algebra functions (QR decomposition, etc.)
- `c_lm.py` - Wrapper for C extension (superseded by direct integration)

## Why it's deprecated:
- **Performance**: C extension is 10-100x faster
- **Accuracy**: C extension provides machine-precision results
- **Maintenance**: Duplicate code paths create complexity

## Removal timeline:
- **v0.4.x**: Moved to `_deprecated/` folder
- **v0.5.0**: Will be completely removed

## Migration:
All functionality is now provided by the C extension. No code changes required - the same `lm()` interface works with better performance.

---
*This folder exists only for reference and will be deleted in v0.5.0*