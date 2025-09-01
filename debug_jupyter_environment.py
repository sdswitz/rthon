#!/usr/bin/env python3
"""
Comprehensive debugging script for Jupyter environment issues with clinalg
"""

import sys
import os
import subprocess

print("🔍 Jupyter Environment Debugging")
print("=" * 50)

# 1. Check current Python executable
print("1. PYTHON EXECUTABLE:")
print(f"   Current Python: {sys.executable}")
print(f"   Python version: {sys.version}")

# 2. Check working directory
print(f"\n2. WORKING DIRECTORY:")
print(f"   Current directory: {os.getcwd()}")

# 3. Check if .so file exists
print(f"\n3. C EXTENSION FILES:")
so_files = [f for f in os.listdir('.') if f.endswith('.so')]
print(f"   .so files found: {so_files}")

for so_file in so_files:
    print(f"   {so_file}: {os.path.getsize(so_file)} bytes")

# 4. Check Python path
print(f"\n4. PYTHON PATH (first 10 entries):")
for i, path in enumerate(sys.path[:10]):
    print(f"   {i}: {path}")

# 5. Try importing clinalg
print(f"\n5. CLINALG IMPORT TEST:")
try:
    import clinalg
    print("   ✅ clinalg imported successfully")
    print(f"   Module file: {clinalg.__file__}")
    print(f"   Available functions: {[name for name in dir(clinalg) if not name.startswith('_')]}")
    
    # Quick function test
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [2.1, 3.9, 6.1]
    result = clinalg.lm(X, y)
    print(f"   ✅ clinalg.lm() works: coefficients = {result.coefficients}")
    
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    
except Exception as e:
    print(f"   ❌ Function call failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# 6. Check Jupyter kernel
print(f"\n6. JUPYTER KERNEL INFO:")
try:
    import ipykernel
    print("   ✅ ipykernel available")
    print(f"   ipykernel version: {ipykernel.__version__}")
except ImportError:
    print("   ❌ ipykernel not available")

try:
    result = subprocess.run(['jupyter', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("   ✅ Jupyter available:")
        for line in result.stdout.strip().split('\n'):
            print(f"      {line}")
    else:
        print("   ❌ Jupyter command failed")
except Exception as e:
    print(f"   ❌ Could not check Jupyter version: {e}")

# 7. Check if we're in a virtual environment
print(f"\n7. VIRTUAL ENVIRONMENT:")
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("   ✅ Running in virtual environment")
    print(f"   Prefix: {sys.prefix}")
    if hasattr(sys, 'base_prefix'):
        print(f"   Base prefix: {sys.base_prefix}")
else:
    print("   ⚠️  Not in virtual environment (using system Python)")

# 8. Python module search
print(f"\n8. MODULE SEARCH TEST:")
import importlib.util
spec = importlib.util.find_spec("clinalg")
if spec:
    print(f"   ✅ clinalg found at: {spec.origin}")
else:
    print("   ❌ clinalg not found in module search path")

print(f"\n" + "=" * 50)
print("🎯 RECOMMENDATIONS:")
print("If clinalg import failed:")
print("1. Make sure Jupyter is started from this directory:")
print(f"   cd {os.getcwd()} && jupyter notebook")
print("2. In Jupyter, add this directory to sys.path:")
print(f"   import sys; sys.path.append('{os.getcwd()}')")
print("3. Check if Jupyter uses the same Python as command line:")
print(f"   !which python  (should show: {sys.executable})")
print("4. Try rebuilding the extension:")
print("   !python setup_c_extension.py build_ext --inplace")