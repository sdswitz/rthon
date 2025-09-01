#!/usr/bin/env python3
"""
Debug script for Jupyter import issues
"""

import sys
import os

print("🔍 Python Environment Debug Info")
print("=" * 40)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path:")
for p in sys.path[:5]:  # Show first 5 paths
    print(f"  {p}")

print("\n🔍 File System Check")
print("=" * 40)
so_files = [f for f in os.listdir('.') if f.endswith('.so')]
print(f"*.so files in current directory: {so_files}")

print("\n🔍 Import Test")
print("=" * 40)
try:
    import clinalg
    print("✅ clinalg imported successfully")
    print(f"Available functions: {[name for name in dir(clinalg) if not name.startswith('_')]}")
    
    # Quick test
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [2.1, 3.9, 6.1]
    result = clinalg.lm(X, y)
    print(f"✅ clinalg.lm() works: coefficients = {result.coefficients}")
    
except ImportError as e:
    print(f"❌ Failed to import clinalg: {e}")
except Exception as e:
    print(f"❌ Error using clinalg: {e}")

print(f"\n🎯 To fix Jupyter import issues:")
print(f"1. Make sure Jupyter is started from: {os.getcwd()}")
print(f"2. In Jupyter, run: import sys; sys.path.append('{os.getcwd()}')")
print(f"3. Then: import clinalg")