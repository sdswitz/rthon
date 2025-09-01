#!/usr/bin/env python3
"""
Test for miniforge3 environment - run this to verify clinalg works
"""

import sys
import os

print("🧪 Testing clinalg with miniforge3")
print("=" * 40)
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"Directory: {os.getcwd()}")

# Check for correct .so file
so_files = [f for f in os.listdir('.') if f.endswith('.so')]
print(f"Available .so files: {so_files}")

# Should see clinalg.cpython-310-darwin.so for miniforge3
expected_file = "clinalg.cpython-310-darwin.so"
if expected_file in so_files:
    print(f"✅ Found correct extension: {expected_file}")
else:
    print(f"❌ Missing {expected_file}")
    print("Run: /Users/samswitz/miniforge3/bin/python setup_c_extension.py build_ext --inplace")

# Test import and functionality
try:
    import clinalg
    print("✅ clinalg imported successfully")
    
    # Test the lm function
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
    y = [2.1, 3.9, 6.1, 7.9, 10.1]
    
    result = clinalg.lm(X, y)
    print("✅ clinalg.lm() works!")
    print(f"Coefficients: {result.coefficients}")
    print(f"R-squared: {result.r_squared:.6f}")
    print(f"Residual Std Error: {result.residual_std_error:.6f}")
    
    print("\n🎉 SUCCESS: Ready to use in Jupyter!")
    print("Make sure Jupyter is using miniforge3 Python:")
    print("In Jupyter cell, run: !which python")
    print("Should show: /Users/samswitz/miniforge3/bin/python")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
print("=" * 40)