"""
Copy and paste this code into a Jupyter cell to diagnose and fix import issues
"""

# Run this in a Jupyter cell to diagnose the issue:

import sys
import os

print("🔍 Jupyter Environment Check")
print("=" * 40)
print(f"Jupyter Python: {sys.executable}")
print(f"Current directory: {os.getcwd()}")

# Check if we're in the right directory
expected_dir = "/Users/samswitz/GitHub/rthon"
if os.getcwd() != expected_dir:
    print(f"⚠️  Wrong directory! Should be in: {expected_dir}")
    print("Solution: Start Jupyter from the right directory:")
    print(f"cd {expected_dir} && jupyter notebook")
else:
    print("✅ Correct directory")

# Check for .so file
so_files = [f for f in os.listdir('.') if f.endswith('.so')]
print(f".so files in current directory: {so_files}")

# Add current directory to path if needed
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
    print("✅ Added current directory to sys.path")
else:
    print("✅ Current directory already in sys.path")

# Now try importing
try:
    import clinalg
    print("🎉 SUCCESS: clinalg imported!")
    print(f"Available functions: {dir(clinalg)}")
    
    # Quick test
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [2.1, 3.9, 6.1] 
    result = clinalg.lm(X, y)
    print(f"🎉 clinalg.lm() works! Coefficients: {result.coefficients}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\n🔧 TROUBLESHOOTING:")
    print("1. Make sure you started Jupyter from the right directory")
    print("2. Run this in a cell: !ls *.so")
    print("3. If no .so file, run: !python setup_c_extension.py build_ext --inplace")
    print("4. Check Python executable: !which python")
except Exception as e:
    print(f"❌ Function test failed: {e}")

print("\n" + "=" * 40)