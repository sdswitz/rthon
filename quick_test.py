#!/usr/bin/env python3
"""
Quick test to verify clinalg C extension works
"""

print("Testing clinalg C extension...")

# Import the extension
try:
    import clinalg
    print("✅ clinalg imported successfully")
except ImportError as e:
    print(f"❌ Failed to import clinalg: {e}")
    exit(1)

# Simple test
X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
y = [2.1, 3.9, 6.1, 7.9, 10.1]

print(f"Input X: {X}")
print(f"Input y: {y}")

try:
    result = clinalg.lm(X, y)
    print(f"✅ Regression successful!")
    print(f"Coefficients: {result.coefficients}")
    print(f"R-squared: {result.r_squared:.6f}")
    print(f"🎉 Ready to use in Jupyter notebook!")
    
except Exception as e:
    print(f"❌ Regression failed: {e}")
    exit(1)