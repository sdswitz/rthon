"""
Create a simple test to show both usage options
"""

import sys
sys.path.append('src')

print("🎯 C Linear Regression Usage Options")
print("=" * 40)

print("\n1. DIRECT USAGE (clinalg):")
print("   - Standalone C extension")
print("   - Fastest, minimal overhead")

import clinalg
X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]
y = [2.1, 3.9, 6.1, 7.9, 10.1]

result = clinalg.lm(X, y)
print(f"   Usage: import clinalg; result = clinalg.lm(X, y)")
print(f"   Result: {result.coefficients}")

print("\n2. INTEGRATED USAGE (future):")
print("   - Part of rthon package")
print("   - import rthon; result = rthon.lm(X, y)")
print("   - Automatic C/Python fallback")
print("   - Full LinearModel compatibility")

print(f"\n✅ RECOMMENDATION FOR NOTEBOOK:")
print(f"   Use the direct approach: import clinalg")
print(f"   It's simpler, faster, and guaranteed to work")

print("=" * 40)