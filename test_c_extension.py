#!/usr/bin/env python3
"""
Test script for the C linear regression extension.
"""

import sys
import os

# Add the current directory to path to find clinalg
sys.path.insert(0, '.')

try:
    import clinalg
    print("✅ Successfully imported clinalg module")
except ImportError as e:
    print(f"❌ Failed to import clinalg: {e}")
    print("Build the extension first with: python setup_c_extension.py build_ext --inplace")
    sys.exit(1)

def test_simple_regression():
    """Test simple linear regression with known data."""
    print("\n=== Testing Simple Linear Regression ===")
    
    # Test data: y ≈ 2x + 0
    X = [
        [1.0, 1.0],  # Intercept, x
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [1.0, 5.0]
    ]
    
    y = [2.1, 3.9, 6.1, 7.9, 10.1]
    
    try:
        result = clinalg.lm(X, y)
        print("✅ Linear regression completed successfully")
        
        coefficients = result.coefficients
        print(f"Coefficients: {coefficients}")
        print(f"  Intercept: {coefficients[0]:.6f}")
        print(f"  Slope: {coefficients[1]:.6f}")
        
        print(f"R-squared: {result.r_squared:.6f}")
        print(f"Residual Std Error: {result.residual_std_error:.6f}")
        
        fitted = result.fitted_values
        residuals = result.residuals
        print(f"Fitted values: {[f'{x:.3f}' for x in fitted]}")
        print(f"Residuals: {[f'{x:.3f}' for x in residuals]}")
        
        # Verify results are reasonable
        assert abs(coefficients[0] - 0.02) < 0.1, f"Intercept should be ~0.02, got {coefficients[0]}"
        assert abs(coefficients[1] - 2.0) < 0.1, f"Slope should be ~2.0, got {coefficients[1]}"
        assert result.r_squared > 0.99, f"R-squared should be >0.99, got {result.r_squared}"
        
        print("✅ Results validation passed")
        
    except Exception as e:
        print(f"❌ Linear regression failed: {e}")
        return False
    
    return True

def test_multiple_regression():
    """Test multiple linear regression."""
    print("\n=== Testing Multiple Linear Regression ===")
    
    # Test data: y ≈ 1 + 2*x1 + 0.5*x2
    # Make sure columns are linearly independent
    X = [
        [1.0, 2.0, 1.0],  # Intercept, x1, x2
        [1.0, 3.0, 2.0],  # Changed x2 values to avoid linear dependence
        [1.0, 4.0, 1.5],
        [1.0, 5.0, 3.0],
        [1.0, 6.0, 2.5],
        [1.0, 7.0, 4.0]
    ]
    
    y = [5.1, 7.2, 9.1, 11.3, 13.0, 15.2]
    
    try:
        result = clinalg.lm(X, y)
        print("✅ Multiple regression completed successfully")
        
        coefficients = result.coefficients
        print(f"Coefficients: {coefficients}")
        print(f"  Intercept: {coefficients[0]:.6f}")
        print(f"  X1: {coefficients[1]:.6f}")
        print(f"  X2: {coefficients[2]:.6f}")
        
        print(f"R-squared: {result.r_squared:.6f}")
        print(f"Residual Std Error: {result.residual_std_error:.6f}")
        
        # Verify R-squared is reasonable
        assert result.r_squared > 0.95, f"R-squared should be >0.95, got {result.r_squared}"
        
        print("✅ Multiple regression validation passed")
        
    except Exception as e:
        print(f"❌ Multiple regression failed: {e}")
        return False
    
    return True

def test_error_handling():
    """Test error handling."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test dimension mismatch
        X = [[1.0, 2.0], [1.0, 3.0]]
        y = [1.0, 2.0, 3.0]  # Wrong length
        
        try:
            result = clinalg.lm(X, y)
            print("❌ Should have raised error for dimension mismatch")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught dimension error: {e}")
        
        # Test singular matrix
        X = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]  # Rank deficient
        y = [1.0, 2.0, 3.0]
        
        try:
            result = clinalg.lm(X, y)
            print("❌ Should have raised error for singular matrix")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught singular matrix error: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error in error handling test: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("C Linear Regression Extension Test Suite")
    print("="*50)
    
    tests = [
        test_simple_regression,
        test_multiple_regression,
        test_error_handling
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            break
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())