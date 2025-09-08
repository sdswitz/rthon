"""
Simple test runner for C extension without pytest dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rthon import lm, lm_info
from prostate_data_loader import (
    load_prostate_data, 
    load_reference_coefficients, 
    load_reference_model_summary
)

def test_basic_functionality():
    """Test basic C extension functionality."""
    print("Testing C extension basic functionality...")
    
    # Load data
    X, y, column_names, data_dict = load_prostate_data()
    ref_coeffs = load_reference_coefficients()
    ref_summary = load_reference_model_summary()
    
    print(f"Data loaded: {len(X)}x{len(X[0])} matrix, {len(y)} responses")
    print(f"Implementation info: {lm_info()}")
    
    # Try to fit model
    try:
        model = lm(X, y=y)
        print("✅ Model fitting successful!")
        
        # Basic checks
        print(f"Coefficients: {len(model.coefficients)} parameters")
        print(f"R-squared: {model.r_squared:.6f} (expected: {ref_summary['r_squared']:.6f})")
        print(f"Residual SE: {model.sigma:.6f} (expected: {ref_summary['sigma']:.6f})")
        
        # Test coefficient accuracy
        print("\nCoefficient comparison:")
        print(f"{'Variable':<12} {'Actual':<12} {'Expected':<12} {'Diff':<12}")
        print("-" * 50)
        
        max_error = 0
        for i, col_name in enumerate(column_names):
            actual = model.coefficients[i]
            expected = ref_coeffs[col_name]['estimate']
            diff = abs(actual - expected)
            max_error = max(max_error, diff)
            
            print(f"{col_name:<12} {actual:>11.6f} {expected:>11.6f} {diff:>11.2e}")
        
        print(f"\nMaximum coefficient error: {max_error:.2e}")
        
        # Test model statistics
        print(f"\nModel statistics comparison:")
        print(f"R-squared error: {abs(model.r_squared - ref_summary['r_squared']):.2e}")
        print(f"Sigma error: {abs(model.sigma - ref_summary['sigma']):.2e}")
        
        if hasattr(model, 'adj_r_squared'):
            print(f"Adj R-squared error: {abs(model.adj_r_squared - ref_summary['adj_r_squared']):.2e}")
        
        # Check if errors are within reasonable bounds
        if max_error < 1e-6:
            print("✅ Coefficients are highly accurate!")
        elif max_error < 1e-3:
            print("⚠️  Coefficients have some errors but are reasonable")
        else:
            print("❌ Coefficients have significant errors")
        
        return model
        
    except Exception as e:
        print(f"❌ Model fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_basic_functionality()