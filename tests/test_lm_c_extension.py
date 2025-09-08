"""
Comprehensive unit tests for C extension implementation of lm() function.
Tests against prostate cancer dataset with known R reference outputs.
"""

import sys
import os
import pytest
from math import isclose

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rthon import lm, lm_info
from prostate_data_loader import (
    load_prostate_data, 
    load_reference_coefficients, 
    load_reference_model_summary
)

class TestLmCExtension:
    """Test suite for C extension implementation of lm() function."""
    
    @classmethod
    def setup_class(cls):
        """Set up test data once for all tests."""
        cls.X, cls.y, cls.column_names, cls.data_dict = load_prostate_data()
        cls.ref_coeffs = load_reference_coefficients()
        cls.ref_summary = load_reference_model_summary()
        
        # Verify we have the expected data structure
        assert len(cls.X) == 97, f"Expected 97 observations, got {len(cls.X)}"
        assert len(cls.X[0]) == 9, f"Expected 9 variables, got {len(cls.X[0])}"
        assert len(cls.y) == 97, f"Expected 97 response values, got {len(cls.y)}"
        
        print(f"Test data loaded: {len(cls.X)}x{len(cls.X[0])} design matrix")
        print(f"Implementation info: {lm_info()}")
    
    def test_c_extension_available(self):
        """Verify C extension is available and being used."""
        info = lm_info()
        assert info['available'] == True, "C extension should be available"
        assert 'C' in info['implementation'], "Should be using C implementation"
    
    def test_model_fitting_basic(self):
        """Test that C extension can fit the prostate model without errors."""
        model = lm(self.X, y=self.y)
        
        # Basic structural checks
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'coefficients'), "Model should have coefficients"
        assert hasattr(model, 'r_squared'), "Model should have r_squared"
        assert hasattr(model, 'residuals'), "Model should have residuals"
        assert hasattr(model, 'fitted_values'), "Model should have fitted_values"
        
        # Correct dimensions
        assert len(model.coefficients) == 9, f"Expected 9 coefficients, got {len(model.coefficients)}"
        assert len(model.residuals) == 97, f"Expected 97 residuals, got {len(model.residuals)}"
        assert len(model.fitted_values) == 97, f"Expected 97 fitted values, got {len(model.fitted_values)}"
    
    def test_coefficient_accuracy(self):
        """Test coefficient estimates against R reference values."""
        model = lm(self.X, y=self.y)
        
        # Test each coefficient against reference
        for i, col_name in enumerate(self.column_names):
            actual = model.coefficients[i]
            expected = self.ref_coeffs[col_name]['estimate']
            
            # Use high precision for coefficient matching
            assert isclose(actual, expected, abs_tol=1e-10), \
                f"Coefficient {col_name}: expected {expected:.12f}, got {actual:.12f}"
    
    def test_r_squared_accuracy(self):
        """Test R-squared against reference value."""
        model = lm(self.X, y=self.y)
        
        expected_r2 = self.ref_summary['r_squared']
        actual_r2 = model.r_squared
        
        assert isclose(actual_r2, expected_r2, abs_tol=1e-8), \
            f"R-squared: expected {expected_r2:.10f}, got {actual_r2:.10f}"
    
    def test_adjusted_r_squared_accuracy(self):
        """Test adjusted R-squared against reference value."""
        model = lm(self.X, y=self.y)
        
        expected_adj_r2 = self.ref_summary['adj_r_squared']
        actual_adj_r2 = model.adj_r_squared
        
        assert isclose(actual_adj_r2, expected_adj_r2, abs_tol=1e-8), \
            f"Adj. R-squared: expected {expected_adj_r2:.10f}, got {actual_adj_r2:.10f}"
    
    def test_residual_standard_error(self):
        """Test residual standard error (sigma) against reference value."""
        model = lm(self.X, y=self.y)
        
        expected_sigma = self.ref_summary['sigma']
        actual_sigma = model.sigma
        
        assert isclose(actual_sigma, expected_sigma, abs_tol=1e-8), \
            f"Sigma: expected {expected_sigma:.10f}, got {actual_sigma:.10f}"
    
    def test_residuals_properties(self):
        """Test mathematical properties of residuals."""
        model = lm(self.X, y=self.y)
        
        # Residuals should sum to approximately zero (with intercept)
        residual_sum = sum(model.residuals)
        assert abs(residual_sum) < 1e-10, \
            f"Residuals should sum to ~0, got {residual_sum:.12f}"
        
        # Fitted + residuals should equal observed
        for i in range(len(self.y)):
            predicted = model.fitted_values[i] + model.residuals[i]
            actual = self.y[i]
            assert isclose(predicted, actual, abs_tol=1e-12), \
                f"Row {i}: fitted + residual != observed ({predicted:.12f} != {actual:.12f})"
    
    def test_numerical_stability(self):
        """Test numerical stability of C implementation."""
        model = lm(self.X, y=self.y)
        
        # No NaN or inf values
        for coef in model.coefficients:
            assert not (coef != coef), "Coefficient should not be NaN"  # NaN != NaN
            assert abs(coef) < float('inf'), "Coefficient should not be infinite"
        
        # R-squared in valid range
        assert 0 <= model.r_squared <= 1, f"R-squared should be in [0,1], got {model.r_squared}"
        
        # Sigma should be positive
        assert model.sigma > 0, f"Sigma should be positive, got {model.sigma}"
    
    def test_standard_errors(self):
        """Test standard error calculations if available."""
        model = lm(self.X, y=self.y)
        
        # If model has standard errors, test them
        if hasattr(model, 'standard_errors') and model.standard_errors:
            for i, col_name in enumerate(self.column_names):
                actual_se = model.standard_errors[i]
                expected_se = self.ref_coeffs[col_name]['std_error']
                
                assert isclose(actual_se, expected_se, abs_tol=1e-8), \
                    f"SE {col_name}: expected {expected_se:.10f}, got {actual_se:.10f}"
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        
        # Empty data
        with pytest.raises((ValueError, TypeError)):
            lm([], [])
        
        # Mismatched dimensions
        with pytest.raises((ValueError, TypeError)):
            lm([[1, 2]], [1, 2])  # 1 row, 2 response values
        
        # Single observation (should fail - can't estimate 9 parameters)
        with pytest.raises((ValueError, TypeError)):
            lm([[1, 1, 1, 1, 1, 1, 1, 1, 1]], [1])
    
    def test_performance_benchmark(self):
        """Basic performance test - should complete reasonably quickly."""
        import time
        
        start_time = time.time()
        model = lm(self.X, y=self.y)
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Should complete in under 1 second for this dataset size
        assert elapsed < 1.0, f"C extension took too long: {elapsed:.3f}s"
        
        print(f"C extension fitting time: {elapsed*1000:.2f}ms")

def test_integration_example():
    """Integration test showing typical usage."""
    # Load data
    X, y, column_names, _ = load_prostate_data()
    
    # Fit model
    model = lm(X, y=y)
    
    # Display results
    print(f"\nProstate Cancer Regression Results (C Extension)")
    print(f"{'Variable':<12} {'Coefficient':<12}")
    print("-" * 25)
    for i, name in enumerate(column_names):
        coef = model.coefficients[i]
        print(f"{name:<12} {coef:>11.6f}")
    
    print(f"\nModel Statistics:")
    print(f"R-squared:     {model.r_squared:.6f}")
    print(f"Adj R-squared: {model.adj_r_squared:.6f}")
    print(f"Sigma:         {model.sigma:.6f}")
    print(f"Observations:  {len(y)}")
    
    # Basic validation
    assert model.r_squared > 0.6, "Should have reasonable predictive power"
    assert len(model.coefficients) == 9, "Should have 9 coefficients"

if __name__ == "__main__":
    # Run individual test for development
    pytest.main([__file__ + "::TestLmCExtension::test_coefficient_accuracy", "-v"])