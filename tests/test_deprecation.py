"""
Test deprecation warning for Python implementation.
"""

import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_deprecation_warning():
    """Test that Python fallback issues deprecation warning."""
    
    # Temporarily force Python fallback by making C import fail
    import sys
    original_modules = sys.modules.copy()
    
    try:
        # Remove any cached C extension
        if 'rthon.regression._c_ext' in sys.modules:
            del sys.modules['rthon.regression._c_ext']
        if 'rthon.regression' in sys.modules:
            del sys.modules['rthon.regression']
        if 'rthon' in sys.modules:
            del sys.modules['rthon']
        
        # Mock the C extension to not be available
        sys.modules['rthon.regression._c_ext'] = None
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure all warnings are captured
            
            # This should fall back to Python and issue warning
            from rthon.regression.lm import lm
            
            # Simple test data
            X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
            y = [1.0, 2.0, 3.0]
            
            # This should trigger the deprecation warning
            model = lm(X, y=y)
            
            # Check if deprecation warning was issued
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            
            if deprecation_warnings:
                print("✅ Deprecation warning issued:")
                for warning in deprecation_warnings:
                    print(f"   {warning.message}")
            else:
                print("❌ No deprecation warning found")
                print(f"All warnings: {[str(warning.message) for warning in w]}")
            
            print(f"Model fitted successfully with coefficients: {model.coefficients}")
    
    finally:
        # Restore original modules state
        sys.modules.clear()
        sys.modules.update(original_modules)

if __name__ == "__main__":
    test_deprecation_warning()