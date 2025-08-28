import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rthon.regression.linear_algebra import qr_decomposition, solve_qr, back_substitution
from rthon.regression.formula import parse_formula, design_matrix_from_formula
from rthon import lm

def test_qr_simple():
    """Test QR decomposition with simple matrix."""
    print("Testing QR decomposition...")
    
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    print(f"X = {X}")
    
    try:
        Q, R = qr_decomposition(X)
        print(f"Q = {Q}")
        print(f"R = {R}")
        print("QR decomposition successful!")
        return Q, R
    except Exception as e:
        print(f"QR decomposition failed: {e}")
        return None, None

def test_formula_parsing():
    """Test formula parsing and design matrix generation."""
    print("\\nTesting formula parsing...")
    
    data = {'y': [1.0, 2.0, 3.0, 4.0, 5.0], 'x': [1.0, 2.0, 3.0, 4.0, 5.0]}
    formula_str = "y ~ x"
    
    try:
        formula = parse_formula(formula_str)
        print(f"Formula: {formula}")
        print(f"Response: {formula.response}")
        print(f"Terms: {[str(t) for t in formula.terms]}")
        
        X, y, column_names = design_matrix_from_formula(formula, data)
        print(f"X = {X}")
        print(f"y = {y}")
        print(f"column_names = {column_names}")
        
        return X, y, column_names
        
    except Exception as e:
        print(f"Formula parsing failed: {e}")
        return None, None, None

def test_back_substitution():
    """Test back substitution."""
    print("\\nTesting back substitution...")
    
    # Simple upper triangular system: R*x = b
    R = [[2.0, 1.0], [0.0, 1.0]]
    b = [3.0, 1.0]
    
    try:
        x = back_substitution(R, b)
        print(f"R = {R}")
        print(f"b = {b}")  
        print(f"x = {x}")
        print("Back substitution successful!")
        return x
    except Exception as e:
        print(f"Back substitution failed: {e}")
        return None

def test_lm_simple():
    """Test simple lm function."""
    print("\\nTesting lm function...")
    
    # Very simple data
    data = {
        'y': [1.0, 2.0, 3.0], 
        'x': [1.0, 2.0, 3.0]
    }
    
    try:
        print(f"Data: {data}")
        model = lm("y ~ x", data=data)
        print("lm function successful!")
        print(f"Coefficients: {dict(zip(model.column_names, model.coefficients))}")
        print(f"R-squared: {model.r_squared:.4f}")
        return model
    except Exception as e:
        print(f"lm function failed: {e}")
        return None

if __name__ == "__main__":
    test_qr_simple()
    test_formula_parsing()
    test_back_substitution()
    test_lm_simple()