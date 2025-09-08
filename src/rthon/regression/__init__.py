# Regression module for rthon
# Implementation of R's lm() function and related statistical methods

from .linear_model import LinearModel
from .formula import Formula, parse_formula
from .linear_algebra import qr_decomposition, matrix_multiply, transpose
from .statistics import r_squared, adjusted_r_squared

# Try to import the fast C implementation first
try:
    from ._c_ext import lm as c_lm
    
    def lm_func(X, y=None, **kwargs):
        """
        High-performance linear regression using C implementation.
        
        Args:
            X: Design matrix (list of lists) or formula string
            y: Response vector (list) when X is matrix
            **kwargs: Additional arguments
        
        Returns:
            LinearModel compatible result
        """
        if isinstance(X, list) and y is not None and isinstance(y, list):
            # Matrix interface - use fast C implementation
            c_result = c_lm(X, y)
            
            # Generate column names
            n_params = len(X[0]) if X else 0
            column_names = [f"X{i+1}" for i in range(n_params)]
            if n_params > 0 and len(X) >= 1 and X[0][0] == 1.0:
                column_names[0] = "(Intercept)"
            
            # Create LinearModel object for compatibility
            model = LinearModel(
                coefficients=c_result.coefficients,
                residuals=c_result.residuals,
                fitted_values=c_result.fitted_values,
                y=y,
                X=X,
                column_names=column_names,
                formula=None,
                X_t_X_inv=None
            )
            
            # Override properties with C extension computed values
            model.r_squared = c_result.r_squared
            model.adj_r_squared = adjusted_r_squared(y, c_result.fitted_values, len(c_result.coefficients))
            model.sigma = c_result.residual_std_error
            
            return model
        else:
            # Handle formula interface by converting to matrix interface
            from .formula import parse_formula, design_matrix_from_formula, Formula
            
            if isinstance(X, (str, Formula)):
                # Formula interface
                if 'data' not in kwargs or kwargs['data'] is None:
                    raise ValueError("Data dictionary required when using formula interface")
                
                data = kwargs['data']
                if isinstance(data, dict):
                    # Parse formula and create design matrix
                    if isinstance(X, str):
                        formula_obj = parse_formula(X)
                    else:
                        formula_obj = X
                    
                    # Convert formula to matrix representation
                    X_matrix, y_vector, column_names = design_matrix_from_formula(formula_obj, data)
                    
                    # Use C implementation with converted data
                    c_result = c_lm(X_matrix, y_vector)
                    
                    # Create LinearModel object
                    model = LinearModel(
                        coefficients=c_result.coefficients,
                        residuals=c_result.residuals,
                        fitted_values=c_result.fitted_values,
                        y=y_vector,
                        X=X_matrix,
                        column_names=column_names,
                        formula=formula_obj,
                        X_t_X_inv=None
                    )
                    
                    # Override properties with C extension computed values
                    model.r_squared = c_result.r_squared
                    model.adj_r_squared = adjusted_r_squared(y_vector, c_result.fitted_values, len(c_result.coefficients))
                    model.sigma = c_result.residual_std_error
                    
                    return model
                else:
                    raise ValueError("Data must be a dictionary when using formulas")
            else:
                # Unsupported interface
                raise ValueError("Unsupported input format for C extension")
    
    # Import other functions from Python implementation
    from .lm import summary_lm, predict_lm, residuals_lm, fitted_lm
    
    # Assign the function to lm after imports to avoid conflicts
    lm = lm_func
    
    _using_c_implementation = True
    
    def lm_info():
        """Get information about the linear regression implementation."""
        return {
            "implementation": "C (high-performance)", 
            "module": "rthon.regression._c_ext",
            "available": True,
            "performance": "~10-100x faster than Python"
        }
    
except ImportError:
    # Fall back to Python implementation (DEPRECATED - will be removed in v0.5.0)
    from .lm import lm, summary_lm, predict_lm, residuals_lm, fitted_lm
    _using_c_implementation = False
    
    def lm_info():
        """Get information about the linear regression implementation."""
        return {
            "implementation": "Python (fallback - DEPRECATED)", 
            "module": "rthon.regression.lm",
            "available": False,
            "note": "C extension not available - using slower Python implementation",
            "deprecation": "Python implementation will be removed in v0.5.0"
        }

__all__ = [
    # Main functions
    "lm", 
    "lm_info",
    "summary_lm", 
    "predict_lm", 
    "residuals_lm", 
    "fitted_lm",
    
    # Classes  
    "LinearModel",
    "Formula",
    
    # Utilities
    "parse_formula",
    "qr_decomposition",
    "matrix_multiply", 
    "transpose",
    "r_squared",
    "adjusted_r_squared"
]