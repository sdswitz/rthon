# Regression module for rthon
# High-performance C implementation of R's lm() function and related statistical methods

from .linear_model import LinearModel
from .formula import Formula, parse_formula, design_matrix_from_formula
from .statistics import r_squared, adjusted_r_squared

# Import C extension (mandatory - no fallback)
try:
    from ._c_ext import lm as c_lm
except ImportError as e:
    raise ImportError(
        "C extension not available. rthon requires the C extension for linear regression. "
        "Please ensure the package was installed correctly with C compilation support. "
        f"Original error: {e}"
    ) from e

def lm(X, y=None, **kwargs):
    """
    High-performance linear regression using C implementation.
    
    Args:
        X: Design matrix (list of lists) or formula string
        y: Response vector (list) when X is matrix
        data: Data dictionary when X is formula string
        
    Returns:
        LinearModel object with fitted regression results
        
    Examples:
        # Matrix interface
        X = [[1, 1], [1, 2], [1, 3]]
        y = [1, 2, 3]
        model = lm(X, y=y)
        
        # Formula interface  
        data = {'y': [1, 2, 3], 'x': [1, 2, 3]}
        model = lm('y ~ x', data=data)
    """
    
    if isinstance(X, list) and y is not None and isinstance(y, list):
        # Matrix interface - use C implementation directly
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
        
    elif isinstance(X, (str, Formula)):
        # Formula interface - convert to matrix and use C implementation
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
        raise ValueError("Unsupported input format. Use matrix interface lm(X, y=y) or formula interface lm('y ~ x', data=data)")

# Convenience functions (simple wrappers around LinearModel methods)
def summary_lm(model: LinearModel) -> str:
    """Print summary of linear model."""
    return model.summary()

def predict_lm(model: LinearModel, newdata=None, interval="none", level=0.95):
    """Predict using linear model."""
    return model.predict(newdata, interval, level)

def residuals_lm(model: LinearModel):
    """Extract residuals from linear model."""  
    return model.residuals_method()

def fitted_lm(model: LinearModel):
    """Extract fitted values from linear model."""
    return model.fitted()

def lm_info():
    """Get information about the linear regression implementation."""
    return {
        "implementation": "C (high-performance)", 
        "module": "rthon.regression._c_ext",
        "available": True,
        "performance": "~10-100x faster than deprecated Python implementation",
        "version": "C extension only - no fallback"
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
    "r_squared",
    "adjusted_r_squared"
]