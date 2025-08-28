# Regression module for rthon
# Implementation of R's lm() function and related statistical methods

from .lm import lm, summary_lm, predict_lm, residuals_lm, fitted_lm
from .linear_model import LinearModel
from .formula import Formula, parse_formula
from .linear_algebra import qr_decomposition, matrix_multiply, transpose
from .statistics import r_squared, adjusted_r_squared

__all__ = [
    # Main functions
    "lm", 
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