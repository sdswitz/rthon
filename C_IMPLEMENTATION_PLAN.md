# C Implementation Plan for lm() Function

## Overview
This document outlines the plan to implement a high-performance C-based least squares regression function to replace the current Python implementation, which produces incorrect results.

## Current State Analysis
- Python implementation exists but is fundamentally flawed and produces bad results
- Uses pure Python linear algebra with QR decomposition approach
- Supports formula interface (`"y ~ x1 + x2"`) and matrix interface
- Need to implement correct algorithm from scratch in C

## Implementation Plan

### 1. C API Design
Design C functions that provide:
- Matrix-based least squares regression
- Formula parsing support (future enhancement)
- Structured results matching Python interface expectations

### 2. Core Data Structures
```c
typedef struct {
    double **data;
    int rows, cols;
} Matrix;

typedef struct {
    double *data;
    int length;
} Vector;

typedef struct {
    Vector coefficients;
    Vector residuals;
    Vector fitted_values;
    double r_squared;
    double residual_std_error;
    double *standard_errors;
    Matrix covariance_matrix;
    int df_residual;
} LinearModelResult;
```

### 3. Linear Algebra Foundation
Core functions needed:
- `matrix_alloc()`, `matrix_free()`, `vector_alloc()`, `vector_free()`
- `matrix_multiply()`, `matrix_transpose()`, `matrix_copy()`
- `vector_dot()`, `vector_norm()`, `vector_subtract()`, `vector_add()`
- Memory-safe implementations with comprehensive error checking

### 4. QR Decomposition Implementation
- Implement Householder reflections algorithm for numerical stability
- Handle over-determined systems (more observations than parameters)
- Graceful handling of rank-deficient matrices
- Pivoting support for improved numerical accuracy

### 5. Least Squares Solver
Core algorithm:
1. QR decompose design matrix X
2. Solve R*β = Q^T*y for coefficients β
3. Calculate fitted values: ŷ = X*β
4. Calculate residuals: e = y - ŷ
5. Compute covariance matrix: (X^T*X)^(-1) = R^(-1)*R^(-T)

### 6. Statistical Calculations
Implement correct formulas:
- **R-squared**: R² = 1 - (SSE/TSS) = 1 - Σ(y_i - ŷ_i)²/Σ(y_i - ȳ)²
- **Residual standard error**: σ̂ = √(SSE/(n-p))
- **Standard errors**: SE(β_j) = σ̂ * √(C_jj) where C = (X^T*X)^(-1)
- **t-statistics**: t_j = β_j / SE(β_j)
- **F-statistic**: F = (MSR/MSE) for overall model significance

### 7. Python Integration
- Python C extension module using CPython API
- Convert Python lists/numpy arrays to C arrays efficiently
- Return results as Python objects (lists, dicts, or custom classes)
- Comprehensive error handling with informative Python exceptions
- Memory management ensuring no leaks

### 8. Build System & Testing
- Cross-platform build system (setup.py with C extension support)
- Unit tests against known correct results (R, scipy, statsmodels)
- Performance benchmarks
- Memory safety validation (valgrind, etc.)

## Key Design Principles

### Numerical Accuracy
- Use double precision throughout
- Implement numerically stable algorithms
- Handle edge cases (perfect collinearity, near-singular matrices)
- Proper scaling and conditioning checks

### Performance Optimization
- Efficient memory layout (row-major vs column-major)
- Cache-friendly algorithms
- Minimize memory allocations in hot paths
- BLAS-style optimizations where applicable

### API Compatibility
- Maintain exact interface with current Python `lm()` function
- Support same input formats (formula strings, matrices)
- Return same result structure for drop-in replacement

## Implementation Priority
1. **Core data structures and memory management**
2. **Basic linear algebra operations**
3. **QR decomposition with Householder reflections**
4. **Least squares solver**
5. **Statistical calculations**
6. **Python wrapper and integration**
7. **Comprehensive testing and validation**
8. **Build system and packaging**

## Success Criteria
- Produces numerically correct results matching R's `lm()` function
- Significant performance improvement over pure Python
- Zero memory leaks under normal operation
- Handles edge cases gracefully
- Maintains full API compatibility
- Comprehensive test coverage

## Notes
- Current Python implementation is incorrect and should not be used as reference for correctness
- Focus on implementing mathematically sound algorithms from first principles
- Validate against established statistical software (R, SAS, etc.)