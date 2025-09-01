#include "linear_regression.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double compute_mean(const Vector *vec) {
    if (!vector_is_valid(vec)) {
        return NAN;
    }
    
    double sum = 0.0;
    for (int i = 0; i < vec->length; i++) {
        sum += vec->data[i];
    }
    
    return sum / vec->length;
}

double compute_sum_of_squares(const Vector *vec) {
    if (!vector_is_valid(vec)) {
        return NAN;
    }
    
    double sum = 0.0;
    for (int i = 0; i < vec->length; i++) {
        sum += vec->data[i] * vec->data[i];
    }
    
    return sum;
}

double compute_total_sum_of_squares(const Vector *y, double y_mean) {
    if (!vector_is_valid(y)) {
        return NAN;
    }
    
    double tss = 0.0;
    for (int i = 0; i < y->length; i++) {
        double diff = y->data[i] - y_mean;
        tss += diff * diff;
    }
    
    return tss;
}

MatrixErrorCode add_intercept_column(const Matrix *X, Matrix *X_with_intercept) {
    if (!matrix_is_valid(X) || !matrix_is_valid(X_with_intercept)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    if (X_with_intercept->rows != X->rows || X_with_intercept->cols != X->cols + 1) {
        return MATRIX_ERROR_INVALID_DIMENSIONS;
    }
    
    for (int i = 0; i < X->rows; i++) {
        X_with_intercept->data[i][0] = 1.0;
        
        for (int j = 0; j < X->cols; j++) {
            X_with_intercept->data[i][j + 1] = X->data[i][j];
        }
    }
    
    return MATRIX_SUCCESS;
}

MatrixErrorCode compute_covariance_matrix(const QRDecomposition *qr, double mse, Matrix *cov_matrix) {
    if (!qr || !matrix_is_valid(cov_matrix)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    if (!matrix_is_valid(&qr->R)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    int p = cov_matrix->rows;
    if (cov_matrix->cols != p || qr->R.cols < p) {
        return MATRIX_ERROR_INVALID_DIMENSIONS;
    }
    
    Matrix *R_inv = matrix_alloc(p, p);
    if (!R_inv) {
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            R_inv->data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    for (int col = 0; col < p; col++) {
        Vector *b = vector_alloc(p);
        Vector *x = vector_alloc(p);
        
        if (!b || !x) {
            vector_free(b);
            vector_free(x);
            matrix_free(R_inv);
            return MATRIX_ERROR_MEMORY_ALLOCATION;
        }
        
        for (int i = 0; i < p; i++) {
            b->data[i] = (i == col) ? 1.0 : 0.0;
        }
        
        MatrixErrorCode result = back_substitution(&qr->R, b, x);
        if (result != MATRIX_SUCCESS) {
            vector_free(b);
            vector_free(x);
            matrix_free(R_inv);
            return result;
        }
        
        for (int i = 0; i < p; i++) {
            R_inv->data[i][col] = x->data[i];
        }
        
        vector_free(b);
        vector_free(x);
    }
    
    Matrix *R_inv_t = matrix_alloc(p, p);
    if (!R_inv_t) {
        matrix_free(R_inv);
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    matrix_transpose(R_inv, R_inv_t);
    
    Matrix *temp = matrix_alloc(p, p);
    if (!temp) {
        matrix_free(R_inv);
        matrix_free(R_inv_t);
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    matrix_multiply(R_inv, R_inv_t, temp);
    
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            cov_matrix->data[i][j] = mse * temp->data[i][j];
        }
    }
    
    matrix_free(R_inv);
    matrix_free(R_inv_t);
    matrix_free(temp);
    
    return MATRIX_SUCCESS;
}

MatrixErrorCode compute_statistics(const LinearModelResult *result, const Vector *y) {
    if (!result || !vector_is_valid(y)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    double y_mean = compute_mean(y);
    double tss = compute_total_sum_of_squares(y, y_mean);
    double ess = compute_sum_of_squares(&result->residuals);
    
    ((LinearModelResult*)result)->r_squared = 1.0 - (ess / tss);
    
    double mse = ess / result->df_residual;
    ((LinearModelResult*)result)->residual_std_error = sqrt(mse);
    
    for (int i = 0; i < result->coefficients.length; i++) {
        ((LinearModelResult*)result)->standard_errors[i] = 
            sqrt(result->covariance_matrix.data[i][i]);
    }
    
    double msr = (tss - ess) / result->df_model;
    ((LinearModelResult*)result)->f_statistic = msr / mse;
    
    return MATRIX_SUCCESS;
}

MatrixErrorCode compute_linear_regression(const Matrix *X, const Vector *y, LinearModelResult *result) {
    printf("compute_linear_regression: Starting function\n");
    if (!matrix_is_valid(X) || !vector_is_valid(y) || !result) {
        printf("compute_linear_regression: Null pointer error\n");
        return MATRIX_ERROR_NULL_POINTER;
    }
    printf("compute_linear_regression: Basic validation passed\n");
    
    if (X->rows != y->length) {
        printf("compute_linear_regression: Dimension mismatch X->rows=%d, y->length=%d\n", X->rows, y->length);
        return MATRIX_ERROR_INVALID_DIMENSIONS;
    }
    
    int n = X->rows;
    int p = X->cols;
    printf("compute_linear_regression: n=%d, p=%d\n", n, p);
    
    if (n <= p) {
        printf("compute_linear_regression: Not enough observations\n");
        return MATRIX_ERROR_INVALID_DIMENSIONS;
    }
    
    printf("compute_linear_regression: Computing matrix rank...\n");
    int rank = matrix_rank(X, 1e-12);
    printf("compute_linear_regression: Matrix rank = %d\n", rank);
    if (rank < p) {
        printf("compute_linear_regression: Matrix is rank deficient\n");
        return MATRIX_ERROR_RANK_DEFICIENT;
    }
    
    QRDecomposition *qr = qr_decomposition_alloc(n, p);
    if (!qr) {
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    MatrixErrorCode qr_result = qr_householder(X, qr);
    if (qr_result != MATRIX_SUCCESS) {
        qr_decomposition_free(qr);
        return qr_result;
    }
    
    Vector *coefficients = vector_alloc(p);
    if (!coefficients) {
        qr_decomposition_free(qr);
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    MatrixErrorCode solve_result = qr_solve(qr, y, coefficients);
    if (solve_result != MATRIX_SUCCESS) {
        vector_free(coefficients);
        qr_decomposition_free(qr);
        return solve_result;
    }
    
    for (int i = 0; i < p; i++) {
        result->coefficients.data[i] = coefficients->data[i];
    }
    
    MatrixErrorCode fitted_result = matrix_vector_multiply(X, &result->coefficients, &result->fitted_values);
    if (fitted_result != MATRIX_SUCCESS) {
        vector_free(coefficients);
        qr_decomposition_free(qr);
        return fitted_result;
    }
    
    MatrixErrorCode residual_result = vector_subtract(y, &result->fitted_values, &result->residuals);
    if (residual_result != MATRIX_SUCCESS) {
        vector_free(coefficients);
        qr_decomposition_free(qr);
        return residual_result;
    }
    
    double ess = compute_sum_of_squares(&result->residuals);
    double mse = ess / result->df_residual;
    
    MatrixErrorCode cov_result = compute_covariance_matrix(qr, mse, &result->covariance_matrix);
    if (cov_result != MATRIX_SUCCESS) {
        vector_free(coefficients);
        qr_decomposition_free(qr);
        return cov_result;
    }
    
    MatrixErrorCode stats_result = compute_statistics(result, y);
    
    vector_free(coefficients);
    qr_decomposition_free(qr);
    
    return stats_result;
}