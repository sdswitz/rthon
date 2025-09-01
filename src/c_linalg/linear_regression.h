#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "matrix.h"
#include "qr_decomposition.h"

MatrixErrorCode compute_linear_regression(const Matrix *X, const Vector *y, LinearModelResult *result);

MatrixErrorCode compute_covariance_matrix(const QRDecomposition *qr, double mse, Matrix *cov_matrix);

MatrixErrorCode compute_statistics(const LinearModelResult *result, const Vector *y);

double compute_mean(const Vector *vec);
double compute_sum_of_squares(const Vector *vec);
double compute_total_sum_of_squares(const Vector *y, double y_mean);

MatrixErrorCode add_intercept_column(const Matrix *X, Matrix *X_with_intercept);

#endif