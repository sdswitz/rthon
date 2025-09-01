#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

typedef struct {
    double *coefficients;
    double *fitted_values;
    double *residuals;
    double r_squared;
    double residual_std_error;
    int n_obs;
    int n_params;
} SimpleLMResult;

SimpleLMResult* simple_lm_result_alloc(int n_obs, int n_params) {
    SimpleLMResult *result = malloc(sizeof(SimpleLMResult));
    if (!result) return NULL;
    
    result->coefficients = malloc(n_params * sizeof(double));
    result->fitted_values = malloc(n_obs * sizeof(double));
    result->residuals = malloc(n_obs * sizeof(double));
    
    if (!result->coefficients || !result->fitted_values || !result->residuals) {
        free(result->coefficients);
        free(result->fitted_values);
        free(result->residuals);
        free(result);
        return NULL;
    }
    
    result->n_obs = n_obs;
    result->n_params = n_params;
    result->r_squared = 0.0;
    result->residual_std_error = 0.0;
    
    return result;
}

void simple_lm_result_free(SimpleLMResult *result) {
    if (!result) return;
    
    free(result->coefficients);
    free(result->fitted_values);
    free(result->residuals);
    free(result);
}

void print_matrix_debug(const Matrix *mat, const char *name) {
    printf("%s (%dx%d):\n", name, mat->rows, mat->cols);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%8.4f ", mat->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector_debug(const Vector *vec, const char *name) {
    printf("%s (length %d): ", name, vec->length);
    for (int i = 0; i < vec->length; i++) {
        printf("%8.4f ", vec->data[i]);
    }
    printf("\n\n");
}

MatrixErrorCode gaussian_elimination(Matrix *A, Vector *b, Vector *x) {
    int n = A->rows;
    
    if (A->cols != n || b->length != n || x->length != n) {
        return MATRIX_ERROR_INVALID_DIMENSIONS;
    }
    
    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(A->data[k][i]) > fabs(A->data[max_row][i])) {
                max_row = k;
            }
        }
        
        if (max_row != i) {
            for (int j = 0; j < n; j++) {
                double temp = A->data[i][j];
                A->data[i][j] = A->data[max_row][j];
                A->data[max_row][j] = temp;
            }
            double temp = b->data[i];
            b->data[i] = b->data[max_row];
            b->data[max_row] = temp;
        }
        
        if (fabs(A->data[i][i]) < 1e-12) {
            return MATRIX_ERROR_SINGULAR_MATRIX;
        }
        
        for (int k = i + 1; k < n; k++) {
            double factor = A->data[k][i] / A->data[i][i];
            for (int j = i; j < n; j++) {
                A->data[k][j] -= factor * A->data[i][j];
            }
            b->data[k] -= factor * b->data[i];
        }
    }
    
    for (int i = n - 1; i >= 0; i--) {
        x->data[i] = b->data[i];
        for (int j = i + 1; j < n; j++) {
            x->data[i] -= A->data[i][j] * x->data[j];
        }
        x->data[i] /= A->data[i][i];
    }
    
    return MATRIX_SUCCESS;
}

MatrixErrorCode simple_linear_regression(const Matrix *X, const Vector *y, SimpleLMResult *result) {
    printf("simple_linear_regression: Starting\n");
    
    if (!matrix_is_valid(X) || !vector_is_valid(y) || !result) {
        printf("simple_linear_regression: Invalid inputs\n");
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    int n = X->rows;
    int p = X->cols;
    
    printf("simple_linear_regression: n=%d, p=%d\n", n, p);
    
    if (n != y->length) {
        printf("simple_linear_regression: Dimension mismatch\n");
        return MATRIX_ERROR_INVALID_DIMENSIONS;
    }
    
    Matrix *X_t = matrix_alloc(p, n);
    Matrix *X_t_X = matrix_alloc(p, p);
    Vector *X_t_y = vector_alloc(p);
    Vector *coefficients = vector_alloc(p);
    
    if (!X_t || !X_t_X || !X_t_y || !coefficients) {
        printf("simple_linear_regression: Memory allocation failed\n");
        matrix_free(X_t);
        matrix_free(X_t_X);
        vector_free(X_t_y);
        vector_free(coefficients);
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    matrix_transpose(X, X_t);
    print_matrix_debug(X_t, "X_t");
    
    matrix_multiply(X_t, X, X_t_X);
    print_matrix_debug(X_t_X, "X_t_X");
    
    matrix_vector_multiply(X_t, y, X_t_y);
    print_vector_debug(X_t_y, "X_t_y");
    
    MatrixErrorCode solve_result = gaussian_elimination(X_t_X, X_t_y, coefficients);
    
    if (solve_result != MATRIX_SUCCESS) {
        printf("simple_linear_regression: Gaussian elimination failed\n");
        matrix_free(X_t);
        matrix_free(X_t_X);
        vector_free(X_t_y);
        vector_free(coefficients);
        return solve_result;
    }
    
    print_vector_debug(coefficients, "coefficients");
    
    for (int i = 0; i < p; i++) {
        result->coefficients[i] = coefficients->data[i];
    }
    
    for (int i = 0; i < n; i++) {
        result->fitted_values[i] = 0.0;
        for (int j = 0; j < p; j++) {
            result->fitted_values[i] += X->data[i][j] * result->coefficients[j];
        }
        result->residuals[i] = y->data[i] - result->fitted_values[i];
    }
    
    double sum_y = 0.0;
    for (int i = 0; i < n; i++) {
        sum_y += y->data[i];
    }
    double mean_y = sum_y / n;
    
    double tss = 0.0;
    double rss = 0.0;
    for (int i = 0; i < n; i++) {
        tss += (y->data[i] - mean_y) * (y->data[i] - mean_y);
        rss += result->residuals[i] * result->residuals[i];
    }
    
    result->r_squared = 1.0 - (rss / tss);
    result->residual_std_error = sqrt(rss / (n - p));
    
    matrix_free(X_t);
    matrix_free(X_t_X);
    vector_free(X_t_y);
    vector_free(coefficients);
    
    printf("simple_linear_regression: Success\n");
    return MATRIX_SUCCESS;
}

void test_simple_lm(void) {
    printf("=== Testing Simple Linear Model (Normal Equations) ===\n");
    
    double X_data[5][2] = {
        {1.0, 1.0},
        {1.0, 2.0},
        {1.0, 3.0},
        {1.0, 4.0},
        {1.0, 5.0}
    };
    
    double y_data[5] = {2.1, 3.9, 6.1, 7.9, 10.1};
    
    Matrix *X = matrix_alloc(5, 2);
    Vector *y = vector_alloc(5);
    
    if (!X || !y) {
        printf("Memory allocation failed\n");
        return;
    }
    
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            X->data[i][j] = X_data[i][j];
        }
        y->data[i] = y_data[i];
    }
    
    print_matrix_debug(X, "X");
    print_vector_debug(y, "y");
    
    SimpleLMResult *result = simple_lm_result_alloc(5, 2);
    if (!result) {
        printf("Result allocation failed\n");
        matrix_free(X);
        vector_free(y);
        return;
    }
    
    MatrixErrorCode error = simple_linear_regression(X, y, result);
    
    if (error == MATRIX_SUCCESS) {
        printf("Linear regression completed successfully!\n");
        printf("Coefficients:\n");
        printf("  Intercept: %.6f\n", result->coefficients[0]);
        printf("  Slope: %.6f\n", result->coefficients[1]);
        printf("R-squared: %.6f\n", result->r_squared);
        printf("Residual Standard Error: %.6f\n", result->residual_std_error);
        
        printf("Fitted Values: ");
        for (int i = 0; i < result->n_obs; i++) {
            printf("%.3f ", result->fitted_values[i]);
        }
        printf("\n");
        
        printf("Residuals: ");
        for (int i = 0; i < result->n_obs; i++) {
            printf("%.3f ", result->residuals[i]);
        }
        printf("\n");
        
    } else {
        printf("Linear regression failed with error code: %d\n", error);
    }
    
    simple_lm_result_free(result);
    matrix_free(X);
    vector_free(y);
}

int main(void) {
    printf("Simple C Linear Regression Test\n");
    printf("===============================\n");
    
    test_simple_lm();
    
    return 0;
}