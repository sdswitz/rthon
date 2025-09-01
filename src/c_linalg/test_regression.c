#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "qr_decomposition.h"
#include "linear_regression.h"

void test_simple_regression(void) {
    printf("=== Testing Simple Linear Regression ===\n");
    
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
    
    printf("Allocating LinearModelResult...\n");
    LinearModelResult *result = linear_model_result_alloc(5, 2);
    if (!result) {
        printf("Result allocation failed\n");
        matrix_free(X);
        vector_free(y);
        return;
    }
    printf("LinearModelResult allocated successfully\n");
    
    printf("Starting compute_linear_regression...\n");
    MatrixErrorCode error = compute_linear_regression(X, y, result);
    printf("compute_linear_regression completed with error code: %d\n", error);
    
    if (error == MATRIX_SUCCESS) {
        printf("Regression completed successfully!\n");
        printf("Coefficients:\n");
        printf("  Intercept: %.6f\n", result->coefficients.data[0]);
        printf("  Slope: %.6f\n", result->coefficients.data[1]);
        printf("R-squared: %.6f\n", result->r_squared);
        printf("Residual Standard Error: %.6f\n", result->residual_std_error);
        printf("F-statistic: %.6f\n", result->f_statistic);
        
        printf("Standard Errors:\n");
        for (int i = 0; i < result->coefficients.length; i++) {
            printf("  SE[%d]: %.6f\n", i, result->standard_errors[i]);
        }
        
        printf("Fitted Values: ");
        for (int i = 0; i < result->fitted_values.length; i++) {
            printf("%.3f ", result->fitted_values.data[i]);
        }
        printf("\n");
        
        printf("Residuals: ");
        for (int i = 0; i < result->residuals.length; i++) {
            printf("%.3f ", result->residuals.data[i]);
        }
        printf("\n");
        
    } else {
        printf("Regression failed with error code: %d\n", error);
    }
    
    linear_model_result_free(result);
    matrix_free(X);
    vector_free(y);
}

void test_multiple_regression(void) {
    printf("\n=== Testing Multiple Linear Regression ===\n");
    
    double X_data[6][3] = {
        {1.0, 2.0, 1.0},
        {1.0, 3.0, 1.5},
        {1.0, 4.0, 2.0},
        {1.0, 5.0, 2.5},
        {1.0, 6.0, 3.0},
        {1.0, 7.0, 3.5}
    };
    
    double y_data[6] = {5.1, 7.2, 9.1, 11.3, 13.0, 15.2};
    
    Matrix *X = matrix_alloc(6, 3);
    Vector *y = vector_alloc(6);
    
    if (!X || !y) {
        printf("Memory allocation failed\n");
        return;
    }
    
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            X->data[i][j] = X_data[i][j];
        }
        y->data[i] = y_data[i];
    }
    
    LinearModelResult *result = linear_model_result_alloc(6, 3);
    if (!result) {
        printf("Result allocation failed\n");
        matrix_free(X);
        vector_free(y);
        return;
    }
    
    MatrixErrorCode error = compute_linear_regression(X, y, result);
    
    if (error == MATRIX_SUCCESS) {
        printf("Multiple regression completed successfully!\n");
        printf("Coefficients:\n");
        printf("  Intercept: %.6f\n", result->coefficients.data[0]);
        printf("  X1: %.6f\n", result->coefficients.data[1]);
        printf("  X2: %.6f\n", result->coefficients.data[2]);
        printf("R-squared: %.6f\n", result->r_squared);
        printf("Residual Standard Error: %.6f\n", result->residual_std_error);
        printf("F-statistic: %.6f\n", result->f_statistic);
        
    } else {
        printf("Multiple regression failed with error code: %d\n", error);
    }
    
    linear_model_result_free(result);
    matrix_free(X);
    vector_free(y);
}

int main(void) {
    printf("C Linear Regression Implementation Test\n");
    printf("======================================\n");
    
    test_simple_regression();
    test_multiple_regression();
    
    return 0;
}