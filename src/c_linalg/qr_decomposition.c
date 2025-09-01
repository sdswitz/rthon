#include "qr_decomposition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

QRDecomposition* qr_decomposition_alloc(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return NULL;
    }
    
    QRDecomposition *qr = malloc(sizeof(QRDecomposition));
    if (!qr) {
        return NULL;
    }
    
    Matrix *Q = matrix_alloc(rows, rows);
    Matrix *R = matrix_alloc(rows, cols);
    
    if (!Q || !R) {
        matrix_free(Q);
        matrix_free(R);
        free(qr);
        return NULL;
    }
    
    qr->Q = *Q;
    qr->R = *R;
    free(Q);
    free(R);
    
    return qr;
}

void qr_decomposition_free(QRDecomposition *qr) {
    if (!qr) return;
    
    matrix_free(&qr->Q);
    matrix_free(&qr->R);
    free(qr);
}

static void householder_vector(const Vector *x, Vector *v, double *beta) {
    if (!vector_is_valid(x) || !vector_is_valid(v) || x->length != v->length) {
        *beta = 0.0;
        return;
    }
    
    int n = x->length;
    double norm_x = vector_norm(x);
    
    if (norm_x < 1e-14) {
        memset(v->data, 0, n * sizeof(double));
        *beta = 0.0;
        return;
    }
    
    double sigma = 0.0;
    for (int i = 1; i < n; i++) {
        sigma += x->data[i] * x->data[i];
    }
    
    v->data[0] = 1.0;
    for (int i = 1; i < n; i++) {
        v->data[i] = x->data[i];
    }
    
    if (sigma < 1e-14) {
        *beta = 0.0;
    } else {
        double mu = sqrt(x->data[0] * x->data[0] + sigma);
        double v1;
        
        if (x->data[0] <= 0) {
            v1 = x->data[0] - mu;
        } else {
            v1 = -sigma / (x->data[0] + mu);
        }
        
        *beta = 2.0 * v1 * v1 / (sigma + v1 * v1);
        
        v->data[0] = v1;
        for (int i = 1; i < n; i++) {
            v->data[i] = x->data[i];
        }
        
        double v_norm = sqrt(v1 * v1 + sigma);
        if (v_norm > 1e-14) {
            for (int i = 0; i < n; i++) {
                v->data[i] /= v_norm;
            }
        }
    }
}

static void apply_householder_left(Matrix *A, const Vector *v, double beta, int start_row, int start_col) {
    if (!matrix_is_valid(A) || !vector_is_valid(v) || beta < 1e-14) {
        return;
    }
    
    int effective_rows = A->rows - start_row;
    if (effective_rows != v->length) {
        return;
    }
    
    for (int j = start_col; j < A->cols; j++) {
        double dot_product = 0.0;
        for (int i = 0; i < effective_rows; i++) {
            dot_product += v->data[i] * A->data[start_row + i][j];
        }
        
        for (int i = 0; i < effective_rows; i++) {
            A->data[start_row + i][j] -= beta * dot_product * v->data[i];
        }
    }
}

MatrixErrorCode qr_householder(const Matrix *A, QRDecomposition *qr) {
    if (!matrix_is_valid(A) || !qr) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    int m = A->rows;
    int n = A->cols;
    
    if (qr->Q.rows != m || qr->Q.cols != m || qr->R.rows != m || qr->R.cols != n) {
        return MATRIX_ERROR_INVALID_DIMENSIONS;
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            qr->R.data[i][j] = A->data[i][j];
        }
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            qr->Q.data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    Vector *v = NULL;
    Vector *x = NULL;
    
    int min_mn = (m < n) ? m : n;
    
    for (int k = 0; k < min_mn - 1; k++) {
        int col_length = m - k;
        
        if (x) vector_free(x);
        if (v) vector_free(v);
        
        x = vector_alloc(col_length);
        v = vector_alloc(col_length);
        
        if (!x || !v) {
            vector_free(x);
            vector_free(v);
            return MATRIX_ERROR_MEMORY_ALLOCATION;
        }
        
        for (int i = 0; i < col_length; i++) {
            x->data[i] = qr->R.data[k + i][k];
        }
        
        double beta;
        householder_vector(x, v, &beta);
        
        if (beta > 1e-14) {
            apply_householder_left(&qr->R, v, beta, k, k);
            
            for (int j = 0; j < m; j++) {
                double dot_product = 0.0;
                for (int i = 0; i < col_length; i++) {
                    dot_product += v->data[i] * qr->Q.data[k + i][j];
                }
                
                for (int i = 0; i < col_length; i++) {
                    qr->Q.data[k + i][j] -= beta * dot_product * v->data[i];
                }
            }
        }
    }
    
    vector_free(x);
    vector_free(v);
    
    Matrix *Q_transpose = matrix_alloc(m, m);
    if (!Q_transpose) {
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    matrix_transpose(&qr->Q, Q_transpose);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            qr->Q.data[i][j] = Q_transpose->data[i][j];
        }
    }
    
    matrix_free(Q_transpose);
    
    return MATRIX_SUCCESS;
}

MatrixErrorCode back_substitution(const Matrix *R, const Vector *b, Vector *x) {
    if (!matrix_is_valid(R) || !vector_is_valid(b) || !vector_is_valid(x)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    int n = x->length;
    
    if (R->rows < n || R->cols < n || b->length < n) {
        return MATRIX_ERROR_INVALID_DIMENSIONS;
    }
    
    for (int i = n - 1; i >= 0; i--) {
        if (fabs(R->data[i][i]) < 1e-12) {
            return MATRIX_ERROR_SINGULAR_MATRIX;
        }
        
        x->data[i] = b->data[i];
        
        for (int j = i + 1; j < n; j++) {
            x->data[i] -= R->data[i][j] * x->data[j];
        }
        
        x->data[i] /= R->data[i][i];
    }
    
    return MATRIX_SUCCESS;
}

MatrixErrorCode qr_solve(const QRDecomposition *qr, const Vector *b, Vector *x) {
    if (!qr || !vector_is_valid(b) || !vector_is_valid(x)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    if (!matrix_is_valid(&qr->Q) || !matrix_is_valid(&qr->R)) {
        return MATRIX_ERROR_NULL_POINTER;
    }
    
    Vector *Qt_b = vector_alloc(b->length);
    if (!Qt_b) {
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    Matrix *Q_transpose = matrix_alloc(qr->Q.cols, qr->Q.rows);
    if (!Q_transpose) {
        vector_free(Qt_b);
        return MATRIX_ERROR_MEMORY_ALLOCATION;
    }
    
    matrix_transpose(&qr->Q, Q_transpose);
    matrix_vector_multiply(Q_transpose, b, Qt_b);
    
    MatrixErrorCode result = back_substitution(&qr->R, Qt_b, x);
    
    matrix_free(Q_transpose);
    vector_free(Qt_b);
    
    return result;
}

int matrix_rank(const Matrix *A, double tolerance) {
    printf("matrix_rank: Starting\n");
    if (!matrix_is_valid(A)) {
        printf("matrix_rank: Invalid matrix\n");
        return -1;
    }
    
    printf("matrix_rank: Allocating QR decomposition %dx%d\n", A->rows, A->cols);
    QRDecomposition *qr = qr_decomposition_alloc(A->rows, A->cols);
    if (!qr) {
        printf("matrix_rank: QR allocation failed\n");
        return -1;
    }
    printf("matrix_rank: QR allocated successfully\n");
    
    MatrixErrorCode result = qr_householder(A, qr);
    if (result != MATRIX_SUCCESS) {
        qr_decomposition_free(qr);
        return -1;
    }
    
    int rank = 0;
    int min_dim = (qr->R.rows < qr->R.cols) ? qr->R.rows : qr->R.cols;
    
    for (int i = 0; i < min_dim; i++) {
        if (fabs(qr->R.data[i][i]) > tolerance) {
            rank++;
        }
    }
    
    qr_decomposition_free(qr);
    return rank;
}

double matrix_condition_number(const Matrix *A) {
    if (!matrix_is_valid(A)) {
        return INFINITY;
    }
    
    QRDecomposition *qr = qr_decomposition_alloc(A->rows, A->cols);
    if (!qr) {
        return INFINITY;
    }
    
    MatrixErrorCode result = qr_householder(A, qr);
    if (result != MATRIX_SUCCESS) {
        qr_decomposition_free(qr);
        return INFINITY;
    }
    
    double max_diag = 0.0;
    double min_diag = INFINITY;
    int min_dim = (qr->R.rows < qr->R.cols) ? qr->R.rows : qr->R.cols;
    
    for (int i = 0; i < min_dim; i++) {
        double abs_diag = fabs(qr->R.data[i][i]);
        if (abs_diag > 1e-14) {
            if (abs_diag > max_diag) max_diag = abs_diag;
            if (abs_diag < min_diag) min_diag = abs_diag;
        }
    }
    
    qr_decomposition_free(qr);
    
    if (min_diag < 1e-14) {
        return INFINITY;
    }
    
    return max_diag / min_diag;
}