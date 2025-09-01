#ifndef QR_DECOMPOSITION_H
#define QR_DECOMPOSITION_H

#include "matrix.h"

typedef struct {
    Matrix Q;
    Matrix R;
} QRDecomposition;

QRDecomposition* qr_decomposition_alloc(int rows, int cols);
void qr_decomposition_free(QRDecomposition *qr);

MatrixErrorCode qr_householder(const Matrix *A, QRDecomposition *qr);

MatrixErrorCode qr_solve(const QRDecomposition *qr, const Vector *b, Vector *x);

MatrixErrorCode back_substitution(const Matrix *R, const Vector *b, Vector *x);

double matrix_condition_number(const Matrix *A);
int matrix_rank(const Matrix *A, double tolerance);

#endif