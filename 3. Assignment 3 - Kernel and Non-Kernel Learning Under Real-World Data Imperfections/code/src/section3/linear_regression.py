import numpy as np
from scipy.linalg import solve


class LinearRegression:
    """Ridge Regression: w = (X^T X + lambda * D)^{-1} X^T y

    D is identity with D[0,0] = 0: the intercept is not penalised, so the fit
    stays invariant to a constant shift of y. (np.eye would shrink the bias.)
    """

    def __init__(self, reg_lambda=1.0, fit_intercept=True):
        self.reg_lambda = reg_lambda
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n, d = X.shape

        if self.fit_intercept:
            X_aug = np.hstack([np.ones((n, 1)), X])
        else:
            X_aug = X

        d_aug = X_aug.shape[1]

        D = np.eye(d_aug)
        if self.fit_intercept:
            D[0, 0] = 0.0

        A = X_aug.T @ X_aug + self.reg_lambda * D
        b = X_aug.T @ y

        # Collinear features can break positive-definiteness; fall back.
        try:
            w = solve(A, b, assume_a='pos')
        except Exception:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        if self.fit_intercept:
            self.bias = w[0]
            self.weights = w[1:]
        else:
            self.weights = w
            self.bias = 0.0

        return self

    def predict(self, X):
        return X @ self.weights + self.bias
