import numpy as np
from scipy.linalg import solve


class LinearRegression:
    # Ridge Regression: w = (X^T X + lambda * I)^{-1} X^T y

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

        A = X_aug.T @ X_aug + self.reg_lambda * np.eye(d_aug)
        b = X_aug.T @ y
        w = solve(A, b, assume_a='pos')

        if self.fit_intercept:
            self.bias = w[0]
            self.weights = w[1:]
        else:
            self.weights = w

        return self

    def predict(self, X):
        return X @ self.weights + self.bias
