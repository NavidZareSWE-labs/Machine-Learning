import numpy as np


class LogisticRegression:
    # Gradient descent logistic regression with one-vs-rest for multi-class.

    def __init__(self, lr=0.01, n_iters=1000, reg_lambda=0.01, fit_intercept=True):
        self.lr = lr
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = 0.0
        self.classes_ = None
        self.ovr_models_ = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _fit_binary(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0

        for i in range(self.n_iters):
            z = X @ w + b
            h = self._sigmoid(z)

            error = h - y
            dw = (1.0 / n) * (X.T @ error) + self.reg_lambda * w
            db = (1.0 / n) * np.sum(error)

            w -= self.lr * dw
            b -= self.lr * db

        return w, b

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            y_binary = (y == self.classes_[1]).astype(float)
            self.weights, self.bias = self._fit_binary(X, y_binary)
        else:
            self.ovr_models_ = []
            for c in self.classes_:
                y_binary = (y == c).astype(float)
                w, b = self._fit_binary(X, y_binary)
                self.ovr_models_.append((w, b))

        return self

    def predict_proba(self, X):
        if len(self.classes_) == 2:
            p1 = self._sigmoid(X @ self.weights + self.bias)
            return np.column_stack([1 - p1, p1])
        else:
            scores = []
            for w, b in self.ovr_models_:
                scores.append(self._sigmoid(X @ w + b))
            scores = np.column_stack(scores)
            row_sums = scores.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return scores / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
