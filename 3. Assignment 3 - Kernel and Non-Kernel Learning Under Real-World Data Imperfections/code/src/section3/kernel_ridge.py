import numpy as np
from scipy.linalg import solve
from utils.kernels import compute_kernel_matrix


class KernelRidgeRegression:
    # Dual form: alpha = (K + lambda * I)^{-1} y, predict: g(x) = K_test @ alpha

    def __init__(self, kernel_type='rbf', reg_lambda=1.0, **kernel_params):
        self.kernel_type = kernel_type
        self.reg_lambda = reg_lambda
        self.kernel_params = kernel_params
        self.alpha = None
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        n = X.shape[0]

        K = compute_kernel_matrix(X, kernel_type=self.kernel_type, **self.kernel_params)
        A = K + self.reg_lambda * np.eye(n)
        self.alpha = solve(A, y, assume_a='pos')

        return self

    def predict(self, X):
        K_test = compute_kernel_matrix(X, self.X_train,
                                        kernel_type=self.kernel_type,
                                        **self.kernel_params)
        return K_test @ self.alpha


class KernelRidgeClassifier:
    # One-vs-rest classification wrapper around KernelRidgeRegression.

    def __init__(self, kernel_type='rbf', reg_lambda=1.0, **kernel_params):
        self.kernel_type = kernel_type
        self.reg_lambda = reg_lambda
        self.kernel_params = kernel_params
        self.ovr_models_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            y_binary = (y == self.classes_[1]).astype(float)
            model = KernelRidgeRegression(self.kernel_type, self.reg_lambda,
                                           **self.kernel_params)
            model.fit(X, y_binary)
            self.ovr_models_ = [model]
        else:
            self.ovr_models_ = []
            for c in self.classes_:
                y_binary = (y == c).astype(float)
                model = KernelRidgeRegression(self.kernel_type, self.reg_lambda,
                                               **self.kernel_params)
                model.fit(X, y_binary)
                self.ovr_models_.append(model)

        return self

    def predict(self, X):
        if len(self.classes_) == 2:
            scores = self.ovr_models_[0].predict(X)
            return self.classes_[(scores >= 0.5).astype(int)]
        else:
            all_scores = np.column_stack([m.predict(X) for m in self.ovr_models_])
            return self.classes_[np.argmax(all_scores, axis=1)]
