import numpy as np
from utils.kernels import compute_kernel_matrix


class KernelSVM:
    # Binary kernel SVM using simplified SMO optimization.

    def __init__(self, C=1.0, kernel_type='rbf', max_iter=100, tol=1e-3,
                 **kernel_params):
        self.C = C
        self.kernel_type = kernel_type
        self.max_iter = max_iter
        self.tol = tol
        self.kernel_params = kernel_params
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.K_train = None

    def _run_smo(self, X, y):
        n = X.shape[0]

        K = compute_kernel_matrix(X, kernel_type=self.kernel_type, **self.kernel_params)
        self.K_train = K

        alpha = np.zeros(n)
        b = 0.0
        passes = 0

        while passes < self.max_iter:
            num_changed = 0

            for i in range(n):
                E_i = float(np.sum(alpha * y * K[i, :])) + b - y[i]

                if ((y[i] * E_i < -self.tol and alpha[i] < self.C) or
                    (y[i] * E_i > self.tol and alpha[i] > 0)):

                    j = i
                    while j == i:
                        j = np.random.randint(0, n)

                    E_j = float(np.sum(alpha * y * K[j, :])) + b - y[j]

                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L >= H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha[j] -= y[j] * (E_i - E_j) / eta
                    alpha[j] = np.clip(alpha[j], L, H)

                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                    b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] \
                         - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] \
                         - y[j] * (alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0

                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        self.alpha = alpha
        self.b = b

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self._run_smo(X, y)
        return self

    def decision_function(self, X):
        K_test = compute_kernel_matrix(X, self.X_train,
                                        kernel_type=self.kernel_type,
                                        **self.kernel_params)
        return K_test @ (self.alpha * self.y_train) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)


class KernelSVMClassifier:
    # Multi-class kernel SVM using one-vs-rest.

    def __init__(self, C=1.0, kernel_type='rbf', max_iter=50, **kernel_params):
        self.C = C
        self.kernel_type = kernel_type
        self.max_iter = max_iter
        self.kernel_params = kernel_params
        self.ovr_models_ = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            y_binary = np.where(y == self.classes_[1], 1, -1).astype(float)
            model = KernelSVM(C=self.C, kernel_type=self.kernel_type,
                              max_iter=self.max_iter, **self.kernel_params)
            model.fit(X, y_binary)
            self.ovr_models_ = [model]
        else:
            self.ovr_models_ = []
            for c in self.classes_:
                y_binary = np.where(y == c, 1, -1).astype(float)
                model = KernelSVM(C=self.C, kernel_type=self.kernel_type,
                                  max_iter=self.max_iter, **self.kernel_params)
                model.fit(X, y_binary)
                self.ovr_models_.append(model)

        return self

    def predict(self, X):
        if len(self.classes_) == 2:
            preds = self.ovr_models_[0].predict(X)
            return np.where(preds >= 0, self.classes_[1], self.classes_[0])
        else:
            scores = np.column_stack([m.decision_function(X) for m in self.ovr_models_])
            return self.classes_[np.argmax(scores, axis=1)]
