import numpy as np
from utils.kernels import compute_kernel_matrix, kernel_diagonal


class KernelKNNClassifier:
    """KNN under the kernel-induced distance d^2 = k(x,x)+k(z,z)-2k(x,z).

    Note for interpreting results: the linear kernel gives exactly Euclidean
    distance, and RBF gives d^2 = 2-2k(x,z), monotone in Euclidean distance.
    Both therefore return identical neighbours to plain KNN. Only poly differs.
    """

    def __init__(self, k=5, kernel_type='rbf', **kernel_params):
        self.k = k
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.X_train = None
        self.y_train = None
        self.K_diag_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        # Closed-form diagonal: avoids an n x n matrix built just to read it.
        self.K_diag_train = kernel_diagonal(X, kernel_type=self.kernel_type,
                                            **self.kernel_params)
        self.kernel_matrix_bytes_ = int(self.K_diag_train.nbytes)
        return self

    def predict(self, X):
        K_cross = compute_kernel_matrix(X, self.X_train,
                                         kernel_type=self.kernel_type,
                                         **self.kernel_params)
        K_diag_test = kernel_diagonal(X, kernel_type=self.kernel_type,
                                      **self.kernel_params)

        dists = np.maximum(
            K_diag_test.reshape(-1, 1) + self.K_diag_train.reshape(1, -1) - 2.0 * K_cross,
            0.0
        )

        predictions = []
        k_actual = min(self.k, max(1, len(self.X_train) - 1))

        self._last_proba = np.zeros((len(X), len(self.classes_)))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}

        for i in range(len(X)):
            nn_idx = np.argpartition(dists[i], k_actual)[:k_actual]
            nn_labels = self.y_train[nn_idx]
            for lab in nn_labels:
                self._last_proba[i, cls_idx[lab]] += 1.0 / k_actual
            unique_labels, counts = np.unique(nn_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])

        return np.array(predictions)

    def predict_proba(self, X):
        self.predict(X)
        return self._last_proba


class KernelKNNRegressor:

    def __init__(self, k=5, kernel_type='rbf', **kernel_params):
        self.k = k
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.X_train = None
        self.y_train = None
        self.K_diag_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.astype(np.float64)
        # Closed-form diagonal: avoids an n x n matrix built just to read it.
        self.K_diag_train = kernel_diagonal(X, kernel_type=self.kernel_type,
                                            **self.kernel_params)
        self.kernel_matrix_bytes_ = int(self.K_diag_train.nbytes)
        return self

    def predict(self, X):
        K_cross = compute_kernel_matrix(X, self.X_train,
                                         kernel_type=self.kernel_type,
                                         **self.kernel_params)
        K_diag_test = kernel_diagonal(X, kernel_type=self.kernel_type,
                                      **self.kernel_params)

        dists = np.maximum(
            K_diag_test.reshape(-1, 1) + self.K_diag_train.reshape(1, -1) - 2.0 * K_cross,
            0.0
        )

        predictions = []
        k_actual = min(self.k, len(self.X_train))

        for i in range(len(X)):
            nn_idx = np.argpartition(dists[i], k_actual)[:k_actual]
            nn_values = self.y_train[nn_idx]
            predictions.append(np.mean(nn_values))

        return np.array(predictions)
