import numpy as np
from utils.kernels import compute_kernel_matrix


class KernelKNNClassifier:
    # KNN using kernel-induced distance: d^2(x,z) = k(x,x) + k(z,z) - 2*k(x,z)

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
        K_train = compute_kernel_matrix(X, kernel_type=self.kernel_type,
                                         **self.kernel_params)
        self.K_diag_train = np.diag(K_train)
        return self

    def predict(self, X):
        K_cross = compute_kernel_matrix(X, self.X_train,
                                         kernel_type=self.kernel_type,
                                         **self.kernel_params)
        K_test = compute_kernel_matrix(X, kernel_type=self.kernel_type,
                                        **self.kernel_params)
        K_diag_test = np.diag(K_test)

        dists = np.maximum(
            K_diag_test.reshape(-1, 1) + self.K_diag_train.reshape(1, -1) - 2.0 * K_cross,
            0.0
        )

        predictions = []
        k_actual = min(self.k, len(self.X_train))

        for i in range(len(X)):
            nn_idx = np.argpartition(dists[i], k_actual)[:k_actual]
            nn_labels = self.y_train[nn_idx]
            unique_labels, counts = np.unique(nn_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])

        return np.array(predictions)


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
        K_train = compute_kernel_matrix(X, kernel_type=self.kernel_type,
                                         **self.kernel_params)
        self.K_diag_train = np.diag(K_train)
        return self

    def predict(self, X):
        K_cross = compute_kernel_matrix(X, self.X_train,
                                         kernel_type=self.kernel_type,
                                         **self.kernel_params)
        K_test = compute_kernel_matrix(X, kernel_type=self.kernel_type,
                                        **self.kernel_params)
        K_diag_test = np.diag(K_test)

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
