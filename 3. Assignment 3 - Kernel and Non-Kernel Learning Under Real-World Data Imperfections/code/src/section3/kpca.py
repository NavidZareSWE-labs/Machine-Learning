import numpy as np
from scipy.linalg import eigh
from utils.kernels import compute_kernel_matrix, center_kernel_matrix


class KernelPCA:

    def __init__(self, n_components=50, kernel_type='rbf', **kernel_params):
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.X_train = None
        self.K_train = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.kernel_matrix_bytes_ = 0

    def fit(self, X):
        self.X_train = X.copy()
        n = X.shape[0]

        K = compute_kernel_matrix(X, kernel_type=self.kernel_type, **self.kernel_params)
        self.K_train = K
        self.kernel_matrix_bytes_ = int(K.nbytes)

        K_centered = center_kernel_matrix(K)

        n_comp = min(self.n_components, n - 1)
        eigenvalues, eigenvectors = eigh(K_centered,
                                          subset_by_index=[n - n_comp, n - 1])

        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        for i in range(len(self.eigenvalues)):
            if self.eigenvalues[i] > 1e-10:
                self.eigenvectors[:, i] /= np.sqrt(self.eigenvalues[i])

        return self

    def transform(self, X):
        K_test = compute_kernel_matrix(X, self.X_train,
                                        kernel_type=self.kernel_type,
                                        **self.kernel_params)
        K_test_centered = center_kernel_matrix(K_test, self.K_train)
        return K_test_centered @ self.eigenvectors

    def fit_transform(self, X):
        self.fit(X)
        K_centered = center_kernel_matrix(self.K_train)
        return K_centered @ self.eigenvectors


class KPCAClassifier:
    # KPCA dimensionality reduction + downstream logistic regression.

    def __init__(self, n_components=50, kernel_type='rbf',
                 lr=0.01, n_iters=500, **kernel_params):
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.kpca = None
        self.classifier = None
        self.lr = lr
        self.n_iters = n_iters
        self.kernel_matrix_bytes_ = 0

    def fit(self, X, y):
        from .logistic_regression import LogisticRegression

        self.kpca = KernelPCA(
            n_components=self.n_components,
            kernel_type=self.kernel_type,
            **self.kernel_params
        )

        X_transformed = self.kpca.fit_transform(X)
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        self.kernel_matrix_bytes_ = self.kpca.kernel_matrix_bytes_

        self.classifier = LogisticRegression(
            lr=self.lr, n_iters=self.n_iters, reg_lambda=0.01
        )
        self.classifier.fit(X_transformed, y)

        return self

    def decision_scores(self, X):
        X_transformed = self.kpca.transform(X)
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        return self.classifier.predict_proba(X_transformed)

    def predict(self, X):
        X_transformed = self.kpca.transform(X)
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        return self.classifier.predict(X_transformed)

    @property
    def classes_(self):
        return self.classifier.classes_
