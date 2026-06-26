import numpy as np


def linear_kernel(X, Y=None, c=0.0):
    # k(x, x') = x^T x' + c
    if Y is None:
        Y = X
    return X @ Y.T + c


def polynomial_kernel(X, Y=None, degree=3, alpha=1.0, c=1.0):
    # k(x, x') = (alpha * x^T x' + c)^d
    if Y is None:
        Y = X
    return (alpha * X @ Y.T + c) ** degree


def rbf_kernel(X, Y=None, gamma=None, sigma=None):
    # k(x, x') = exp(-gamma * ||x - x'||^2)
    if Y is None:
        Y = X

    if gamma is not None:
        pass
    elif sigma is not None:
        gamma = 1.0 / (2.0 * sigma ** 2)
    else:
        gamma = 1.0 / (2.0 * X.shape[1])

    X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)
    dist_sq = np.maximum(X_sq + Y_sq - 2.0 * X @ Y.T, 0.0)
    return np.exp(-gamma * dist_sq)


def compute_kernel_matrix(X, Y=None, kernel_type='rbf', **kwargs):
    if kernel_type == 'linear':
        return linear_kernel(X, Y, c=kwargs.get('c', 0.0))
    elif kernel_type == 'poly':
        return polynomial_kernel(
            X, Y,
            degree=kwargs.get('degree', 3),
            alpha=kwargs.get('alpha', 1.0),
            c=kwargs.get('c', 1.0)
        )
    elif kernel_type == 'rbf':
        return rbf_kernel(X, Y, gamma=kwargs.get('gamma', None),
                          sigma=kwargs.get('sigma', None))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def center_kernel_matrix(K, K_train=None):
    n = K.shape[0]
    if K_train is None:
        one_n = np.ones((n, n)) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n
    else:
        n_train = K_train.shape[0]
        one_n_test = np.ones((K.shape[0], n_train)) / n_train
        one_n_train = np.ones((n_train, n_train)) / n_train
        return (K - one_n_test @ K_train
                - K @ one_n_train
                + one_n_test @ K_train @ one_n_train)
