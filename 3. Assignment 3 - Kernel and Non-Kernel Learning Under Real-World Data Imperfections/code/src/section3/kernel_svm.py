import numpy as np
from utils.kernels import compute_kernel_matrix


class KernelSVM:
    """Binary kernel SVM using simplified SMO (Platt 1998; CS229 variant).

    The working-set index j is drawn from a per-model RandomState, not the
    unseeded global np.random, so fits are reproducible.

    max_passes -- consecutive no-change passes to declare convergence (CS229).
    max_epochs -- hard ceiling on outer loops, so the fit always terminates.
    converged_ -- True if max_passes was reached, False if truncated.
    """

    def __init__(self, C=1.0, kernel_type='rbf', max_passes=30,
                 max_epochs=1000, tol=1e-3, random_state=42, **kernel_params):
        self.C = C
        self.kernel_type = kernel_type
        self.max_passes = max_passes
        self.max_epochs = max_epochs
        self.tol = tol
        self.random_state = random_state
        self.kernel_params = kernel_params
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.K_train = None
        self.kernel_matrix_bytes_ = 0
        self.n_epochs_ = 0
        self.converged_ = False

    def _run_smo(self, X, y):
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        K = compute_kernel_matrix(X, kernel_type=self.kernel_type,
                                  **self.kernel_params)
        self.K_train = K
        self.kernel_matrix_bytes_ = int(K.nbytes)  # O(n^2) footprint

        alpha = np.zeros(n)
        b = 0.0
        passes = 0
        epochs = 0

        while passes < self.max_passes and epochs < self.max_epochs:
            num_changed = 0
            epochs += 1

            for i in range(n):
                E_i = float(np.sum(alpha * y * K[i, :])) + b - y[i]

                if ((y[i] * E_i < -self.tol and alpha[i] < self.C) or
                        (y[i] * E_i > self.tol and alpha[i] > 0)):

                    j = i
                    while j == i:
                        j = rng.randint(0, n)

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
        self.n_epochs_ = epochs
        self.converged_ = passes >= self.max_passes

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

    @property
    def n_support_(self):
        return int(np.sum(self.alpha > 1e-8))


class KernelSVMClassifier:
    """Multi-class kernel SVM (one-vs-rest).

    Each OVR sub-problem is seeded with random_state + index.
    """

    def __init__(self, C=1.0, kernel_type='rbf', max_passes=30,
                 max_epochs=1000, tol=1e-3, random_state=42, **kernel_params):
        self.C = C
        self.kernel_type = kernel_type
        self.max_passes = max_passes
        self.max_epochs = max_epochs
        self.tol = tol
        self.random_state = random_state
        self.kernel_params = kernel_params
        self.ovr_models_ = []
        self.classes_ = None

    def _make(self, seed_offset):
        return KernelSVM(C=self.C, kernel_type=self.kernel_type,
                         max_passes=self.max_passes,
                         max_epochs=self.max_epochs, tol=self.tol,
                         random_state=self.random_state + seed_offset,
                         **self.kernel_params)

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        if len(self.classes_) == 2:
            y_binary = np.where(y == self.classes_[1], 1, -1).astype(float)
            model = self._make(0)
            model.fit(X, y_binary)
            self.ovr_models_ = [model]
        else:
            self.ovr_models_ = []
            for ci, c in enumerate(self.classes_):
                y_binary = np.where(y == c, 1, -1).astype(float)
                model = self._make(ci)
                model.fit(X, y_binary)
                self.ovr_models_.append(model)

        return self

    def decision_scores(self, X):
        """Raw margins, for ROC-AUC."""
        if len(self.classes_) == 2:
            s = self.ovr_models_[0].decision_function(X)
            return np.column_stack([-s, s])
        return np.column_stack([m.decision_function(X)
                                for m in self.ovr_models_])

    def predict(self, X):
        if len(self.classes_) == 2:
            preds = self.ovr_models_[0].predict(X)
            return np.where(preds >= 0, self.classes_[1], self.classes_[0])
        scores = np.column_stack([m.decision_function(X)
                                  for m in self.ovr_models_])
        return self.classes_[np.argmax(scores, axis=1)]

    @property
    def kernel_matrix_bytes_(self):
        return int(sum(m.kernel_matrix_bytes_ for m in self.ovr_models_))

    @property
    def converged_(self):
        return all(m.converged_ for m in self.ovr_models_)
