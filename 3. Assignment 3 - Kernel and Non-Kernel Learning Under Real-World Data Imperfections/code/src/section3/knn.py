import numpy as np


class KNNClassifier:

    def __init__(self, k=5, weighted=False):
        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.astype(np.float64)
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        return self

    def _neighbour_labels(self, X):
        """Yields (row_index, neighbour_labels, neighbour_distances)."""
        X = X.astype(np.float64)
        batch_size = 500
        # argpartition needs kth < n, hence n_train - 1 rather than n_train.
        k_actual = min(self.k, max(1, len(self.X_train) - 1))

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]

            X_sq = np.sum(X_batch ** 2, axis=1).reshape(-1, 1)
            Xt_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)
            dists = np.maximum(X_sq + Xt_sq - 2.0 * X_batch @ self.X_train.T, 0.0)
            dists = np.sqrt(dists)

            nn_idx = np.argpartition(dists, k_actual, axis=1)[:, :k_actual]
            for i in range(len(X_batch)):
                idx = nn_idx[i]
                yield start + i, self.y_train[idx], dists[i, idx]

    def predict_proba(self, X):
        """Class distribution over the k neighbours (for ROC-AUC).

        With k=5 the score resolution is coarse: only six attainable values.
        """
        out = np.zeros((len(X), len(self.classes_)))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        for i, labels, dists in self._neighbour_labels(X):
            if self.weighted:
                w = 1.0 / (dists + 1e-10)
                for lab, wt in zip(labels, w):
                    out[i, cls_idx[lab]] += wt
            else:
                for lab in labels:
                    out[i, cls_idx[lab]] += 1.0
        rs = out.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        return out / rs

    def predict(self, X):
        X = X.astype(np.float64)
        predictions = []
        batch_size = 500

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]

            # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x^T x'
            X_sq = np.sum(X_batch ** 2, axis=1).reshape(-1, 1)
            Xt_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)
            dists = np.maximum(X_sq + Xt_sq - 2.0 * X_batch @ self.X_train.T, 0.0)
            dists = np.sqrt(dists)

            k_actual = min(self.k, len(self.X_train))
            nn_idx = np.argpartition(dists, k_actual, axis=1)[:, :k_actual]

            for i in range(len(X_batch)):
                idx = nn_idx[i]
                nn_labels = self.y_train[idx]
                nn_dists = dists[i, idx]

                if self.weighted:
                    weights = 1.0 / (nn_dists + 1e-10)
                    vote_counts = {}
                    for label, weight in zip(nn_labels, weights):
                        vote_counts[label] = vote_counts.get(label, 0) + weight
                    predictions.append(max(vote_counts, key=vote_counts.get))
                else:
                    unique_labels, counts = np.unique(nn_labels, return_counts=True)
                    predictions.append(unique_labels[np.argmax(counts)])

        return np.array(predictions)


class KNNRegressor:

    def __init__(self, k=5, weighted=False):
        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.astype(np.float64)
        self.y_train = y.astype(np.float64)
        return self

    def predict(self, X):
        X = X.astype(np.float64)
        predictions = []
        batch_size = 500

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]

            X_sq = np.sum(X_batch ** 2, axis=1).reshape(-1, 1)
            Xt_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)
            dists = np.maximum(X_sq + Xt_sq - 2.0 * X_batch @ self.X_train.T, 0.0)
            dists = np.sqrt(dists)

            k_actual = min(self.k, len(self.X_train))
            nn_idx = np.argpartition(dists, k_actual, axis=1)[:, :k_actual]

            for i in range(len(X_batch)):
                idx = nn_idx[i]
                nn_values = self.y_train[idx]
                nn_dists = dists[i, idx]

                if self.weighted:
                    weights = 1.0 / (nn_dists + 1e-10)
                    predictions.append(np.average(nn_values, weights=weights))
                else:
                    predictions.append(np.mean(nn_values))

        return np.array(predictions)
