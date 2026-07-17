import numpy as np


class _TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, is_leaf=False, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf
        # Leaf class distribution: a majority label alone gives a degenerate ROC.
        self.proba = proba


class DecisionTreeClassifier:
    # Gini-impurity-based decision tree classifier.

    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.classes_ = None
        self.rng = np.random.RandomState(42)

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _find_best_split(self, X, y):
        n, d = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        current_gini = self._gini_impurity(y)

        n_features = d if self.max_features is None else min(self.max_features, d)
        features = self.rng.choice(d, n_features, replace=False)

        for f in features:
            values = np.unique(X[:, f])
            if len(values) > 20:
                thresholds = np.percentile(X[:, f], np.linspace(5, 95, 15))
            else:
                thresholds = (values[:-1] + values[1:]) / 2

            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gain = current_gini - (
                    (n_left / n) * self._gini_impurity(y[left_mask]) +
                    (n_right / n) * self._gini_impurity(y[right_mask])
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        n = len(y)

        if (depth >= self.max_depth or
            n < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self._make_leaf(y)

        feature, threshold, gain = self._find_best_split(X, y)

        if feature is None or gain <= 0:
            return self._make_leaf(y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(feature=feature, threshold=threshold,
                         left=left_child, right=right_child)

    def _make_leaf(self, y):
        unique, counts = np.unique(y, return_counts=True)
        proba = np.zeros(len(self.classes_))
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        for c, ct in zip(unique, counts):
            proba[cls_idx[c]] = ct / len(y)
        return _TreeNode(value=unique[np.argmax(counts)], is_leaf=True,
                         proba=proba)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _proba_single(self, x, node):
        if node.is_leaf:
            return node.proba
        if x[node.feature] <= node.threshold:
            return self._proba_single(x, node.left)
        return self._proba_single(x, node.right)

    def predict_proba(self, X):
        return np.array([self._proba_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])


class DecisionTreeRegressor:
    # MSE-reduction-based decision tree regressor.

    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.rng = np.random.RandomState(42)

    def _variance(self, y):
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)

    def _find_best_split(self, X, y):
        n, d = X.shape
        best_reduction = -np.inf
        best_feature = None
        best_threshold = None

        current_mse = self._variance(y)

        n_features = d if self.max_features is None else min(self.max_features, d)
        features = self.rng.choice(d, n_features, replace=False)

        for f in features:
            values = np.unique(X[:, f])
            if len(values) > 20:
                thresholds = np.percentile(X[:, f], np.linspace(5, 95, 15))
            else:
                thresholds = (values[:-1] + values[1:]) / 2

            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                reduction = current_mse - (
                    (n_left / n) * self._variance(y[left_mask]) +
                    (n_right / n) * self._variance(y[right_mask])
                )

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold, best_reduction

    def _build_tree(self, X, y, depth):
        n = len(y)

        if (depth >= self.max_depth or
            n < self.min_samples_split or
            np.std(y) < 1e-10):
            return _TreeNode(value=np.mean(y), is_leaf=True)

        feature, threshold, reduction = self._find_best_split(X, y)

        if feature is None or reduction <= 0:
            return _TreeNode(value=np.mean(y), is_leaf=True)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(feature=feature, threshold=threshold,
                         left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])
