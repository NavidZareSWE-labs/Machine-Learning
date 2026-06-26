import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class LabelEncoder:

    def __init__(self):
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, y):
        self.classes_ = sorted(set(y))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self.class_to_idx_.get(v, -1) for v in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_int):
        return np.array([self.classes_[i] for i in y_int])


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=False):
    rng = np.random.RandomState(random_state)
    n = len(y)

    if stratify:
        train_idx = []
        test_idx = []
        classes = np.unique(y)
        for c in classes:
            c_idx = np.where(y == c)[0]
            rng.shuffle(c_idx)
            n_test = max(1, int(len(c_idx) * test_size))
            test_idx.extend(c_idx[:n_test])
            train_idx.extend(c_idx[n_test:])
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(n * test_size)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def smote_oversample(X, y, random_state=42, k=5):
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_new = [X.copy()]
    y_new = [y.copy()]

    for cls, count in zip(classes, counts):
        if count >= max_count:
            continue
        n_needed = max_count - count
        cls_idx = np.where(y == cls)[0]
        X_cls = X[cls_idx]
        n_cls = len(cls_idx)

        k_actual = min(k, n_cls - 1)
        if k_actual < 1:
            dup_idx = rng.choice(n_cls, size=n_needed, replace=True)
            X_new.append(X_cls[dup_idx])
            y_new.append(np.full(n_needed, cls))
            continue

        synthetic = []
        for _ in range(n_needed):
            i = rng.randint(0, n_cls)
            x_i = X_cls[i]
            dists = np.sum((X_cls - x_i) ** 2, axis=1)
            dists[i] = np.inf
            nn_idx = np.argsort(dists)[:k_actual]
            j = nn_idx[rng.randint(0, len(nn_idx))]
            lam = rng.random()
            synthetic.append(x_i + lam * (X_cls[j] - x_i))

        X_new.append(np.array(synthetic))
        y_new.append(np.full(n_needed, cls))

    return np.vstack(X_new), np.concatenate(y_new)


def random_undersample(X, y, random_state=42):
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_new = []
    y_new = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        chosen = rng.choice(cls_idx, size=min_count, replace=False)
        X_new.append(X[chosen])
        y_new.append(y[chosen])

    X_out = np.vstack(X_new)
    y_out = np.concatenate(y_new)
    idx = np.arange(len(y_out))
    rng.shuffle(idx)
    return X_out[idx], y_out[idx]
