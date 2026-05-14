import numpy as np
import pandas as pd
from preprocessing import load_and_preprocess
from utils import compute_metrics, aggregate_runs, print_metrics_table
import visualize
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Covid.csv"

np.random.seed(42)


class _Node:
    __slots__ = ['feat_idx', 'threshold', 'left', 'right',
                 'is_leaf', 'leaf_class', 'leaf_proba']

    def __init__(self):
        self.feat_idx = None   # feature to split on
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.leaf_class = None   # majority class at leaf
        self.leaf_proba = None


class HDDT:
    def __init__(self, min_samples_split=10, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def _hellinger_score(pos_left, neg_left, pos_right, neg_right, pos, neg):
        if pos == 0 or neg == 0:
            return 0.0
        return ((np.sqrt(pos_left / pos) - np.sqrt(neg_left / neg)) ** 2 +
                (np.sqrt(pos_right / pos) - np.sqrt(neg_right / neg)) ** 2)

    def _best_split(self, X, y):
        pos_total = int((y == 1).sum())
        neg_total = int((y == -1).sum())
        if pos_total == 0 or neg_total == 0:
            return None, None, 0.0

        best_score = -1.0
        best_feat = None
        best_thresh = None

        for f in range(X.shape[1]):
            col = X[:, f]
            sorted_unique = np.sort(np.unique(col))
            if len(sorted_unique) < 2:
                continue
            midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0

            for thresh in midpoints:
                samples_left = col <= thresh
                samples_right = ~samples_left
                if samples_left.sum() == 0 or samples_right.sum() == 0:
                    continue

                labels_left = y[samples_left]
                labels_right = y[samples_right]
                pos_left = int((labels_left == 1).sum())
                neg_left = int((labels_left == -1).sum())
                pos_right = int((labels_right == 1).sum())
                neg_right = int((labels_right == -1).sum())

                score = self._hellinger_score(
                    pos_left, neg_left, pos_right, neg_right, pos_total, neg_total)
                if score > best_score:
                    best_score = score
                    best_feat = f
                    best_thresh = thresh

        return best_feat, best_thresh, best_score

    @staticmethod
    def _make_leaf(y):

        node = _Node()
        node.is_leaf = True
        num_pos = int((y == 1).sum())
        num_neg = int((y == -1).sum())
        node.leaf_class = 1 if num_pos >= num_neg else -1
        node.leaf_proba = num_pos / len(y) if len(y) > 0 else 0.0
        return node

    def _build(self, X, y, depth):
        # Recursively build the decision tree.
        has_stop_criteria_reached = (
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1 or
            (self.max_depth is not None and
             depth >= self.max_depth)
        )
        if has_stop_criteria_reached:
            return self._make_leaf(y)

        feat, thresh, score = self._best_split(X, y)
        if feat is None:
            return self._make_leaf(y)

        node = _Node()
        node.feat_idx = feat
        node.threshold = thresh

        mask_left = X[:, feat] <= thresh
        node.left = self._build(X[mask_left], y[mask_left], depth + 1)
        node.right = self._build(X[~mask_left], y[~mask_left], depth + 1)
        return node

    def _traverse(self, x):
        node = self.root
        while not node.is_leaf:
            node = (node.left if x[node.feat_idx] <= node.threshold
                    else node.right)
        return node.leaf_class, node.leaf_proba

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self._traverse(x)[0] for x in X])

    def predict_proba(self, X):
        return np.array([self._traverse(x)[1] for x in X])


def run_single(seed, max_depth=None):
    (X_tr, X_te, y_tr, y_te, *_) = load_and_preprocess(
        CSV_PATH, random_state=seed, verbose=False)
    tree = HDDT(min_samples_split=10, max_depth=max_depth)
    tree.fit(X_tr, y_tr)
    y_pred = tree.predict(X_te)
    y_proba = tree.predict_proba(X_te)
    return compute_metrics(y_te, y_pred, y_proba)


def run_10(max_depth=None, label=""):
    tag = f"max_depth={max_depth}"

    print(f"\n{'='*65}")
    print(f"  HDDT | {tag}  {label}")
    print(f"{'='*65}")

    results = []
    for seed in range(10):
        m = run_single(seed, max_depth)

        results.append(m)
        print(f"  Seed {seed:2d} | Acc={m['accuracy']:.4f}  "
              f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}  "
              f"Gmean={m['gmean']:.4f}")
    means, stds = aggregate_runs(results)
    print_metrics_table(means, stds,
                        title=f"Aggregate 10 runs | {tag}")
    return means, stds


def main():
    print("\n" + "#"*65)
    print("#  TASK 1 - Hellinger Distance Decision Tree (HDDT)")
    print("#"*65)

    # === Full preprocessing printout (seed=0) for report documentation ===
    print("\n -> Full preprocessing output (seed=0):\n")
    (_, _, _, _, _, missing_rates, _, _) = load_and_preprocess(
        CSV_PATH, random_state=0, verbose=True)

    # ========= Section 1 plots =========
    df_raw = pd.read_csv(CSV_PATH)
    visualize.plot_class_distribution(
        df_raw['Label'].values,
        path='output/section1/class_distribution.png')
    visualize.plot_missing_rates(
        missing_rates,
        path='output/section1/missing_rates.png')

    # === Section 2.3: baseline evaluation (max_depth=None, 10 runs) ===
    print("\n Section 2.3 - Baseline (max_depth=None, 10 runs)")
    run_10(max_depth=None, label="(baseline)")

    # ========= Section 2.4: pruning experiment =========
    print("\n Section 2.4 - Pruning experiment "
          "(max_depth in {None, 2, 3, 4, 5})")
    depths = [None, 2, 3, 4, 5]
    gmean_m, gmean_s, f1_m, f1_s = [], [], [], []
    pruning = {}

    for d in depths:
        means, stds = run_10(max_depth=d)
        pruning[d] = (means, stds)
        gmean_m.append(means['gmean'])
        gmean_s.append(stds['gmean'])
        f1_m.append(means['f1'])
        f1_s.append(stds['f1'])

    visualize.plot_pruning_experiment(
        depths, gmean_m, gmean_s, f1_m, f1_s,
        path='output/section2/pruning_experiment.png')

    print("\n Pruning Summary:")
    hdr = f"  {'Depth':<10} {'Gmean_mean':>12} {'Gmean_std':>11} " \
        f"{'F1_mean':>10} {'F1_std':>10}"
    print(hdr)
    print("  " + "-"*57)
    for d, (means, stds) in pruning.items():
        dl = 'None' if d is None else str(d)
        print(f"  {dl:<10} {means['gmean']:>12.4f} {stds['gmean']:>11.4f} "
              f"{means['f1']:>10.4f} {stds['f1']:>10.4f}")

    best_d = depths[int(np.argmax(gmean_m))]
    print(f"\n  Best depth by G-mean: max_depth={best_d}")
    print("\n Task 1 complete.\n")


if __name__ == '__main__':
    main()
