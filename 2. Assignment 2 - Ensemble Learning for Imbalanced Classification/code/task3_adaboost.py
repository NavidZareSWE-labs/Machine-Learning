import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from preprocessing import load_and_preprocess
from utils import compute_metrics, aggregate_runs, print_metrics_table
import visualize
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Covid.csv"

np.random.seed(42)


class SMOTE:

    def __init__(self, k=5, random_state=42):
        self.k = k
        self.random_state = random_state

    @staticmethod
    def _euclidean_distances(a, B):
        diff = B - a
        return np.sqrt((diff ** 2).sum(axis=1))

    def fit_resample(self, X, y):
        np.random.seed(self.random_state)

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == -1)[0]
        n_maj = len(neg_idx)
        n_min = len(pos_idx)
        n_needed = n_maj - n_min

        X_min = X[pos_idx]

        neighbors = []
        for i in range(n_min):
            dists = self._euclidean_distances(X_min[i], X_min)
            dists[i] = np.inf
            nn_idx = np.argsort(dists)[:self.k]
            neighbors.append(nn_idx)

        synthetic = []
        for _ in range(n_needed):
            i = np.random.choice(n_min)
            nn_idx = neighbors[i][np.random.choice(self.k)]
            lam = np.random.uniform(0, 1)
            x_new = X_min[i] + lam * (X_min[nn_idx] - X_min[i])
            synthetic.append(x_new)

        X_synthetic = np.array(synthetic)
        y_synthetic = np.ones(len(X_synthetic), dtype=int)

        X_res = np.vstack([X, X_synthetic])
        y_res = np.concatenate([y, y_synthetic])
        return X_res, y_res


class AdaBoostM1:

    def __init__(self, T=50, learning_rate=1.0):
        self.T = T
        self.learning_rate = learning_rate
        self.alphas_ = []
        self.estimators_ = []
        self.n_rounds_ = 0

    def fit(self, X, y):
        N = len(y)
        # D_1(i) = 1/m
        w = np.ones(N) / N
        self.alphas_ = []
        self.estimators_ = []

        for t in range(self.T):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)
            y_pred = stump.predict(X)

            incorrect = (y_pred != y).astype(float)
            err_t = float(np.dot(w, incorrect))

            # Stopping Condition
            if err_t >= 0.5:
                print(f"    [AdaBoost] Early stop at round {t+1}: "
                      f"eps={err_t:.4f} >= 0.5")
                break

            # Edge case: perfect classifier
            if err_t == 0.0:
                # Assign high alpha and stop (perfect classifier found)
                alpha_t = self.learning_rate * 10.0
                self.alphas_.append(alpha_t)
                self.estimators_.append(stump)
                break

            alpha_t = self.learning_rate * 0.5 * np.log((1.0 - err_t) / err_t)

            # Update weights
            w = w * np.exp(-alpha_t * y * y_pred)
            w_sum = w.sum()
            if w_sum == 0:
                break
            w = w / w_sum

            self.alphas_.append(alpha_t)
            self.estimators_.append(stump)

        self.n_rounds_ = len(self.alphas_)
        return self

    def _raw_scores(self, X):
        scores = np.zeros(len(X))
        for alpha, classifier in zip(self.alphas_, self.estimators_):
            scores += alpha * classifier.predict(X)
        return scores

    def predict(self, X):
        scores = self._raw_scores(X)
        return np.where(scores >= 0, 1, -1)

    def predict_proba(self, X):
        scores = self._raw_scores(X)
        # sigmoid
        return 1.0 / (1.0 + np.exp(-scores))


def run_single(seed, T=50, use_smote=False):
    (X_tr, X_te, y_tr, y_te, *_) = load_and_preprocess(
        CSV_PATH, random_state=seed, verbose=False)

    if use_smote:
        smote = SMOTE(k=5, random_state=seed)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

    ada = AdaBoostM1(T=T, learning_rate=1.0)
    ada.fit(X_tr, y_tr)

    y_pred = ada.predict(X_te)
    y_proba = ada.predict_proba(X_te)
    return compute_metrics(y_te, y_pred, y_proba), ada.n_rounds_


def run_10(T=50, use_smote=False, label=""):
    smote_tag = "with SMOTE" if use_smote else "no SMOTE"
    tag = f"AdaBoost | T={T} | {smote_tag}"

    print(f"\n{'=' * 65}")
    print(f"  {tag}  {label}")
    print(f"{'=' * 65}")

    results = []
    for seed in range(10):
        m, rounds = run_single(seed, T, use_smote)
        results.append(m)
        print(f"  Seed {seed:2d} | "
              f"Acc {m['accuracy']:.4f}  Prec {m['precision']:.4f}  "
              f"Rec {m['recall']:.4f}  F1 {m['f1']:.4f}  "
              f"AUC {m['auc_roc']:.4f}  Gm {m['gmean']:.4f}  "
              f"Rounds {rounds}")

    means, stds = aggregate_runs(results)
    print_metrics_table(means, stds, title=f"Aggregate 10 runs | {tag}")
    return means, stds


def main():
    os.makedirs('output/section4', exist_ok=True)
    print(f"\n{'#' * 65}")
    print(f"#  TASK 3 - AdaBoost.M1 + SMOTE")
    print(f"{'#' * 65}")

    # ==== Experiment A: AdaBoost vs AdaBoost+SMOTE (T=50) ==============
    print("\n Experiment A - AdaBoost vs AdaBoost + SMOTE (T=50, 10 runs each)")

    no_smote_m, _ = run_10(T=50, use_smote=False, label="(Exp A)")
    smote_m,    _ = run_10(T=50, use_smote=True,  label="(Exp A)")

    cols = [('Acc', 'accuracy'), ('Prec', 'precision'), ('Rec', 'recall'),
            ('F1', 'f1'), ('AUC-ROC', 'auc_roc'), ('G-mean', 'gmean')]
    visualize._print_comparison_table(
        [('AdaBoost (no SMOTE)', no_smote_m),
         ('AdaBoost + SMOTE',    smote_m)],
        cols, title="Experiment A Summary")

    visualize.plot_adaboost_smote_comparison(
        no_smote_m, smote_m,
        path='output/section4/adaboost_smote_comparison.png')

    # ==== Experiment B: Varying T with SMOTE ============================
    print("\n Experiment B - Vary T with SMOTE (seed=0, train+test error tracking)")

    T_vals = [10, 25, 50, 100]

    (X_tr0, X_te0, y_tr0, y_te0, *_) = load_and_preprocess(
        CSV_PATH, random_state=0, verbose=False)
    smote0 = SMOTE(k=5, random_state=0)
    X_tr_sm0, y_tr_sm0 = smote0.fit_resample(X_tr0, y_tr0)

    train_errors, test_errors, f1_scores = [], [], []

    print(f"\n  {'T':>5} | {'Train Err':>10} {'Test Err':>10} {'F1 (+1)':>9}")
    print(f"  {'-' * 5}-+-{'-' * 10}-{'-' * 10}-{'-' * 9}")

    for T in T_vals:
        ada = AdaBoostM1(T=T, learning_rate=1.0)
        ada.fit(X_tr_sm0, y_tr_sm0)

        tr_err = (ada.predict(X_tr_sm0) != y_tr_sm0).mean()
        y_te_pred = ada.predict(X_te0)
        y_te_prob = ada.predict_proba(X_te0)
        te_err = (y_te_pred != y_te0).mean()
        m_te = compute_metrics(y_te0, y_te_pred, y_te_prob)

        train_errors.append(tr_err)
        test_errors.append(te_err)
        f1_scores.append(m_te['f1'])

        print(f"  {T:>5} | {tr_err:>10.4f} {te_err:>10.4f} {m_te['f1']:>9.4f}")

    visualize.plot_adaboost_rounds(
        T_vals, train_errors, test_errors, f1_scores,
        path='output/section4/adaboost_rounds.png')

    # ==== Section 4.4: Full comparison ========================================
    print("\n Section 4.4 - Full Method Comparison (best configs from each task)")

    from task1_hddt import run_10 as hddt_run10
    from task2_bagging import run_experiment as bag_run_experiment

    hddt_m, _ = hddt_run10(max_depth=3, label="(Final)")

    bag_hddt_m, _ = bag_run_experiment([51], base_learner='hddt')[51]
    bag_dt_m,   _ = bag_run_experiment([51], base_learner='dt')[51]

    ada_m,    _ = run_10(T=50, use_smote=False, label="(Final)")
    ada_sm_m, _ = run_10(T=50, use_smote=True,  label="(Final)")

    final = {
        'HDDT (depth=3)':        ('max_depth=3', hddt_m),
        'Bagging+HDDT (T=51)':   ('T=51',        bag_hddt_m),
        'Bagging+DT (T=51)':     ('T=51',        bag_dt_m),
        'AdaBoost (T=50)':       ('T=50',        ada_m),
        'AdaBoost+SMOTE (T=50)': ('T=50',        ada_sm_m),
    }

    visualize.plot_final_comparison(
        {k: {'f1': v['f1'], 'gmean': v['gmean'], 'auc_roc': v['auc_roc']}
         for k, (_, v) in final.items()},
        path='output/section4/final_comparison.png')

    print(f"\n  {'Method':<28} {'Config':<14} {'F1(+1)':>8} "
          f"{'G-mean':>8} {'AUC-ROC':>9}")
    print(f"  {'-' * 70}")
    for method, (config, m) in final.items():
        print(f"  {method:<28} {config:<14} "
              f"{m['f1']:>8.4f} {m['gmean']:>8.4f} {m['auc_roc']:>9.4f}")

    print(f"\n  Task 3 complete. Plots saved to output/section4/\n")


if __name__ == '__main__':
    main()
