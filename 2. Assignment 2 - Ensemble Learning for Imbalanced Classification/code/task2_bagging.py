import numpy as np
from sklearn.tree import DecisionTreeClassifier
from preprocessing import load_and_preprocess
from task1_hddt import HDDT
from utils import compute_metrics, aggregate_runs, print_metrics_table
import visualize
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Covid.csv"


class BaggingImbalanced:
    def __init__(self, base_learner='hddt', T=31, max_depth=3, min_samples_split=10, random_state=None):
        self.base_learner = base_learner
        self.T = T
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.estimators = []

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.estimators = []
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == -1)[0]
        
        n_minority = len(pos_indices)
        
        for _ in range(self.T):
            # Bootstrap minority (with replacement)
            pos_sample = np.random.choice(pos_indices, size=n_minority, replace=True)
            # Undersample majority (without replacement)
            neg_sample = np.random.choice(neg_indices, size=n_minority, replace=False)

            subset_indices = np.concatenate([pos_sample, neg_sample])
            np.random.shuffle(subset_indices)

            X_subset = X[subset_indices]
            y_subset = y[subset_indices]

            if self.base_learner == 'hddt':
                model = HDDT(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            elif self.base_learner == 'dt':
                model = DecisionTreeClassifier(
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    random_state=np.random.randint(10000)
                )
            else:
                raise ValueError("base_learner must be 'hddt' or 'dt'")
                
            model.fit(X_subset, y_subset)
            self.estimators.append(model)

        return self

    def predict(self, X):
        # Majority vote
        preds = np.zeros((X.shape[0], self.T))
        for i, model in enumerate(self.estimators):
            preds[:, i] = model.predict(X)

        # Sum predictions: positive if sum > 0 (majority are +1), else -1
        votes = np.sum(preds, axis=1)
        return np.where(votes >= 0, 1, -1)

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], self.T))
        for i, model in enumerate(self.estimators):
            if self.base_learner == 'hddt':
                probs[:, i] = model.predict_proba(X)
            else:
                probs[:, i] = model.predict_proba(X)[:, 1]

        return np.mean(probs, axis=1)


def run_experiment(T_values, base_learner='hddt', seeds=range(10)):
    results = {}
    
    for T in T_values:
        print(f"\nRunning Bagging (base={base_learner}, T={T})")
        T_metrics = []
        for seed in seeds:
            (X_tr, X_te, y_tr, y_te, *_) = load_and_preprocess(CSV_PATH, random_state=seed, verbose=False)

            model = BaggingImbalanced(base_learner=base_learner, T=T, max_depth=3, random_state=seed)
            model.fit(X_tr, y_tr)
            
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)
            
            m = compute_metrics(y_te, y_pred, y_proba)
            T_metrics.append(m)

        means, stds = aggregate_runs(T_metrics)
        results[T] = (means, stds)
        print(f"  Result: G-mean = {means['gmean']:.4f} ± {stds['gmean']:.4f}")
        
    return results


def main():
    print("\n" + "#"*65)
    print("#  TASK 2 - Bagging for Imbalanced Data")
    print("#"*65)

    # Experiment 1: Varying Ensemble Size T with HDDT
    T_values = [11, 31, 51, 101]
    print("\n--- Experiment A: Varying T with HDDT ---")
    results_hddt = run_experiment(T_values, base_learner='hddt')

    gmean_m = [results_hddt[T][0]['gmean'] for T in T_values]
    gmean_s = [results_hddt[T][1]['gmean'] for T in T_values]
    f1_m = [results_hddt[T][0]['f1'] for T in T_values]
    f1_s = [results_hddt[T][1]['f1'] for T in T_values]
    
    visualize.plot_bagging_experiment(
        T_values, gmean_m, gmean_s, f1_m, f1_s, 
        path='output/section3/bagging_T_experiment.png'
    )
    
    # Experiment 2: Comparing Base Learners at T=51
    print("\n--- Experiment B: HDDT vs scikit-learn DT (T=51) ---")
    results_dt = run_experiment([51], base_learner='dt')

    print("\nComparison Table (T=51):")
    print_metrics_table(results_hddt[51][0], results_hddt[51][1], title="Bagging with HDDT (T=51)")
    print_metrics_table(results_dt[51][0], results_dt[51][1], title="Bagging with Sklearn DT (T=51)")

if __name__ == '__main__':
    main()
