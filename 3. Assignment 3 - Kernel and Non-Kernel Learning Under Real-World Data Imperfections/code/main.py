import traceback
import warnings
import time
import os
import numpy as np
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# ---- Path Setup ----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))


from visualize import (plot_missing_data, plot_outlier_comparison,  # noqa: E402
                       plot_feature_distributions, plot_correlation_matrix,  # noqa: E402
                       plot_confusion_matrix, plot_regression_scatter,  # noqa: E402
                       plot_model_comparison, plot_kernel_comparison,  # noqa: E402
                       plot_computational_analysis, plot_worst_predictions)  # noqa: E402
from utils.metrics import (classification_report, regression_report,  # noqa: E402
                           confusion_matrix as compute_cm, Timer)  # noqa: E402
from utils.preprocessing import StandardScaler, train_test_split, smote_oversample  # noqa: E402
from section3.kpca import KPCAClassifier  # noqa: E402
from section3.kernel_knn import KernelKNNClassifier, KernelKNNRegressor  # noqa: E402
from section3.kernel_svm import KernelSVMClassifier  # noqa: E402
from section3.kernel_ridge import KernelRidgeRegression, KernelRidgeClassifier  # noqa: E402
from section3.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402
from section3.knn import KNNClassifier, KNNRegressor  # noqa: E402
from section3.logistic_regression import LogisticRegression  # noqa: E402
from section3.linear_regression import LinearRegression  # noqa: E402
from section2.data_quality import run_full_investigation  # noqa: E402
from section1 import airbnb, nyc311, ibm_hr, online_retail  # noqa: E402


warnings.filterwarnings('ignore')

# ---- Configuration ----
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

MAX_SAMPLES_STANDARD = 3000
MAX_SAMPLES_KERNEL = 300


def subsample_if_needed(X, y, max_n, random_state=42):
    if len(y) <= max_n:
        return X, y
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(y), size=max_n, replace=False)
    return X[idx], y[idx]


def run_classification_models(X_train, y_train, X_test, y_test,
                              dataset_name, output_dir, timings):
    results = {}

    # ---- Logistic Regression ----
    print(f"\n  >> Logistic Regression ({dataset_name})")
    with Timer(f"LogReg train - {dataset_name}") as t:
        model = LogisticRegression(lr=0.05, n_iters=300, reg_lambda=0.01)
        model.fit(X_train, y_train)
    timings[dataset_name]['LogisticReg'] = t.elapsed
    y_pred = model.predict(X_test)
    results['LogisticReg'] = classification_report(
        y_test, y_pred, dataset_name, "Logistic Regression")
    cm, labels = compute_cm(y_test, y_pred)
    plot_confusion_matrix(cm, labels, dataset_name, 'LogisticReg', output_dir)

    # ---- KNN ----
    print(f"\n  >> KNN Classifier ({dataset_name})")
    with Timer(f"KNN train - {dataset_name}") as t:
        model = KNNClassifier(k=5)
        model.fit(X_train, y_train)
    timings[dataset_name]['KNN'] = t.elapsed
    y_pred = model.predict(X_test)
    results['KNN'] = classification_report(y_test, y_pred, dataset_name, "KNN")
    cm, labels = compute_cm(y_test, y_pred)
    plot_confusion_matrix(cm, labels, dataset_name, 'KNN', output_dir)

    # ---- Decision Tree ----
    print(f"\n  >> Decision Tree Classifier ({dataset_name})")
    with Timer(f"DTree train - {dataset_name}") as t:
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
        model.fit(X_train, y_train)
    timings[dataset_name]['DecisionTree'] = t.elapsed
    y_pred = model.predict(X_test)
    results['DecisionTree'] = classification_report(
        y_test, y_pred, dataset_name, "Decision Tree")
    cm, labels = compute_cm(y_test, y_pred)
    plot_confusion_matrix(cm, labels, dataset_name, 'DecisionTree', output_dir)

    return results


def run_regression_models(X_train, y_train, X_test, y_test,
                          dataset_name, output_dir, timings):
    results = {}

    # ---- Linear Regression (Ridge) ----
    print(f"\n  >> Linear Regression ({dataset_name})")
    with Timer(f"LinReg train - {dataset_name}") as t:
        model = LinearRegression(reg_lambda=1.0)
        model.fit(X_train, y_train)
    timings[dataset_name]['LinearReg'] = t.elapsed
    y_pred = model.predict(X_test)
    results['LinearReg'] = regression_report(
        y_test, y_pred, dataset_name, "Linear Regression")
    plot_regression_scatter(y_test, y_pred, dataset_name,
                            'LinearReg', output_dir)

    # ---- KNN Regressor ----
    print(f"\n  >> KNN Regressor ({dataset_name})")
    with Timer(f"KNN_Reg train - {dataset_name}") as t:
        model = KNNRegressor(k=5)
        model.fit(X_train, y_train)
    timings[dataset_name]['KNN_Reg'] = t.elapsed
    y_pred = model.predict(X_test)
    results['KNN_Reg'] = regression_report(
        y_test, y_pred, dataset_name, "KNN Regressor")
    plot_regression_scatter(
        y_test, y_pred, dataset_name, 'KNN_Reg', output_dir)

    # ---- Decision Tree Regressor ----
    print(f"\n  >> Decision Tree Regressor ({dataset_name})")
    with Timer(f"DTree_Reg train - {dataset_name}") as t:
        model = DecisionTreeRegressor(max_depth=10, min_samples_split=5)
        model.fit(X_train, y_train)
    timings[dataset_name]['DTree_Reg'] = t.elapsed
    y_pred = model.predict(X_test)
    results['DTree_Reg'] = regression_report(
        y_test, y_pred, dataset_name, "Decision Tree Regressor")
    plot_regression_scatter(y_test, y_pred, dataset_name,
                            'DTree_Reg', output_dir)

    return results


def run_kernel_classification(X_train, y_train, X_test, y_test,
                              dataset_name, output_dir, timings):
    results = {}
    kernel_comparison = {}

    for kernel_type in ['linear', 'poly', 'rbf']:
        kernel_comparison[kernel_type] = {}
        gamma = 1.0 / (2.0 * X_train.shape[1]
                       ) if kernel_type == 'rbf' else None
        kp = {}
        if kernel_type == 'rbf':
            kp['gamma'] = gamma
        elif kernel_type == 'poly':
            kp['degree'] = 3
            kp['c'] = 1.0

        # ---- Kernel SVM ----
        model_name = f'KernelSVM_{kernel_type}'
        print(f"\n  >> Kernel SVM ({kernel_type}) ({dataset_name})")
        try:
            with Timer(f"{model_name} - {dataset_name}") as t:
                # Comment out the next line for more accurate results. Poly SVM took ~2hr on my rig without it. The Report was generated with the 2 hour run.
                svm_max_iter = 10 if kernel_type == 'poly' else 30
                model = KernelSVMClassifier(C=1.0, kernel_type=kernel_type,
                                            max_iter=svm_max_iter, **kp)
                model.fit(X_train, y_train)
            timings[dataset_name][model_name] = t.elapsed
            y_pred = model.predict(X_test)
            res = classification_report(
                y_test, y_pred, dataset_name, model_name)
            results[model_name] = res
            kernel_comparison[kernel_type]['KernelSVM'] = res['accuracy']
        except Exception as e:
            print(f"    ERROR: {e}")
            results[model_name] = {'accuracy': 0.0}
            kernel_comparison[kernel_type]['KernelSVM'] = 0.0

        # ---- Kernel Ridge Classifier ----
        model_name = f'KRR_clf_{kernel_type}'
        print(
            f"\n  >> Kernel Ridge Classifier ({kernel_type}) ({dataset_name})")
        try:
            with Timer(f"{model_name} - {dataset_name}") as t:
                model = KernelRidgeClassifier(kernel_type=kernel_type,
                                              reg_lambda=1.0, **kp)
                model.fit(X_train, y_train)
            timings[dataset_name][model_name] = t.elapsed
            y_pred = model.predict(X_test)
            res = classification_report(
                y_test, y_pred, dataset_name, model_name)
            results[model_name] = res
            kernel_comparison[kernel_type]['KRR'] = res['accuracy']
        except Exception as e:
            print(f"    ERROR: {e}")
            results[model_name] = {'accuracy': 0.0}
            kernel_comparison[kernel_type]['KRR'] = 0.0

        # ---- Kernel KNN ----
        model_name = f'KernelKNN_{kernel_type}'
        print(f"\n  >> Kernel KNN ({kernel_type}) ({dataset_name})")
        try:
            with Timer(f"{model_name} - {dataset_name}") as t:
                model = KernelKNNClassifier(k=5, kernel_type=kernel_type, **kp)
                model.fit(X_train, y_train)
            timings[dataset_name][model_name] = t.elapsed
            y_pred = model.predict(X_test)
            res = classification_report(
                y_test, y_pred, dataset_name, model_name)
            results[model_name] = res
            kernel_comparison[kernel_type]['KernelKNN'] = res['accuracy']
        except Exception as e:
            print(f"    ERROR: {e}")
            results[model_name] = {'accuracy': 0.0}
            kernel_comparison[kernel_type]['KernelKNN'] = 0.0

    # ---- KPCA + Logistic Regression (RBF only) ----
    print(f"\n  >> KPCA + LogReg ({dataset_name})")
    try:
        n_comp = min(50, X_train.shape[0] - 1, X_train.shape[1])
        gamma_kpca = 1.0 / (2.0 * X_train.shape[1])
        with Timer(f"KPCA_LogReg - {dataset_name}") as t:
            model = KPCAClassifier(n_components=n_comp, kernel_type='rbf',
                                   gamma=gamma_kpca, lr=0.05, n_iters=300)
            model.fit(X_train, y_train)
        timings[dataset_name]['KPCA_LogReg'] = t.elapsed
        y_pred = model.predict(X_test)
        res = classification_report(
            y_test, y_pred, dataset_name, "KPCA+LogReg")
        results['KPCA_LogReg'] = res
        kernel_comparison.setdefault(
            'rbf', {})['KPCA_LogReg'] = res['accuracy']
    except Exception as e:
        print(f"    ERROR: {e}")
        traceback.print_exc()
        results['KPCA_LogReg'] = {'accuracy': 0.0}

    plot_kernel_comparison(kernel_comparison, dataset_name, output_dir)
    return results, kernel_comparison


def run_kernel_regression(X_train, y_train, X_test, y_test,
                          dataset_name, output_dir, timings):
    results = {}

    for kernel_type in ['linear', 'poly', 'rbf']:
        kp = {}
        if kernel_type == 'rbf':
            kp['gamma'] = 1.0 / X_train.shape[1]
        elif kernel_type == 'poly':
            kp['degree'] = 3
            kp['c'] = 1.0

        # ---- Kernel Ridge Regression ----
        model_name = f'KRR_reg_{kernel_type}'
        print(
            f"\n  >> Kernel Ridge Regression ({kernel_type}) ({dataset_name})")
        try:
            with Timer(f"{model_name} - {dataset_name}") as t:
                model = KernelRidgeRegression(kernel_type=kernel_type,
                                              reg_lambda=1.0, **kp)
                model.fit(X_train, y_train)
            timings[dataset_name][model_name] = t.elapsed
            y_pred = model.predict(X_test)
            res = regression_report(y_test, y_pred, dataset_name, model_name)
            results[model_name] = res
            plot_regression_scatter(
                y_test, y_pred, dataset_name, model_name, output_dir)
        except Exception as e:
            print(f"    ERROR: {e}")
            results[model_name] = {'r2': 0.0, 'rmse': float('inf')}

        # ---- Kernel KNN Regressor ----
        model_name = f'KernelKNN_reg_{kernel_type}'
        print(f"\n  >> Kernel KNN Regressor ({kernel_type}) ({dataset_name})")
        try:
            with Timer(f"{model_name} - {dataset_name}") as t:
                model = KernelKNNRegressor(k=5, kernel_type=kernel_type, **kp)
                model.fit(X_train, y_train)
            timings[dataset_name][model_name] = t.elapsed
            y_pred = model.predict(X_test)
            res = regression_report(y_test, y_pred, dataset_name, model_name)
            results[model_name] = res
        except Exception as e:
            print(f"    ERROR: {e}")
            results[model_name] = {'r2': 0.0, 'rmse': float('inf')}

    return results


def analyze_worst_predictions(y_test, y_pred, dataset_name, model_name, output_dir, task='classification'):
    print(f"\n  --- Failure Analysis: {model_name} on {dataset_name} ---")

    if task == 'regression':
        errors = np.abs(y_test - y_pred)
    else:
        errors = np.where(y_test != y_pred, 1.0, 0.0)

    worst_indices = np.argsort(errors)[-10:][::-1]

    print(f"    10 worst prediction indices: {worst_indices.tolist()}")
    for rank, idx in enumerate(worst_indices):
        if task == 'regression':
            print(f"      #{rank+1}: idx={idx}, actual={y_test[idx]:.4f}, "
                  f"predicted={y_pred[idx]:.4f}, error={errors[idx]:.4f}")
        else:
            print(f"      #{rank+1}: idx={idx}, actual={y_test[idx]}, "
                  f"predicted={y_pred[idx]}")

    plot_worst_predictions(y_test, y_pred, worst_indices, dataset_name, model_name,
                           output_dir, task)
    return worst_indices


def process_dataset(data_dict, output_base, timings, all_results):
    dataset_name = data_dict['dataset_name']
    out_s1 = os.path.join(output_base, 'section1')
    out_s2 = os.path.join(output_base, 'section2')
    out_s3 = os.path.join(output_base, 'section3')
    out_s4 = os.path.join(output_base, 'section4')
    out_s5 = os.path.join(output_base, 'section5')

    timings.setdefault(dataset_name, {})

    # ---- SECTION 2: Data Quality Investigation ----
    print(f"\n{'#' * 70}")
    print(f"# SECTION 2: Data Quality Investigation - {dataset_name}")
    print(f"{'#' * 70}")

    dq_results = run_full_investigation(data_dict, out_s2)
    if 'missing' in dq_results and dq_results['missing'].get('columns'):
        plot_missing_data(dq_results['missing'], dataset_name, out_s2)
    if 'outliers' in dq_results:
        plot_outlier_comparison(dq_results['outliers'], dataset_name, out_s2)

    if data_dict.get('clean_df') is not None and data_dict.get('feature_names'):
        plot_feature_distributions(data_dict['clean_df'], data_dict['feature_names'],
                                   dataset_name, out_s1)
        plot_correlation_matrix(data_dict['clean_df'], data_dict['feature_names'],
                                dataset_name, out_s1)

    # ---- SECTION 3: Non-Kernel Models ----
    print(f"\n{'#' * 70}")
    print(f"# SECTION 3: Non-Kernel Models - {dataset_name}")
    print(f"{'#' * 70}")

    clf_results = {}
    reg_results = {}

    # ---- Classification ----
    if data_dict.get('X_clf') is not None and data_dict.get('y_clf') is not None:
        X, y = data_dict['X_clf'], data_dict['y_clf']
        print(
            f"\n  Classification data: X={X.shape}, classes={len(np.unique(y))}")

        X_s, y_s = subsample_if_needed(X, y, MAX_SAMPLES_STANDARD)

        # NOTE: the train/test split MUST happen before any resampling.
        # SMOTE synthesises minority points by interpolating between a real
        # point and its k nearest neighbours. If it runs on the full dataset
        # first, a synthetic point in the training fold can be built from a
        # real point that later lands in the test fold, and the test set is
        # itself half synthetic. Both leak. Split first, resample train only.
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.2, random_state=42, stratify=True
        )
        print(f"  Train: {X_train.shape}, Test: {X_test.shape} "
              f"(split before resampling)")

        if dataset_name == 'IBM_HR':
            print(f"  Applying SMOTE oversampling to the TRAINING fold only...")
            X_train, y_train = smote_oversample(
                X_train, y_train, random_state=42)
            print(
                f"  After SMOTE: train={X_train.shape}, " f"distribution={({int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))})}")
            print(
                f"  Test fold left at natural prevalence: " f"distribution={({int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))})}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf_results = run_classification_models(
            X_train_scaled, y_train, X_test_scaled, y_test, dataset_name, out_s3, timings
        )

        # ---- Kernel Methods (smaller sample) ----
        print(f"\n{'#' * 70}")
        print(f"# SECTION 3+4: Kernel Models - {dataset_name}")
        print(f"{'#' * 70}")

        # The kernel block re-subsamples from X_s / y_s, which are now the
        # ORIGINAL (un-resampled) arrays. It therefore needs its own
        # split-then-SMOTE, for the same reason as above.
        X_k, y_k = subsample_if_needed(X_s, y_s, MAX_SAMPLES_KERNEL)
        X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
            X_k, y_k, test_size=0.2, random_state=42, stratify=True
        )

        if dataset_name == 'IBM_HR':
            print(f"  Applying SMOTE to the kernel TRAINING fold only...")
            X_train_k, y_train_k = smote_oversample(
                X_train_k, y_train_k, random_state=42)
            print(
                f"  After SMOTE: kernel train={X_train_k.shape}, " f"distribution={({int(k): int(v) for k, v in zip(*np.unique(y_train_k, return_counts=True))})}")

        scaler_k = StandardScaler()
        X_train_ks = scaler_k.fit_transform(X_train_k)
        X_test_ks = scaler_k.transform(X_test_k)
        print(f"  Kernel train: {X_train_ks.shape}, test: {X_test_ks.shape}")
        print(
            f"  Kernel test distribution: " f"{({int(k): int(v) for k, v in zip(*np.unique(y_test_k, return_counts=True))})}")

        kernel_clf_results, kernel_comparison = run_kernel_classification(
            X_train_ks, y_train_k, X_test_ks, y_test_k, dataset_name, out_s4, timings
        )
        clf_results.update(kernel_clf_results)

        # ---- Model Comparison Plot ----
        acc_dict = {k: v.get('accuracy', 0)
                    for k, v in clf_results.items() if 'accuracy' in v}
        if acc_dict:
            plot_model_comparison(acc_dict, 'Accuracy',
                                  dataset_name, out_s3, 'classification')

        # ---- Failure Analysis (Section 5) ----
        best_model_name = max(
            {k: v for k, v in clf_results.items() if 'accuracy' in v},
            key=lambda k: clf_results[k]['accuracy'], default=None
        )
        if best_model_name:
            print(f"\n  Failure analysis using {best_model_name}...")
            if best_model_name == 'LogisticReg':
                model = LogisticRegression(
                    lr=0.05, n_iters=300, reg_lambda=0.01)
                model.fit(X_train_scaled, y_train)
                y_pred_fa = model.predict(X_test_scaled)
            elif best_model_name == 'KNN':
                model = KNNClassifier(k=5)
                model.fit(X_train_scaled, y_train)
                y_pred_fa = model.predict(X_test_scaled)
            else:
                model = DecisionTreeClassifier(max_depth=10)
                model.fit(X_train_scaled, y_train)
                y_pred_fa = model.predict(X_test_scaled)
            analyze_worst_predictions(y_test, y_pred_fa, dataset_name,
                                      best_model_name, out_s5, 'classification')

    # ---- Regression ----
    if data_dict.get('X_reg') is not None and data_dict.get('y_reg') is not None:
        X, y = data_dict['X_reg'], data_dict['y_reg']
        print(f"\n  Regression data: X={X.shape}, target mean={y.mean():.4f}")

        X_s, y_s = subsample_if_needed(X, y, MAX_SAMPLES_STANDARD)
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(
            f"  Regression Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

        reg_results = run_regression_models(
            X_train_scaled, y_train, X_test_scaled, y_test, dataset_name, out_s3, timings
        )

        X_k, y_k = subsample_if_needed(X_s, y_s, MAX_SAMPLES_KERNEL)
        X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
            X_k, y_k, test_size=0.2, random_state=42
        )
        scaler_k = StandardScaler()
        X_train_ks = scaler_k.fit_transform(X_train_k)
        X_test_ks = scaler_k.transform(X_test_k)

        kernel_reg = run_kernel_regression(
            X_train_ks, y_train_k, X_test_ks, y_test_k, dataset_name, out_s4, timings
        )
        reg_results.update(kernel_reg)

        r2_dict = {k: v.get('r2', 0)
                   for k, v in reg_results.items() if 'r2' in v}
        if r2_dict:
            plot_model_comparison(r2_dict, 'R_squared',
                                  dataset_name, out_s3, 'regression')

        model = LinearRegression(reg_lambda=1.0)
        model.fit(X_train_scaled, y_train)
        y_pred_fa = model.predict(X_test_scaled)
        analyze_worst_predictions(y_test, y_pred_fa, dataset_name,
                                  'LinearReg', out_s5, 'regression')

    all_results[dataset_name] = {
        'classification': clf_results,
        'regression': reg_results,
        'data_quality': dq_results
    }


def print_summary(all_results, timings):
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    for ds_name, ds_results in all_results.items():
        print(f"\n{'-' * 60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'-' * 60}")

        clf = ds_results.get('classification', {})
        if clf:
            print(f"\n  === CLASSIFICATION RESULTS ===")
            print(f"  {'Model':<25} {'Accuracy':>10} {'Prec(M)':>10} {'Rec(M)':>10} "
                  f"{'F1(M)':>10} {'Prec(W)':>10} {'Rec(W)':>10} {'F1(W)':>10}")
            print(
                f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for model_name, metrics in sorted(clf.items(),
                                              key=lambda x: x[1].get('accuracy', 0), reverse=True):
                if 'accuracy' in metrics:
                    print(f"  {model_name:<25} "
                          f"{metrics['accuracy']:>10.4f} "
                          f"{metrics.get('precision_macro', 0):>10.4f} "
                          f"{metrics.get('recall_macro', 0):>10.4f} "
                          f"{metrics.get('f1_macro', 0):>10.4f} "
                          f"{metrics.get('precision_weighted', 0):>10.4f} "
                          f"{metrics.get('recall_weighted', 0):>10.4f} "
                          f"{metrics.get('f1_weighted', 0):>10.4f}")

        reg = ds_results.get('regression', {})
        if reg:
            print(f"\n  === REGRESSION RESULTS ===")
            print(
                f"  {'Model':<25} {'R2':>10} {'MSE':>14} {'RMSE':>12} {'MAE':>12}")
            print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*12} {'-'*12}")
            for model_name, metrics in sorted(reg.items(),
                                              key=lambda x: x[1].get('r2', -999), reverse=True):
                if 'r2' in metrics:
                    print(f"  {model_name:<25} "
                          f"{metrics['r2']:>10.4f} "
                          f"{metrics.get('mse', 0):>14.4f} "
                          f"{metrics.get('rmse', 0):>12.4f} "
                          f"{metrics.get('mae', 0):>12.4f}")

    # ---- Timing Summary ----
    print(f"\n{'=' * 60}")
    print(f"  COMPUTATIONAL ANALYSIS (Training Times in seconds)")
    print(f"{'=' * 60}")
    for ds_name, ds_timings in timings.items():
        print(f"\n  {ds_name}:")
        total_time = sum(ds_timings.values())
        for model_name, elapsed in sorted(ds_timings.items(), key=lambda x: x[1], reverse=True):
            print(f"    {model_name:<30} {elapsed:>10.4f}s")
        print(f"    {'TOTAL':<30} {total_time:>10.4f}s")


def main():
    total_start = time.time()

    print("=" * 80)
    print("HOMEWORK 3: KERNEL AND NON-KERNEL LEARNING")
    print("UNDER REAL-WORLD DATA IMPERFECTIONS")
    print("=" * 80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in ['section1', 'section2', 'section3', 'section4', 'section5']:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    all_results = {}
    timings = {}

    # ---- SECTION 1: Data Engineering ----
    print(f"\n{'#' * 70}")
    print(f"# SECTION 1: DATA LOADING & ENGINEERING")
    print(f"{'#' * 70}")

    datasets = {}

    try:
        datasets['Airbnb'] = airbnb.load_and_engineer(DATA_DIR, os.path.join(OUTPUT_DIR, 'section1'),
                                                      max_rows=None)
    except Exception as e:
        print(f"  ERROR loading Airbnb: {e}")
        traceback.print_exc()

    try:
        datasets['NYC_311'] = nyc311.load_and_engineer(DATA_DIR, os.path.join(OUTPUT_DIR, 'section1'),
                                                       max_rows=80000)
    except Exception as e:
        print(f"  ERROR loading NYC 311: {e}")
        traceback.print_exc()

    try:
        datasets['IBM_HR'] = ibm_hr.load_and_engineer(
            DATA_DIR, os.path.join(OUTPUT_DIR, 'section1'))
    except Exception as e:
        print(f"  ERROR loading IBM HR: {e}")
        traceback.print_exc()

    try:
        datasets['OnlineRetail'] = online_retail.load_and_engineer(
            DATA_DIR, os.path.join(OUTPUT_DIR, 'section1'), max_rows=200000)
    except Exception as e:
        print(f"  ERROR loading Online Retail: {e}")
        traceback.print_exc()

    # ---- SECTIONS 2-5: Process Each Dataset ----
    for ds_name, data_dict in datasets.items():
        print(f"\n{'*' * 70}")
        print(f"* PROCESSING DATASET: {ds_name}")
        print(f"{'*' * 70}")
        try:
            process_dataset(data_dict, OUTPUT_DIR, timings, all_results)
        except Exception as e:
            print(f"  CRITICAL ERROR processing {ds_name}: {e}")
            traceback.print_exc()

    # ---- SECTION 5: Cross-Dataset Analysis ----
    print(f"\n{'#' * 70}")
    print(f"# SECTION 5: CROSS-DATASET COMPARISON & DISCUSSION")
    print(f"{'#' * 70}")

    if timings:
        plot_computational_analysis(
            timings, os.path.join(OUTPUT_DIR, 'section5'))

    print_summary(all_results, timings)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(
        f"TOTAL EXECUTION TIME: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()