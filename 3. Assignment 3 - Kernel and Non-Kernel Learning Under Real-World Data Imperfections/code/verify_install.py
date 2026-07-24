"""Confirm the installed code is one consistent generation, not a mix.

Run from the folder containing main.py:   python verify_install.py
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

REQUIRED = {
    'utils.kernels': ['compute_kernel_matrix', 'center_kernel_matrix',
                      'kernel_diagonal'],
    'utils.metrics': ['classification_report', 'Timer', 'compute_auc',
                      'roc_auc_binary', 'kernel_matrix_memory'],
    'utils.preprocessing': ['StandardScaler', 'train_test_split',
                            'smote_oversample', 'k_fold_indices',
                            'cross_val_score'],
    'section2.data_quality': ['run_full_investigation',
                              'compute_isolation_forest_scores'],
    'section3.kernel_svm': ['KernelSVM', 'KernelSVMClassifier'],
    'section3.kpca': ['KernelPCA', 'KPCAClassifier'],
    'section3.knn': ['KNNClassifier'],
    'section3.decision_tree': ['DecisionTreeClassifier'],
    'section3.linear_regression': ['LinearRegression'],
}


def main():
    stale = [os.path.join(d, '__pycache__') for d, ds, _ in os.walk(ROOT)
             if '__pycache__' in ds]
    if stale:
        print(f"Clearing {len(stale)} stale __pycache__ folder(s)...")
        for p in stale:
            shutil.rmtree(p, ignore_errors=True)

    failures = []
    for mod, names in REQUIRED.items():
        try:
            m = __import__(mod, fromlist=names)
        except Exception as e:
            failures.append(f"{mod}: import failed -> {e}")
            print(f"  FAIL  {mod:<28} {e}")
            continue
        missing = [n for n in names if not hasattr(m, n)]
        if missing:
            failures.append(f"{mod}: missing {missing}")
            print(f"  FAIL  {mod:<28} missing {missing}")
        else:
            print(f"  ok    {mod:<28} {len(names)} symbol(s)")

    for pkg in ['utils', 'section1', 'section2', 'section3']:
        try:
            m = __import__(pkg)
            bad = [n for n in getattr(m, '__all__', []) if not hasattr(m, n)]
            if bad:
                failures.append(f"{pkg}.__all__: unresolved {bad}")
                print(f"  FAIL  {pkg:<28} __all__ unresolved: {bad}")
            else:
                print(f"  ok    {pkg:<28} __all__ = {len(m.__all__)} exports")
        except Exception as e:
            failures.append(f"{pkg}: {e}")
            print(f"  FAIL  {pkg:<28} {e}")

    print()
    if failures:
        print("MIXED INSTALL DETECTED. Delete src/ and main.py, then re-copy")
        print("BOTH from the zip. Do not copy individual files.")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("INSTALL CONSISTENT: all modules are the same generation.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
