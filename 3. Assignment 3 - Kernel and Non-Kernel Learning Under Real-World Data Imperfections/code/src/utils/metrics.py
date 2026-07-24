import numpy as np
import time
import tracemalloc
from scipy.stats import rankdata


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def precision_recall_f1(y_true, y_pred, average='macro'):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm, _ = confusion_matrix(y_true, y_pred, labels)

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        supports.append(support)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports, dtype=float)

    if average == 'macro':
        return precisions.mean(), recalls.mean(), f1s.mean()
    elif average == 'weighted':
        total = supports.sum()
        if total == 0:
            return 0.0, 0.0, 0.0
        weights = supports / total
        return (precisions * weights).sum(), (recalls * weights).sum(), (f1s * weights).sum()
    elif average == 'per_class':
        return precisions, recalls, f1s, supports, labels
    else:
        return precisions, recalls, f1s


def roc_auc_binary(y_true, y_score, positive_label=1):
    """ROC-AUC for a binary target, via the Mann-Whitney U identity:

        AUC = (sum(ranks of positives) - n_pos(n_pos+1)/2) / (n_pos * n_neg)

    rankdata gives mid-ranks to ties, i.e. the 0.5 credit a tie deserves.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    pos = (y_true == positive_label)
    n_pos = int(pos.sum())
    n_neg = int(len(y_true) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float('nan')

    ranks = rankdata(y_score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def roc_auc_ovr_macro(y_true, score_matrix, classes):
    """Macro-averaged one-vs-rest ROC-AUC for multi-class problems."""
    y_true = np.asarray(y_true)
    aucs = []
    for i, c in enumerate(classes):
        a = roc_auc_binary((y_true == c).astype(int), score_matrix[:, i], 1)
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float('nan')


def compute_auc(y_true, score_matrix, classes):
    """Dispatch to binary or macro-OvR AUC based on class count."""
    classes = np.asarray(classes)
    if len(classes) == 2:
        return roc_auc_binary((np.asarray(y_true) == classes[1]).astype(int),
                              score_matrix[:, 1], 1)
    return roc_auc_ovr_macro(y_true, score_matrix, classes)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def regression_report(y_true, y_pred, dataset_name="", model_name=""):
    m = mse(y_true, y_pred)
    rm = rmse(y_true, y_pred)
    ma = mae(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    y_var = np.var(y_true)
    y_mean = np.mean(y_true)
    y_std = np.std(y_true)
    pred_mean = np.mean(y_pred)
    pred_std = np.std(y_pred)
    n_samples = len(y_true)

    print(f"  [{dataset_name}] {model_name} Regression Metrics:")
    print(f"    Num test samples:   {n_samples}")
    print(f"    Target mean:        {y_mean:.6f}")
    print(f"    Target std:         {y_std:.6f}")
    print(f"    Prediction mean:    {pred_mean:.6f}")
    print(f"    Prediction std:     {pred_std:.6f}")
    print(f"    MSE:                {m:.6f}")
    print(f"    RMSE:               {rm:.6f}")
    print(f"    MAE:                {ma:.6f}")
    print(f"    R-squared (R2):     {r2:.6f}")
    print(f"    Target variance:    {y_var:.6f}")

    return {'mse': m, 'rmse': rm, 'mae': ma, 'r2': r2,
            'n_samples': n_samples, 'y_mean': y_mean, 'y_std': y_std,
            'pred_mean': pred_mean, 'pred_std': pred_std}


def classification_report(y_true, y_pred, dataset_name="", model_name="",
                          y_score=None, classes=None):
    acc = accuracy(y_true, y_pred)
    p_macro, r_macro, f1_macro = precision_recall_f1(y_true, y_pred, 'macro')
    p_w, r_w, f1_w = precision_recall_f1(y_true, y_pred, 'weighted')
    per_p, per_r, per_f1, per_sup, labels = precision_recall_f1(y_true, y_pred, 'per_class')
    cm, cm_labels = confusion_matrix(y_true, y_pred)
    n_samples = len(y_true)
    n_correct = int(np.sum(y_true == y_pred))
    n_wrong = n_samples - n_correct

    print(f"  [{dataset_name}] {model_name} Classification Metrics:")
    print(f"    Num test samples:       {n_samples}")
    print(f"    Correct predictions:    {n_correct}")
    print(f"    Wrong predictions:      {n_wrong}")
    print(f"    Accuracy:               {acc:.6f}")
    print(f"    Precision (macro):      {p_macro:.6f}")
    print(f"    Recall (macro):         {r_macro:.6f}")
    print(f"    F1-score (macro):       {f1_macro:.6f}")
    print(f"    Precision (weighted):   {p_w:.6f}")
    print(f"    Recall (weighted):      {r_w:.6f}")
    print(f"    F1-score (weighted):    {f1_w:.6f}")

    auc = float('nan')
    if y_score is not None and classes is not None:
        try:
            auc = compute_auc(y_true, y_score, classes)
        except Exception:
            auc = float('nan')
        if not np.isnan(auc):
            tag = "binary" if len(np.asarray(classes)) == 2 else "macro OvR"
            print(f"    ROC-AUC ({tag}):{'':<8}{auc:.6f}")
    else:
        print(f"    ROC-AUC:                not available (model exposes no scores)")

    print(f"    --- Per-Class Breakdown ---")
    print(f"    {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for i, lbl in enumerate(labels):
        lbl_str = str(lbl)[:12]
        print(f"    {lbl_str:<12} {per_p[i]:>10.4f} {per_r[i]:>10.4f} {per_f1[i]:>10.4f} {int(per_sup[i]):>10}")

    print(f"    --- Confusion Matrix (rows=actual, cols=predicted) ---")
    header = "    {:>12}".format("")
    for lbl in cm_labels:
        header += " {:>8}".format(str(lbl)[:8])
    print(header)
    for i, lbl in enumerate(cm_labels):
        row_str = "    {:>12}".format(str(lbl)[:12])
        for j in range(len(cm_labels)):
            row_str += " {:>8d}".format(cm[i, j])
        print(row_str)

    return {
        'accuracy': acc, 'roc_auc': auc, 'n_samples': n_samples,
        'n_correct': n_correct, 'n_wrong': n_wrong,
        'precision_macro': p_macro, 'recall_macro': r_macro, 'f1_macro': f1_macro,
        'precision_weighted': p_w, 'recall_weighted': r_w, 'f1_weighted': f1_w,
        'per_class_precision': per_p.tolist(),
        'per_class_recall': per_r.tolist(),
        'per_class_f1': per_f1.tolist(),
        'per_class_support': per_sup.tolist(),
        'class_labels': [str(l) for l in labels],
        'confusion_matrix': cm.tolist()
    }


class Timer:
    """Wall-clock timer with optional peak-allocation tracing.

    trace_memory defaults to False: tracemalloc costs ~2x wall-clock and would
    corrupt the timings. Kernel-matrix memory is instead read for free from
    ndarray.nbytes (kernel_matrix_bytes_ on each kernel model).
    """

    def __init__(self, label="", trace_memory=False):
        self.label = label
        self.trace_memory = trace_memory
        self.elapsed = 0.0
        self.peak_mem_bytes = 0

    def __enter__(self):
        if self.trace_memory:
            tracemalloc.start()
            tracemalloc.reset_peak()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.trace_memory:
            _, peak = tracemalloc.get_traced_memory()
            self.peak_mem_bytes = int(peak)
            tracemalloc.stop()
            print(f"  [Timer] {self.label}: {self.elapsed:.4f} seconds, "
                  f"peak {self.peak_mem_bytes / 1e6:.2f} MB")
        else:
            print(f"  [Timer] {self.label}: {self.elapsed:.4f} seconds")


def kernel_matrix_memory(n, dtype_bytes=8):
    """Analytic memory of a dense n x n kernel matrix, in bytes."""
    return int(n) * int(n) * int(dtype_bytes)
