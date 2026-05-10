import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = 0.0

    # Confusion matrix with labels=[-1,1]:  [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    gmean = np.sqrt(sensitivity * specificity)

    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, auc_roc=auc, gmean=gmean)


def aggregate_runs(metrics_list):
    keys = metrics_list[0].keys()
    means = {k: np.mean([m[k] for m in metrics_list]) for k in keys}
    stds = {k: np.std([m[k] for m in metrics_list]) for k in keys}
    return means, stds


def print_metrics_table(means, stds, title="Results"):
    border = "-" * 54
    print(f"\n  +{border}+")
    print(f"  | {title:<52} |")
    print(f"  +{border}+")
    print(f"  | {'Metric':<22} {'Mean':>12}  {'Std Dev':>12}  |")
    print(f"  +{border}+")
    rows = [('accuracy', 'Accuracy'), ('precision', 'Precision (+1)'),
            ('recall',   'Recall (+1)'), ('f1', 'F1-score (+1)'),
            ('auc_roc',  'AUC-ROC'),   ('gmean', 'G-mean')]
    for key, label in rows:
        if key in means:
            print(
                f"  | {label:<22} {means[key]:>12.4f}  {stds[key]:>12.4f}  |")
    print(f"  +{border}+\n")
