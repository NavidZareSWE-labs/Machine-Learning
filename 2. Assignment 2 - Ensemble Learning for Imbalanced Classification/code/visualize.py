import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] Saved -> {path}")


# ========= Section 1: dataset overview =========

def plot_missing_rates(missing_rates, path='output/section1/missing_rates.png'):
    nonzero = missing_rates[missing_rates > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['crimson' if v > 0.5 else 'steelblue' for v in nonzero.values]
    ax.bar(nonzero.index, nonzero.values * 100, color=colors,
           edgecolor='black', linewidth=0.5)
    ax.axhline(50, color='orange', linestyle='--', linewidth=1.5, label='50%')
    ax.axhline(82, color='red',    linestyle='--', linewidth=1.5, label='82%')
    ax.set_xlabel('Feature', fontsize=11)
    ax.set_ylabel('Missing Rate (%)', fontsize=11)
    ax.set_title('Missing Value Rates per Feature\n'
                 '(red bars exceed 50%; dashed lines at 50% and 82%)',
                 fontsize=12, fontweight='bold')
    plt.xticks(rotation=55, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_class_distribution(y, path='output/section1/class_distribution.png'):
    classes, counts = np.unique(y, return_counts=True)
    labels = ['+1 (Severe)' if c == 1 else '-1 (Non-severe)' for c in classes]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts, color=['#e74c3c', '#2980b9'],
                  edgecolor='black', linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10, str(cnt),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Class Distribution in COVID-19 Dataset\n'
                 '(Imbalance ratio ~15:1)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


# ========= Section 2: Task 1 HDDT =========

def plot_pruning_experiment(depths, gmean_means, gmean_stds,
                            f1_means, f1_stds,
                            path='output/section2/pruning_experiment.png'):
    """G-mean and F1 vs max_depth with error bars (mean +/- std, 10 runs)."""
    labels = ['None' if d is None else str(d) for d in depths]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, gmean_means, yerr=gmean_stds, marker='o',
                label='G-mean', capsize=5, linewidth=2, markersize=7,
                color='steelblue')
    ax.errorbar(x, f1_means,    yerr=f1_stds,    marker='s',
                label='F1 (+1)', capsize=5, linewidth=2, markersize=7,
                color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel('max_depth', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('HDDT Pruning: G-mean and F1 vs. max_depth\n'
                 '(mean +/- std over 10 runs)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, path)


# ========= Section 3: Task 2 Bagging =========

def plot_bagging_experiment(T_values, gmean_means, gmean_stds, f1_means, f1_stds, path='output/section3/bagging_T_experiment.png'):
    x = np.arange(len(T_values))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, gmean_means, yerr=gmean_stds, marker='o',
                label='G-mean', capsize=5, linewidth=2, color='steelblue')
    ax.errorbar(x, f1_means, yerr=f1_stds, marker='s',
                label='F1 (+1)', capsize=5, linewidth=2, color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in T_values], fontsize=11)
    ax.set_xlabel('Ensemble Size (T)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Bagging Imbalanced: G-mean and F1 vs. T\n(HDDT max_depth=3, 10 runs)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, path)


# ========= Section 4: Task 3 AdaBoost =========
def _print_comparison_table(rows, columns, title=""):
    if title:
        print(f"\n  {title}")

    headers = [col[0] for col in columns]
    col_w = [max(len(h), 8) for h in headers]
    label_w = max(len(r[0]) for r in rows) + 2

    header_line = f"  {'Method':<{label_w}}" + "".join(
        f"{h:>{w}}" for h, w in zip(headers, col_w))
    print(header_line)
    print(f"  {'-' * (label_w + sum(col_w))}")

    for name, m in rows:
        vals = "".join(f"{m[col[1]]:>{w}.4f}" for col,
                       w in zip(columns, col_w))
        print(f"  {name:<{label_w}}{vals}")


def _annotate_bars(ax, bars, fmt='.3f', fontsize=7.5):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f'{h:{fmt}}', ha='center', va='bottom', fontsize=fontsize)


def plot_adaboost_rounds(T_values, train_errors, test_errors, f1_scores,
                         path='output/section4/adaboost_rounds.png'):
    fig, (ax_err, ax_f1) = plt.subplots(1, 2, figsize=(13, 5),
                                        facecolor='white')
    common = dict(linewidth=2.2, markersize=7)

    # -- Train / Test error --
    ax_err.plot(T_values, train_errors, marker='o', color='steelblue',
                label='Train Error', **common)
    ax_err.plot(T_values, test_errors,  marker='s', color='tomato',
                label='Test Error',  **common)
    for t, tr, te in zip(T_values, train_errors, test_errors):
        ax_err.annotate(f'{tr:.3f}', (t, tr), textcoords='offset points',
                        xytext=(0, 9),   ha='center', fontsize=8, color='steelblue')
        ax_err.annotate(f'{te:.3f}', (t, te), textcoords='offset points',
                        xytext=(0, -13), ha='center', fontsize=8, color='tomato')
    ax_err.fill_between(T_values, train_errors, alpha=0.08, color='steelblue')
    ax_err.fill_between(T_values, test_errors,  alpha=0.08, color='tomato')
    ax_err.set_xlabel('Boosting Rounds T', fontsize=11)
    ax_err.set_ylabel('Error Rate',        fontsize=11)
    ax_err.set_title('AdaBoost + SMOTE:\nTrain / Test Error vs. T',
                     fontsize=12, fontweight='bold')
    ax_err.legend(fontsize=10, framealpha=0.9)
    ax_err.grid(True, alpha=0.25, linestyle='--')
    ax_err.spines['top'].set_visible(False)
    ax_err.spines['right'].set_visible(False)

    # -- F1 score --
    ax_f1.plot(T_values, f1_scores, marker='^', color='seagreen', **common)
    for t, f1 in zip(T_values, f1_scores):
        ax_f1.annotate(f'{f1:.3f}', (t, f1), textcoords='offset points',
                       xytext=(0, 9), ha='center', fontsize=8, color='seagreen')
    ax_f1.fill_between(T_values, f1_scores, alpha=0.08, color='seagreen')
    ax_f1.set_xlabel('Boosting Rounds T', fontsize=11)
    ax_f1.set_ylabel('F1-score (+1)',     fontsize=11)
    ax_f1.set_title('AdaBoost + SMOTE:\nF1 Score (+1) vs. T',
                    fontsize=12, fontweight='bold')
    ax_f1.grid(True, alpha=0.25, linestyle='--')
    ax_f1.spines['top'].set_visible(False)
    ax_f1.spines['right'].set_visible(False)

    fig.tight_layout(pad=2.0)
    _save(fig, path)


def plot_adaboost_smote_comparison(no_smote_means, smote_means,
                                   path='output/section4/adaboost_smote_comparison.png'):
    metrics = ['Accuracy', 'Precision (+1)', 'Recall (+1)',
               'F1 (+1)', 'AUC-ROC', 'G-mean']
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'gmean']

    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor='white')
    bars1 = ax.bar(x - w/2, [no_smote_means[k] for k in keys], w,
                   label='AdaBoost (no SMOTE)', color='steelblue', edgecolor='white',
                   linewidth=0.8)
    bars2 = ax.bar(x + w/2, [smote_means[k] for k in keys], w,
                   label='AdaBoost + SMOTE',    color='tomato',    edgecolor='white',
                   linewidth=0.8)

    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.09)
    ax.set_title('AdaBoost vs. AdaBoost + SMOTE  (T=50, mean over 10 runs)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    _save(fig, path)


def plot_final_comparison(results_dict,
                          path='output/section4/final_comparison.png'):
    methods = list(results_dict.keys())
    f1_vals = [results_dict[m]['f1'] for m in methods]
    gm_vals = [results_dict[m]['gmean'] for m in methods]
    auc_vals = [results_dict[m]['auc_roc'] for m in methods]

    x = np.arange(len(methods))
    w = 0.24

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor='white')
    bars1 = ax.bar(x - w, f1_vals,  w, label='F1 (+1)',
                   color='steelblue', edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x,     gm_vals,  w, label='G-mean',
                   color='tomato',    edgecolor='white', linewidth=0.8)
    bars3 = ax.bar(x + w, auc_vals, w, label='AUC-ROC',
                   color='seagreen',  edgecolor='white', linewidth=0.8)

    _annotate_bars(ax, bars1)
    _annotate_bars(ax, bars2)
    _annotate_bars(ax, bars3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.09)
    ax.set_title('Final Method Comparison: Best Configuration per Approach',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.9, loc='upper left')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    _save(fig, path)
