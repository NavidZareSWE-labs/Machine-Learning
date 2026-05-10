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
