import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
FIGSIZE = (10, 6)
DPI = 150


def plot_missing_data(missing_dict, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    columns = missing_dict.get('columns', {})
    if not columns:
        return

    cols = list(columns.keys())[:15]
    pcts = [columns[c]['pct_missing'] for c in cols]
    types = [columns[c]['type'] for c in cols]
    type_colors = {'MCAR': '#2ecc71', 'MAR': '#f39c12', 'MNAR': '#e74c3c'}
    bar_colors = [type_colors.get(t, '#95a5a6') for t in types]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(cols)), pcts, color=bar_colors)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    ax.set_xlabel('Missing (%)')
    ax.set_title(f'Missing Data Analysis - {dataset_name}')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=type_colors[t], label=t) for t in ['MCAR', 'MAR', 'MNAR']]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_missing_data.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_outlier_comparison(outlier_dict, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not outlier_dict:
        return

    # Keys starting with '_' are summary entries (e.g. '_multivariate'),
    # not per-column outlier counts. Filter them out, and be defensive about
    # any entry that lacks the expected per-column fields.
    cols = [c for c in outlier_dict
            if not str(c).startswith('_')
            and isinstance(outlier_dict[c], dict)
            and 'z_score_outliers' in outlier_dict[c]][:10]
    if not cols:
        return

    z_counts = [outlier_dict[c]['z_score_outliers'] for c in cols]
    iqr_counts = [outlier_dict[c]['iqr_outliers'] for c in cols]
    iso_counts = [outlier_dict[c].get('isolation_forest_outliers', 0) for c in cols]

    x = np.arange(len(cols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, z_counts, width, label='Z-Score', color='#3498db')
    ax.bar(x, iqr_counts, width, label='IQR', color='#e74c3c')
    ax.bar(x + width, iso_counts, width, label='Isolation Forest', color='#2ecc71')

    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Outliers')
    ax.set_title(f'Outlier Detection Comparison - {dataset_name}')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_outlier_comparison.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_feature_distributions(df, feature_cols, dataset_name, output_dir, max_features=12):
    os.makedirs(output_dir, exist_ok=True)

    cols = [c for c in feature_cols if c in df.columns][:max_features]
    if not cols:
        return

    n_cols = min(3, len(cols))
    n_rows = (len(cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        if len(data) > 0:
            axes[i].hist(data, bins=30, edgecolor='white', alpha=0.8)
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)

    for i in range(len(cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Feature Distributions - {dataset_name}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_feature_distributions.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_correlation_matrix(df, feature_cols, dataset_name, output_dir, max_features=20):
    os.makedirs(output_dir, exist_ok=True)

    cols = [c for c in feature_cols if c in df.columns and df[c].dtype in [np.float64, np.int64, float, int]][:max_features]
    if len(cols) < 2:
        return

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5), max(6, len(cols) * 0.4)))
    sns.heatmap(corr, annot=len(cols) <= 12, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5,
                xticklabels=cols, yticklabels=cols)
    ax.set_title(f'Feature Correlation Matrix - {dataset_name}')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_correlation_matrix.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_confusion_matrix(cm, labels, dataset_name, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.7)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels[:10], yticklabels=labels[:10])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix: {model_name} - {dataset_name}')

    plt.tight_layout()
    fname = f'{dataset_name}_{model_name.replace(" ", "_")}_confusion_matrix.png'
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_regression_scatter(y_true, y_pred, dataset_name, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='#3498db')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Regression: {model_name} - {dataset_name}')
    ax.legend()

    plt.tight_layout()
    fname = f'{dataset_name}_{model_name.replace(" ", "_")}_regression_scatter.png'
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_model_comparison(results_dict, metric_name, dataset_name, output_dir, task='classification'):
    os.makedirs(output_dir, exist_ok=True)

    models = list(results_dict.keys())
    values = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.8), 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(range(len(models)), values, color=colors)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison ({task}) - {dataset_name}')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fname = f'{dataset_name}_{metric_name.replace(" ", "_")}_{task}_comparison.png'
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_kernel_comparison(kernel_results, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not kernel_results:
        return

    kernels = list(kernel_results.keys())
    models = list(kernel_results[kernels[0]].keys()) if kernels else []

    if not models:
        return

    x = np.arange(len(models))
    width = 0.8 / len(kernels)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    for i, kernel in enumerate(kernels):
        values = [kernel_results[kernel].get(m, 0) for m in models]
        ax.bar(x + i * width, values, width, label=kernel)

    ax.set_xticks(x + width * (len(kernels) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Performance')
    ax.set_title(f'Kernel Comparison - {dataset_name}')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name}_kernel_comparison.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_computational_analysis(timing_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    datasets = list(timing_dict.keys())
    if not datasets:
        return

    all_models = set()
    for d in datasets:
        all_models.update(timing_dict[d].keys())
    models = sorted(all_models)

    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 2), 6))
    for i, model in enumerate(models):
        values = [timing_dict[d].get(model, 0) for d in datasets]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Computational Analysis: Training Time Across Datasets')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_yscale('log')

    plt.tight_layout()
    path = os.path.join(output_dir, 'computational_analysis.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def plot_worst_predictions(y_true, y_pred, indices, dataset_name, model_name, output_dir, task='classification'):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(indices))

    if task == 'regression':
        errors = np.abs(y_true[indices] - y_pred[indices])
        ax.bar(x, errors, color='#e74c3c', alpha=0.8)
        ax.set_ylabel('Absolute Error')

        for i, idx in enumerate(indices):
            ax.text(i, errors[i] + 0.01 * errors.max(),
                    f'A:{y_true[idx]:.1f}\nP:{y_pred[idx]:.1f}',
                    ha='center', fontsize=7)
    else:
        ax.bar(x - 0.15, y_true[indices], 0.3, label='Actual', color='#3498db', alpha=0.8)
        ax.bar(x + 0.15, y_pred[indices], 0.3, label='Predicted', color='#e74c3c', alpha=0.8)
        ax.set_ylabel('Class Label')
        ax.legend()

    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i}' for i in indices], fontsize=8)
    ax.set_xlabel('Sample Index')
    ax.set_title(f'10 Worst Predictions: {model_name} - {dataset_name}')

    plt.tight_layout()
    fname = f'{dataset_name}_{model_name.replace(" ", "_")}_worst_predictions.png'
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")
