import numpy as np
import pandas as pd
import os


def analyze_missing_data(df, dataset_name, output_dir):
    print(f"\n  --- Missing Data Analysis: {dataset_name} ---")
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    print(
        f"    Total cells: {total_cells}, Total missing: {total_missing} ({total_missing/total_cells*100:.2f}%)")

    missing_per_column = df.isna().sum()
    missing_per_column = missing_per_column[missing_per_column > 0].sort_values(
        ascending=False)

    report = {'dataset': dataset_name, 'columns': {}}

    for col in missing_per_column.index[:15]:
        n_miss = missing_per_column[col]
        pct = n_miss / len(df) * 100
        missingness_type = 'MCAR'

        miss_indicator = df[col].isna().astype(int)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        max_corr = 0
        corr_col = None
        for other_col in numeric_cols:
            if other_col != col and df[other_col].notna().sum() > 10:
                try:
                    corr = abs(miss_indicator.corr(df[other_col]))
                    if not np.isnan(corr) and corr > max_corr:
                        max_corr = corr
                        corr_col = other_col
                except:
                    pass

        if max_corr > 0.3:
            missingness_type = 'MAR'
        elif pct > 60:
            missingness_type = 'MNAR'

        print(
            f"    {col}: {n_miss} missing ({pct:.1f}%), Type: {missingness_type}")
        if corr_col and max_corr > 0.1:
            print(
                f"      Max correlation with missingness: {corr_col} (r={max_corr:.3f})")

        report['columns'][col] = {
            'n_missing': int(n_miss), 'pct_missing': float(pct),
            'type': missingness_type, 'max_corr': float(max_corr),
            'corr_column': str(corr_col)
        }

    return report


def _build_isolation_tree(X, max_depth, rng, current_depth=0):
    n, d = X.shape

    if n <= 1 or current_depth >= max_depth:
        return {'type': 'leaf', 'size': n}

    feature_index = rng.randint(0, d)
    col = X[:, feature_index]
    col_min, col_max = col.min(), col.max()

    if col_min == col_max:
        return {'type': 'leaf', 'size': n}

    threshold = rng.uniform(col_min, col_max)

    left_mask = col < threshold
    right_mask = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return {'type': 'leaf', 'size': n}

    return {
        'type': 'internal',
        'feature': feature_index,
        'threshold': threshold,
        'left': _build_isolation_tree(X[left_mask], max_depth, rng, current_depth + 1),
        'right': _build_isolation_tree(X[right_mask], max_depth, rng, current_depth + 1),
    }


def _compute_path_length(x, node, current_depth=0):
    if node['type'] == 'leaf':
        n = node['size']
        if n <= 1:
            return current_depth
        return current_depth + _bst_average_path_length(n)

    if x[node['feature']] < node['threshold']:
        return _compute_path_length(x, node['left'], current_depth + 1)
    else:
        return _compute_path_length(x, node['right'], current_depth + 1)


def _bst_average_path_length(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def compute_isolation_forest_scores(X, n_trees=100, max_samples=256, random_state=42):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    max_samples = min(max_samples, n)
    max_depth = int(np.ceil(np.log2(max_samples)))

    trees = []
    for _ in range(n_trees):
        idx = rng.choice(n, size=max_samples, replace=False)
        tree = _build_isolation_tree(X[idx], max_depth, rng)
        trees.append(tree)

    avg_path_lengths = np.zeros(n)
    for i in range(n):
        total = 0.0
        for tree in trees:
            total += _compute_path_length(X[i], tree)
        avg_path_lengths[i] = total / n_trees

    normalization_factor = _bst_average_path_length(max_samples)
    if normalization_factor == 0:
        normalization_factor = 1.0
    scores = 2.0 ** (-avg_path_lengths / normalization_factor)

    labels = scores > 0.5
    return scores, labels


def analyze_outliers(df, dataset_name, output_dir):
    print(f"\n  --- Outlier Analysis: {dataset_name} ---")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:15]
    results = {}

    iso_data = df[numeric_cols].dropna().values
    if len(iso_data) > 5000:
        rng = np.random.RandomState(42)
        iso_idx = rng.choice(len(iso_data), 5000, replace=False)
        iso_sub = iso_data[iso_idx]
    else:
        iso_sub = iso_data

    print(f"    Running Isolation Forest (n_trees=100, max_samples=256)...")
    iso_scores, iso_labels = compute_isolation_forest_scores(
        iso_sub, n_trees=100, max_samples=256)
    iso_total_anomalies = int(iso_labels.sum())
    print(
        f"    Isolation Forest total anomalies (multivariate): {iso_total_anomalies}/{len(iso_sub)}")

    for col in numeric_cols:
        data = df[col].dropna().values
        if len(data) < 10:
            continue

        # ---- Z-Score Method ----
        mean, std = np.mean(data), np.std(data)
        if std > 0:
            z_scores = np.abs((data - mean) / std)
            z_outlier_count = int(np.sum(z_scores > 3))
        else:
            z_outlier_count = 0

        # ---- IQR Method ----
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        iqr_outlier_count = int(
            np.sum((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)))

        # ---- Per-Column Isolation Forest ----
        col_data = data.reshape(-1, 1)
        if len(col_data) > 5000:
            rng_col = np.random.RandomState(42)
            col_sub_idx = rng_col.choice(len(col_data), 5000, replace=False)
            col_sub = col_data[col_sub_idx]
        else:
            col_sub = col_data
        col_scores, _ = compute_isolation_forest_scores(
            col_sub, n_trees=50, max_samples=256)
        threshold = np.percentile(col_scores, 95)
        iso_outlier_count = int(np.sum(col_scores >= threshold))

        if z_outlier_count > 0 or iqr_outlier_count > 0 or iso_outlier_count > 0:
            results[col] = {
                'z_score_outliers': z_outlier_count,
                'iqr_outliers': iqr_outlier_count,
                'isolation_forest_outliers': iso_outlier_count,
                'total_samples': len(data)
            }
            print(f"    {col}: Z-score={z_outlier_count}, IQR={iqr_outlier_count}, "
                  f"IsoForest={iso_outlier_count} (n={len(data)})")

    print(
        f"    Analyzed {len(numeric_cols)} numeric columns, found outliers in {len(results)}")
    return results


def analyze_feature_quality(df, dataset_name, output_dir):
    print(f"\n  --- Feature Quality Analysis: {dataset_name} ---")

    results = {
        'near_constant': [],
        'highly_correlated_pairs': [],
        'potential_leakage': []
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ---- Near-Constant Features ----
    for col in numeric_cols:
        top_freq = df[col].value_counts(
            normalize=True).iloc[0] if df[col].nunique() > 0 else 1.0
        if top_freq > 0.99:
            results['near_constant'].append(col)

    if results['near_constant']:
        print(f"    Near-constant features: {results['near_constant']}")
    else:
        print(f"    No near-constant features found.")

    # ---- Highly Correlated Pairs ----
    if len(numeric_cols) > 1:
        sample_cols = numeric_cols[:30]
        corr = df[sample_cols].corr().abs()
        for i in range(len(sample_cols)):
            for j in range(i + 1, len(sample_cols)):
                if corr.iloc[i, j] > 0.90:
                    results['highly_correlated_pairs'].append(
                        (sample_cols[i], sample_cols[j],
                         float(corr.iloc[i, j]))
                    )

    if results['highly_correlated_pairs']:
        print(
            f"    Highly correlated pairs (r > 0.90): {len(results['highly_correlated_pairs'])}")
        for f1, f2, r in results['highly_correlated_pairs'][:5]:
            print(f"      {f1} <-> {f2}: r={r:.3f}")
    else:
        print(f"    No highly correlated pairs found (r > 0.90).")

    return results


def analyze_label_quality(df, target_col, dataset_name, output_dir):
    print(
        f"\n  --- Label Quality Analysis: {dataset_name} (target: {target_col}) ---")

    results = {}

    if target_col not in df.columns:
        print(f"    Target column '{target_col}' not found.")
        return results

    if df[target_col].dtype in [np.float64, np.int64, float, int]:
        vals = df[target_col].dropna()
        results['distribution'] = {
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'skewness': float(vals.skew()) if hasattr(vals, 'skew') else 0,
            'min': float(vals.min()),
            'max': float(vals.max()),
            'n_unique': int(vals.nunique())
        }
        print(f"    Distribution: mean={results['distribution']['mean']:.2f}, "
              f"std={results['distribution']['std']:.2f}, "
              f"skew={results['distribution']['skewness']:.2f}")
    else:
        vc = df[target_col].value_counts()
        results['class_distribution'] = {str(k): int(v) for k, v in vc.items()}
        max_class = vc.max()
        min_class = vc.min()
        imbalance = max_class / min_class if min_class > 0 else float('inf')
        results['imbalance_ratio'] = float(imbalance)
        print(
            f"    Class distribution: " f"{({str(k): int(v) for k, v in vc.head(10).items()})}")
        print(f"    Imbalance ratio (max/min): {imbalance:.2f}")

    print(f"    Potential bias: temporal/geographic distribution should be examined in report")

    return results


def run_full_investigation(data_dict, output_dir):
    dataset_name = data_dict['dataset_name']
    raw_df = data_dict['raw_df']

    print(f"\n{'=' * 60}")
    print(f"DATA QUALITY INVESTIGATION: {dataset_name}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    results = {
        'missing': analyze_missing_data(raw_df, dataset_name, output_dir),
        'outliers': analyze_outliers(raw_df, dataset_name, output_dir),
        'feature_quality': analyze_feature_quality(raw_df, dataset_name, output_dir),
    }

    target_map = {
        'Airbnb': 'price',
        'NYC_311': 'Problem (formerly Complaint Type)',
        'IBM_HR': 'Attrition',
        'OnlineRetail': 'CustomerID'
    }
    target = target_map.get(dataset_name, None)
    if target and target in raw_df.columns:
        results['label_quality'] = analyze_label_quality(
            raw_df, target, dataset_name, output_dir)

    return results
