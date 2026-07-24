import numpy as np
import pandas as pd
import os


# ---- Configuration ----

# Fraction of points the Isolation Forest may label anomalous.
ISO_CONTAMINATION = 0.05

# |corr(feature, target)| above which a feature is flagged as a possible leak.
LEAKAGE_CORR_THRESHOLD = 0.95


def analyze_missing_data(df, dataset_name, output_dir):
    print(f"\n  --- Missing Data Analysis: {dataset_name} ---")
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    print(
        f"    Total cells: {total_cells}, Total missing: {total_missing} ({total_missing/total_cells*100:.2f}%)")

    missing_per_column = df.isna().sum()
    missing_per_column = missing_per_column[missing_per_column > 0].sort_values(
        ascending=False)

    # MAR vs MNAR is not identifiable from observed data; labels are heuristic.
    print("    NOTE: classification below is heuristic. MAR vs MNAR is not")
    print("          identifiable from observed data; treat as indicative.")

    report = {'dataset': dataset_name, 'columns': {}, 'heuristic': True}

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
                except Exception:
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
    # c(n) = 2H(n-1) - 2(n-1)/n. Liu, Ting & Zhou (2008), Eq. 1.
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def compute_isolation_forest_scores(X, n_trees=100, max_samples=256,
                                    random_state=42,
                                    contamination=ISO_CONTAMINATION):
    """Isolation Forest anomaly scores s(x) = 2^(-E[h(x)] / c(n)).

    Labels are the top `contamination` fraction by score. A fixed cutoff of
    s > 0.5 is equivalent to E[h] < c(n), i.e. roughly half the data, so it
    ranks rather than isolates. Returns (scores, labels, threshold).
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    max_samples = min(max_samples, n)
    max_depth = int(np.ceil(np.log2(max_samples))) if max_samples > 1 else 1

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

    # A constant column isolates nothing: every point gets an identical score,
    # so `scores >= percentile` would flag 100% of rows (EmployeeCount did
    # exactly that: 1470/1470). No spread means no anomalies.
    if float(scores.max() - scores.min()) < 1e-12:
        return scores, np.zeros(n, dtype=bool), float(scores.max())

    threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))
    labels = scores >= threshold
    return scores, labels, threshold


def analyze_outliers(df, dataset_name, output_dir,
                     contamination=ISO_CONTAMINATION):
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

    print(f"    Running Isolation Forest (n_trees=100, max_samples=256, "
          f"contamination={contamination:.0%})...")
    iso_scores, iso_labels, iso_thresh = compute_isolation_forest_scores(
        iso_sub, n_trees=100, max_samples=256, contamination=contamination)
    iso_total_anomalies = int(iso_labels.sum())
    print(f"    Isolation Forest anomalies (multivariate): "
          f"{iso_total_anomalies}/{len(iso_sub)} "
          f"(score threshold {iso_thresh:.4f})")
    print(f"    Score distribution: min={iso_scores.min():.4f}, "
          f"median={np.median(iso_scores):.4f}, max={iso_scores.max():.4f}")

    results['_multivariate'] = {
        'n_anomalies': iso_total_anomalies,
        'n_samples': int(len(iso_sub)),
        'score_threshold': iso_thresh,
        'contamination': float(contamination),
        'score_min': float(iso_scores.min()),
        'score_median': float(np.median(iso_scores)),
        'score_max': float(iso_scores.max()),
    }

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
        col_scores, col_labels, _ = compute_isolation_forest_scores(
            col_sub, n_trees=50, max_samples=256, contamination=contamination)
        iso_outlier_count = int(col_labels.sum())

        if z_outlier_count > 0 or iqr_outlier_count > 0 or iso_outlier_count > 0:
            results[col] = {
                'z_score_outliers': z_outlier_count,
                'iqr_outliers': iqr_outlier_count,
                'isolation_forest_outliers': iso_outlier_count,
                'total_samples': len(data)
            }
            print(f"    {col}: Z-score={z_outlier_count}, IQR={iqr_outlier_count}, "
                  f"IsoForest={iso_outlier_count} (n={len(data)})")

    n_cols_with_outliers = len([k for k in results if not k.startswith('_')])
    print(
        f"    Analyzed {len(numeric_cols)} numeric columns, found outliers in {n_cols_with_outliers}")
    return results


def analyze_feature_quality(df, dataset_name, output_dir, target_cols=None,
                            feature_cols=None,
                            leakage_corr_threshold=LEAKAGE_CORR_THRESHOLD):
    """Near-constant, redundant and leaking feature detection.

    The leakage test is correlation-based, so it finds near-linear images of
    the target only. Ratio/product reconstructions (e.g. price = revenue /
    occupancy) are invisible here and are caught by guards in the loaders.
    """
    print(f"\n  --- Feature Quality Analysis: {dataset_name} ---")

    results = {
        'near_constant': [],
        'highly_correlated_pairs': [],
        'potential_leakage': []
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [t for t in (target_cols or []) if t in df.columns]

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

    # ---- Potential Target Leakage ----
    if not target_cols:
        print(f"    Leakage scan skipped: no target column supplied.")
        return results

    for target_col in target_cols:
        tgt = df[target_col]
        if pd.api.types.is_numeric_dtype(tgt):
            tgt_num = tgt
        else:
            tgt_num = tgt.astype('category').cat.codes.replace(-1, np.nan)
            if tgt.nunique() > 2:
                # Correlation against category codes is meaningless if unordered.
                print(f"    Leakage scan skipped for '{target_col}': "
                      f"unordered multi-class target ({tgt.nunique()} levels).")
                continue

        for col in numeric_cols:
            if col == target_col:
                continue
            try:
                r = abs(df[col].corr(tgt_num))
            except Exception:
                continue
            if not np.isnan(r) and r > leakage_corr_threshold:
                used = feature_cols is None or col in feature_cols
                results['potential_leakage'].append(
                    (col, target_col, float(r), bool(used)))

    if results['potential_leakage']:
        live = [x for x in results['potential_leakage'] if x[3]]
        print(f"    Columns correlating |r| > {leakage_corr_threshold:.2f} "
              f"with a target:")
        for f, t, r, used in results['potential_leakage']:
            tag = "IN FEATURE MATRIX -- LEAK" if used else "excluded from X (ok)"
            print(f"      {f} vs {t}: r={r:.4f}  [{tag}]")
        if live:
            print(f"    WARNING: {len(live)} leaking column(s) reach the model.")
        else:
            print(f"    None reach the model: all are intermediate columns "
                  f"correctly excluded from the feature matrix.")
    else:
        print(f"    No linear target leakage detected "
              f"(|r| > {leakage_corr_threshold:.2f}).")
    print(f"    (Correlation cannot detect ratio/product reconstructions; "
          f"see loader guards.)")

    return results


def analyze_label_quality(df, target_col, dataset_name, output_dir):
    print(
        f"\n  --- Label Quality Analysis: {dataset_name} (target: {target_col}) ---")

    results = {}

    if target_col not in df.columns:
        print(f"    Target column '{target_col}' not found.")
        return results

    # Use the pandas dtype API: a literal dtype list misroutes nullable types.
    if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20:
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
        print(f"    Range: [{results['distribution']['min']:.2f}, "
              f"{results['distribution']['max']:.2f}], "
              f"n_unique={results['distribution']['n_unique']}")
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
    """Run all four Section 2 analyses.

    target_map names (source_frame, column) per task: some targets exist only
    on the engineered frame, not the raw log.
    """
    dataset_name = data_dict['dataset_name']
    raw_df = data_dict['raw_df']
    clean_df = data_dict.get('clean_df')

    print(f"\n{'=' * 60}")
    print(f"DATA QUALITY INVESTIGATION: {dataset_name}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    frames = {'raw': raw_df, 'clean': clean_df}

    target_map = {
        'Airbnb': [('raw', 'price'), ('raw', 'host_is_superhost')],
        'NYC_311': [('raw', 'Problem (formerly Complaint Type)'),
                    ('raw', 'Complaint Type'),
                    ('clean', 'resolution_hours')],
        'IBM_HR': [('raw', 'Attrition')],
        'OnlineRetail': [('clean', 'segment'),
                         ('clean', 'log_future_monetary')],
    }

    leakage_target_map = {
        'Airbnb': ['price', 'host_is_superhost'],
        'NYC_311': ['resolution_hours'],
        'IBM_HR': ['Attrition_binary'],
        'OnlineRetail': ['log_future_monetary', 'segment'],
    }

    fq_frame = clean_df if clean_df is not None else raw_df

    results = {
        'missing': analyze_missing_data(raw_df, dataset_name, output_dir),
        'outliers': analyze_outliers(raw_df, dataset_name, output_dir),
        'feature_quality': analyze_feature_quality(
            fq_frame, dataset_name, output_dir,
            target_cols=leakage_target_map.get(dataset_name, []),
            feature_cols=data_dict.get('feature_names')),
    }

    results['label_quality'] = {}
    for source, col in target_map.get(dataset_name, []):
        frame = frames.get(source)
        if frame is None or col not in frame.columns:
            continue
        results['label_quality'][col] = analyze_label_quality(
            frame, col, dataset_name, output_dir)

    if not results['label_quality']:
        print(f"\n    WARNING: no label-quality target resolved for "
              f"{dataset_name}.")

    return results
