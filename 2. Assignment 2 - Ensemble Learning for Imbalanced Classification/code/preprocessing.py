import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Covid.csv"


def _identify_column_types(df):
    binary_cols, continuous_cols = [], []
    known_categoricals = ['age', 'Intubation', 'PO2']

    for col in df.columns:
        if col in known_categoricals:
            binary_cols.append(col)
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)

    return binary_cols, continuous_cols


def _impute(X_train_df, X_test_df, binary_cols, continuous_cols, verbose):
    X_train = X_train_df.copy()
    X_test = X_test_df.copy()

    for col in binary_cols:
        mode_series = X_train[col].mode()
        fill = mode_series.iloc[0] if len(mode_series) > 0 else 0
        X_train[col] = X_train[col].fillna(fill)
        X_test[col] = X_test[col].fillna(fill)

    for col in continuous_cols:
        fill = X_train[col].median()
        X_train[col] = X_train[col].fillna(fill)
        X_test[col] = X_test[col].fillna(fill)

    if verbose:
        remaining_NAN = X_train.isnull().sum().sum() + X_test.isnull().sum().sum()
        print(
            f"  [Impute] Binary->mode, Continuous->median. Remaining NaN: {remaining_NAN}")

    return X_train, X_test


def _kendall_feature_selection(X_train_df, y_train, threshold=0.5, verbose=True):
    if verbose:
        print(
            f"\n  [Feature Selection] Kendall correlation on {X_train_df.shape[1]} features ...")

    corr_matrix = X_train_df.corr(method='kendall')

    label_series = pd.Series(y_train, index=X_train_df.index)
    feat_label_corr = {}
    for col in X_train_df.columns:
        tau = X_train_df[col].corr(label_series, method='kendall')
        feat_label_corr[col] = abs(tau) if not np.isnan(tau) else 0.0

    # Greedy removal: iterate all pairs, remove the weaker one
    to_remove = set()
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        if cols[i] in to_remove:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_remove:
                continue
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Keep the feature more correlated with the label
                if feat_label_corr[cols[i]] >= feat_label_corr[cols[j]]:
                    to_remove.add(cols[j])
                else:
                    to_remove.add(cols[i])
                    break

    selected = [c for c in cols if c not in to_remove]
    if verbose:
        print(
            f"  [Feature Selection] Removed {len(to_remove)} features: {sorted(to_remove)}")
        print(f"  [Feature Selection] Remaining: {len(selected)} features")

    return selected, to_remove


def load_and_preprocess(path=CSV_PATH, random_state=42, verbose=True):
    df = pd.read_csv(path)
    X_raw = df.drop(columns=['Label'])
    y = df['Label'].values.astype(int)
    if verbose:
        print("=" * 65)
        print(f"PREPROCESSING  (seed={random_state})")
        print("=" * 65)

        pos = (y == 1).sum()
        neg = (y == -1).sum()
        print(
            f"  Dataset : {df.shape[0]} samples x {df.shape[1]} columns(features)")
        print(f"  Classes : +1(severe) = {pos}\t, -1(non-severe) = {neg}")
        print(f"  Imbalance ratio : {neg/pos:.1f}:1")

    # ========= Missing value rates =========
    missing_rates = X_raw.isnull().mean()
    nonzero = missing_rates[missing_rates > 0].sort_values(ascending=False)
    if verbose:
        print(
            f"\n   Missing value rates ({len(nonzero)} features affected):")
        for col, rate in nonzero.items():
            print(f"    {col:<35s} {rate*100:6.2f}%")

    binary_cols, continuous_cols = _identify_column_types(X_raw)
    if verbose:
        print(
            f"\n  Column types: binary={len(binary_cols)}, continuous={len(continuous_cols)}")

    # =========  Stratified 70/30 split =========
    X_arr = X_raw.values
    X_train_arr, X_test_arr, y_train, y_test = train_test_split(
        X_arr, y, test_size=0.3, stratify=y, random_state=random_state
    )
    X_train_df = pd.DataFrame(X_train_arr, columns=X_raw.columns)
    X_test_df = pd.DataFrame(X_test_arr,  columns=X_raw.columns)
    if verbose:
        print(f"\n  Stratified 70/30 split:")
        print(f"    Train : {X_train_df.shape[0]} samples | "
              f"+1 = {(y_train == 1).sum()}, -1 = {(y_train == -1).sum()}")
        print(f"    Test  : {X_test_df.shape[0]}  samples | "
              f"+1 = {(y_test == 1).sum()}, -1 = {(y_test == -1).sum()}")

    # ========= Impute =========
    X_train_df, X_test_df = _impute(
        X_train_df, X_test_df, binary_cols, continuous_cols, verbose
    )

    # ========= Kendall feature selection =========
    selected, _ = _kendall_feature_selection(
        X_train_df, y_train, threshold=0.5
    )
    X_train_df = X_train_df[selected]
    X_test_df = X_test_df[selected]

    # ========= StandardScaler on continuous features only  =========
    selected_cont = [c for c in continuous_cols if c in selected]
    cont_indices = [selected.index(c) for c in selected_cont]
    if verbose:
        print(f"\n  StandardScaler on {len(cont_indices)} continuous features "
              f"(fit on train only)")

    X_train_np = X_train_df.values.astype(float)
    X_test_np = X_test_df.values.astype(float)

    if cont_indices:
        scaler = StandardScaler()
        X_train_np[:, cont_indices] = scaler.fit_transform(
            X_train_np[:, cont_indices])
        X_test_np[:, cont_indices] = scaler.transform(
            X_test_np[:, cont_indices])

    print("\n  Preprocessing complete.")

    return (X_train_np, X_test_np, y_train, y_test,
            selected, missing_rates, binary_cols, continuous_cols)
