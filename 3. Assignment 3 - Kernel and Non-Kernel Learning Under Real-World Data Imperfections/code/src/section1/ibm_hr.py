import numpy as np
import pandas as pd
import os


def load_and_engineer(data_dir, output_dir, max_rows=None):
    print("\n" + "=" * 80)
    print("DATASET 3: IBM HR EMPLOYEE ATTRITION")
    print("=" * 80)

    csv_path = os.path.join(data_dir, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_dir, "IBM_HR_Analytics_Attrition_Dataset",
                                "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print(f"  Loading from: {csv_path}")
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"  Raw shape: {df.shape}")

    raw_df = df.copy()

    # ---- Target Distribution ----
    print("\n  [Step 1] Target distribution...")
    print(f"    Attrition: {dict(df['Attrition'].value_counts().astype(int))}")
    df['Attrition_binary'] = (df['Attrition'] == 'Yes').astype(int)
    imbalance_ratio = df['Attrition_binary'].value_counts(
    )[0] / df['Attrition_binary'].value_counts()[1]
    print(f"    Imbalance ratio (No/Yes): {imbalance_ratio:.2f}")

    # ---- Drop Near-Constant / Leakage Features ----
    print("\n  [Step 2] Dropping near-constant and leakage features...")
    columns_to_drop = ['EmployeeCount', 'StandardHours',
                       'Over18', 'EmployeeNumber', 'Attrition']
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"    Dropped: {columns_to_drop}")

    # ---- Root Driver Investigation ----
    print("\n  [Step 3] Investigating root drivers of attrition...")
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corrwith(
        df['Attrition_binary']).abs().sort_values(ascending=False)
    print(f"    Top correlates with attrition:")
    for feat, corr in correlations.head(10).items():
        if feat != 'Attrition_binary':
            print(f"      {feat}: {corr:.4f}")

    # ---- Encode Categorical Variables ----
    print("\n  [Step 4] Encoding categorical variables...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"    Categorical columns: {categorical_cols}")

    ordinal_encoding_maps = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
    }
    for col, mapping in ordinal_encoding_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    nominal_cols = [
        c for c in categorical_cols if c not in ordinal_encoding_maps and c in df.columns]
    for col in nominal_cols:
        dummies = pd.get_dummies(df[col], prefix=col[:5], drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)

    print(f"    Shape after encoding: {df.shape}")

    # ---- Remove Highly Correlated Redundant Features ----
    print("\n  [Step 5] Removing highly correlated redundant features...")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != 'Attrition_binary']
    corr_matrix = df[numeric_features].corr().abs()

    high_correlation_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.85:
                high_correlation_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j],
                                               corr_matrix.iloc[i, j]))

    redundant_features = set()
    for f1, f2, corr_val in high_correlation_pairs:
        target_corr_f1 = abs(df[f1].corr(df['Attrition_binary']))
        target_corr_f2 = abs(df[f2].corr(df['Attrition_binary']))
        feature_to_drop = f2 if target_corr_f1 >= target_corr_f2 else f1
        redundant_features.add(feature_to_drop)
        print(
            f"      Corr({f1}, {f2})={corr_val:.3f}, dropping {feature_to_drop}")

    if redundant_features:
        df.drop(columns=list(redundant_features), inplace=True)
    print(
        f"    Dropped {len(redundant_features)} redundant features. Shape: {df.shape}")

    # ---- Handle Missing Values ----
    print("\n  [Step 6] Handling missing values...")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"    Missing values found:")
        for col, cnt in missing.items():
            print(f"      {col}: {cnt}")
            if df[col].dtype != object:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print(f"    No missing values found.")

    # ---- Prepare Arrays ----
    print("\n  [Step 7] Preparing final arrays...")
    feature_cols = [c for c in df.select_dtypes(
        include=[np.number]).columns if c != 'Attrition_binary']

    X = df[feature_cols].values.astype(np.float64)
    y = df['Attrition_binary'].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Feature matrix: {X.shape}")
    print(
        f"    Target distribution: " f"{({int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})}")
    print(f"    Features ({len(feature_cols)}): {feature_cols[:10]}...")

    clean_df = df.copy()

    return {
        'X_clf': X, 'y_clf': y,
        'X_reg': None, 'y_reg': None,
        'feature_names': feature_cols,
        'raw_df': raw_df,
        'clean_df': clean_df,
        'dataset_name': 'IBM_HR'
    }
