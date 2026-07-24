import numpy as np
import pandas as pd
import os


def load_and_engineer(data_dir, output_dir, max_rows=None):
    print("\n" + "=" * 80)
    print("DATASET 1: AIRBNB OPEN DATA")
    print("=" * 80)

    # ---- Load ----
    csv_path = os.path.join(data_dir, "airbnb_5cities_listings.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(
            data_dir, "Inside Airbnb Dataset", "airbnb_5cities_listings.csv")
    print(f"  Loading from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, nrows=max_rows)
    print(f"  Raw shape: {df.shape}")
    print(f"  Cities: {df['city'].unique().tolist()}")

    raw_df = df.copy()

    # ---- Price Parsing ----
    print("\n  [Step 1] Parsing price column...")
    if df['price'].dtype == object or df['price'].dtype.name == 'str':
        df['price'] = df['price'].astype(str).replace(r'[\$,]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'estimated_revenue_l365d' in df.columns:
        df['estimated_revenue_l365d'] = df['estimated_revenue_l365d'].astype(
            str).replace(r'[\$,]', '', regex=True)
        df['estimated_revenue_l365d'] = pd.to_numeric(
            df['estimated_revenue_l365d'], errors='coerce')
    df = df.dropna(subset=['price'])
    print(
        f"    Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"    Price median: ${df['price'].median():.2f}")

    # ---- Drop Irrelevant Columns ----
    print("\n  [Step 2] Dropping irrelevant columns...")
    columns_to_drop = [
        'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name',
        'description', 'neighborhood_overview', 'picture_url',
        'host_id', 'host_url', 'host_name', 'host_about',
        'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
        'host_verifications', 'neighbourhood', 'calendar_updated',
        'calendar_last_scraped', 'first_review', 'last_review', 'license',
        'bathrooms_text', 'amenities', 'host_location'
    ]
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)
    print(
        f"    Dropped {len(columns_to_drop)} columns. Remaining: {df.shape[1]}")

    # ---- Parse Percentage Columns ----
    print("\n  [Step 3] Parsing percentage columns...")
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(
        f"    host_response_rate missing: {df['host_response_rate'].isna().sum()}")
    print(
        f"    host_acceptance_rate missing: {df['host_acceptance_rate'].isna().sum()}")

    # ---- Superhost Target ----
    print("\n  [Step 4] Encoding superhost target...")
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    df.dropna(subset=['host_is_superhost'], inplace=True)
    print(
        f"    Superhost distribution: {dict(df['host_is_superhost'].value_counts().astype(int))}")

    # ---- Boolean Encoding ----
    print("\n  [Step 5] Encoding boolean columns...")
    boolean_columns = ['host_has_profile_pic',
                       'host_identity_verified', 'instant_bookable', 'has_availability']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0, True: 1, False: 0})
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---- Categorical Encoding ----
    print("\n  [Step 6] Encoding categorical columns...")
    response_time_ordinal_map = {
        'within an hour': 4, 'within a few hours': 3,
        'within a day': 2, 'a few days or more': 1
    }
    if 'host_response_time' in df.columns:
        df['host_response_time'] = df['host_response_time'].map(
            response_time_ordinal_map)

    if 'room_type' in df.columns:
        room_dummies = pd.get_dummies(
            df['room_type'], prefix='room', drop_first=True)
        df = pd.concat([df, room_dummies], axis=1)
        df.drop(columns=['room_type'], inplace=True)

    if 'property_type' in df.columns:
        top_property_types = df['property_type'].value_counts().head(8).index
        df['property_type'] = df['property_type'].apply(
            lambda x: x if x in top_property_types else 'Other')
        prop_dummies = pd.get_dummies(
            df['property_type'], prefix='prop', drop_first=True)
        df = pd.concat([df, prop_dummies], axis=1)
        df.drop(columns=['property_type'], inplace=True)

    if 'city' in df.columns:
        city_dummies = pd.get_dummies(
            df['city'], prefix='city', drop_first=True)
        df = pd.concat([df, city_dummies], axis=1)
        df.drop(columns=['city'], inplace=True)

    for col in ['neighbourhood_group_cleansed', 'neighbourhood_cleansed']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if 'host_since' in df.columns:
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        df['host_years'] = (pd.Timestamp.now() -
                            df['host_since']).dt.days / 365.25
        df.drop(columns=['host_since'], inplace=True)

    # ---- Missing Value Imputation ----
    print("\n  [Step 7] Imputing missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_before = df[numeric_cols].isna().sum()
    columns_with_missing = missing_before[missing_before > 0]
    if len(columns_with_missing) > 0:
        print(f"    Columns with missing values: {len(columns_with_missing)}")
        for col in columns_with_missing.index[:10]:
            pct = columns_with_missing[col] / len(df) * 100
            print(f"      {col}: {columns_with_missing[col]} ({pct:.1f}%)")
            df[col] = df[col].fillna(df[col].median())
        if len(columns_with_missing) > 10:
            for col in columns_with_missing.index[10:]:
                df[col] = df[col].fillna(df[col].median())

    remaining_object_cols = df.select_dtypes(
        include=['object']).columns.tolist()
    if remaining_object_cols:
        print(
            f"    Dropping remaining object columns: {remaining_object_cols}")
        df.drop(columns=remaining_object_cols, inplace=True)

    # ---- Remove Price Outliers ----
    print("\n  [Step 8] Removing price outliers...")
    price_floor = df['price'].quantile(0.01)
    price_ceiling = df['price'].quantile(0.99)
    before = len(df)
    df = df[(df['price'] > 0) & (df['price'] >= price_floor)
            & (df['price'] <= price_ceiling)]
    print(f"    Removed {before - len(df)} outlier rows. Remaining: {len(df)}")
    print(
        f"    Price range after: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

    # ---- Drop Near-Constant Features ----
    print("\n  [Step 9] Removing near-constant features...")
    constant_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 1:
            constant_features.append(col)
    if constant_features:
        df.drop(columns=constant_features, inplace=True)
        print(
            f"    Dropped {len(constant_features)} near-constant features: {constant_features}")

    # ---- Prepare Final Arrays ----
    print("\n  [Step 10] Preparing final feature matrices...")
    clean_df = df.copy()

    # ---- Target leakage: price regression ----
    # estimated_revenue_l365d == price * estimated_occupancy_l365d, verified
    # exactly (18728/18728 rows, max deviation 0.0), so price is recoverable
    # as their ratio. Dropping revenue breaks the identity; occupancy is kept
    # since it is not price-derived and reconstructs nothing alone.
    # Leak is target-specific: neither column derives from superhost status,
    # so both stay in the classification matrix.
    PRICE_DERIVED_FEATURES = ['estimated_revenue_l365d']

    base_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in ['price', 'host_is_superhost']]

    feature_cols_clf = list(base_cols)
    feature_cols_reg = [c for c in base_cols
                        if c not in PRICE_DERIVED_FEATURES]

    dropped = [c for c in base_cols if c in PRICE_DERIVED_FEATURES]
    print(f"    Dropped {len(dropped)} price-derived feature(s) from the "
          f"REGRESSION matrix: {dropped}")
    print(f"    Retained for CLASSIFICATION (not superhost-derived): "
          f"{dropped}")

    # Guard: the reconstruction identity must stay broken.
    if ('estimated_revenue_l365d' in feature_cols_reg
            and 'estimated_occupancy_l365d' in feature_cols_reg):
        raise ValueError(
            "price is reconstructible from estimated_revenue_l365d / "
            "estimated_occupancy_l365d; refusing to build the regression "
            "matrix.")

    X_clf = df[feature_cols_clf].values.astype(np.float64)
    X_reg = df[feature_cols_reg].values.astype(np.float64)
    y_reg = df['price'].values.astype(np.float64)
    y_clf = df['host_is_superhost'].values.astype(np.int32)

    X_clf = np.nan_to_num(X_clf, nan=0.0, posinf=0.0, neginf=0.0)
    X_reg = np.nan_to_num(X_reg, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Regression feature matrix:     {X_reg.shape}")
    print(f"    Classification feature matrix: {X_clf.shape}")
    print(
        f"    Regression target (price): mean={y_reg.mean():.2f}, std={y_reg.std():.2f}")
    print(
        f"    Classification target (superhost): " f"{({int(k): int(v) for k, v in zip(*np.unique(y_clf, return_counts=True))})}")
    print(f"    Feature names ({len(feature_cols_reg)} reg / "
          f"{len(feature_cols_clf)} clf): {feature_cols_reg[:10]}...")
    print(f"    NOTE: superhost status is awarded against thresholds on "
          f"response rate, rating and stay count, all present as features, "
          f"so the classifier partly recovers a rule. Inherent to the task; "
          f"disclose in the report.")

    return {
        'X_reg': X_reg, 'y_reg': y_reg,
        'X_clf': X_clf, 'y_clf': y_clf,
        'feature_names': feature_cols_clf,
        'feature_names_reg': feature_cols_reg,
        'raw_df': raw_df,
        'clean_df': clean_df,
        'dataset_name': 'Airbnb'
    }
