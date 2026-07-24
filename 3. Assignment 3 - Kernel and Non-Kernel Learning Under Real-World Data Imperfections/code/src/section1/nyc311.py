import numpy as np
import pandas as pd
import os


def _dedup(seq):
    """Order-preserving de-duplication."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_and_engineer(data_dir, output_dir, max_rows=100000):
    print("\n" + "=" * 80)
    print("DATASET 2: NYC 311 SERVICE REQUESTS")
    print("=" * 80)

    csv_path = os.path.join(data_dir, "311_Service_Requests_from_2020_to_Present_20260618.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_dir, "NYC 311 Service Requests",
                                "311_Service_Requests_from_2020_to_Present_20260618.csv")
    print(f"  Loading from: {csv_path}")

    # ---- Intelligent Sampling ----
    print(f"  [Step 1] Loading with stratified sampling (max {max_rows} rows)...")
    df_full = pd.read_csv(csv_path, low_memory=False, nrows=max_rows)
    print(f"    Loaded shape: {df_full.shape}")

    raw_df = df_full.copy()

    column_rename_map = {
        'Problem (formerly Complaint Type)': 'ComplaintType',
        'Problem Detail (formerly Descriptor)': 'Descriptor',
        'Additional Details': 'AdditionalDetails',
        'Complaint Type': 'ComplaintType',
        'complaint_type': 'ComplaintType',
        'Descriptor': 'Descriptor',
        'descriptor': 'Descriptor',
        'descriptor_2': 'AdditionalDetails',
        'Created Date': 'CreatedDate',
        'created_date': 'CreatedDate',
        'Closed Date': 'ClosedDate',
        'closed_date': 'ClosedDate',
        'Resolution Action Updated Date': 'ResolutionDate',
        'resolution_action_updated_date': 'ResolutionDate',
        'X Coordinate (State Plane)': 'X_Coord',
        'x_coordinate_state_plane': 'X_Coord',
        'Y Coordinate (State Plane)': 'Y_Coord',
        'y_coordinate_state_plane': 'Y_Coord',
        'Open Data Channel Type': 'ChannelType',
        'open_data_channel_type': 'ChannelType',
        'Location Type': 'LocationType',
        'location_type': 'LocationType',
        'Incident Zip': 'Zip',
        'incident_zip': 'Zip',
        'Community Board': 'CommunityBoard',
        'community_board': 'CommunityBoard',
        'Council District': 'CouncilDistrict',
        'council_district': 'CouncilDistrict',
        'Police Precinct': 'PolicePrecinct',
        'police_precinct': 'PolicePrecinct',
        'Borough': 'Borough',
        'borough': 'Borough',
        'Unique Key': 'Unique Key',
        'unique_key': 'Unique Key',
    }
    df = df_full.rename(columns={k: v for k, v in column_rename_map.items() if k in df_full.columns})

    # ---- Remove Duplicates ----
    print("\n  [Step 2] Removing duplicate reports...")
    before = len(df)
    df.drop_duplicates(subset=['Unique Key'], inplace=True)
    print(f"    Removed {before - len(df)} duplicates. Remaining: {len(df)}")

    # ---- Temporal Feature Engineering ----
    print("\n  [Step 3] Engineering temporal features...")
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], errors='coerce')
    df['ClosedDate'] = pd.to_datetime(df['ClosedDate'], errors='coerce')

    df['resolution_hours'] = (df['ClosedDate'] - df['CreatedDate']).dt.total_seconds() / 3600.0

    df['created_hour'] = df['CreatedDate'].dt.hour
    df['created_dayofweek'] = df['CreatedDate'].dt.dayofweek
    df['created_month'] = df['CreatedDate'].dt.month
    df['created_year'] = df['CreatedDate'].dt.year
    df['is_weekend'] = (df['created_dayofweek'] >= 5).astype(int)
    df['is_business_hours'] = ((df['created_hour'] >= 9) & (df['created_hour'] <= 17)).astype(int)

    print(f"    Temporal features created: hour, dayofweek, month, year, is_weekend, is_business_hours")

    # ---- Classification Target: Top Complaint Categories ----
    print("\n  [Step 4] Preparing classification target (Complaint Category)...")
    if 'ComplaintType' in df.columns:
        complaint_col = 'ComplaintType'
    else:
        complaint_col = [c for c in df.columns if 'complaint' in c.lower() or 'problem' in c.lower()]
        complaint_col = complaint_col[0] if complaint_col else None

    if complaint_col:
        top_complaints = df[complaint_col].value_counts().head(10).index.tolist()
        print(f"    Top 10 complaint types: {top_complaints}")
        df_clf = df[df[complaint_col].isin(top_complaints)].copy()
        complaint_label_map = {c: i for i, c in enumerate(top_complaints)}
        df_clf['complaint_label'] = df_clf[complaint_col].map(complaint_label_map)
    else:
        print("    WARNING: No complaint type column found!")
        df_clf = df.copy()
        df_clf['complaint_label'] = 0

    # ---- Regression Target: Resolution Time ----
    print("\n  [Step 5] Preparing regression target (Resolution Time)...")
    df_reg = df.dropna(subset=['resolution_hours']).copy()
    df_reg = df_reg[(df_reg['resolution_hours'] > 0) & (df_reg['resolution_hours'] < 720)]
    print(f"    Valid resolution time samples: {len(df_reg)}")
    print(f"    Resolution time: mean={df_reg['resolution_hours'].mean():.2f}h, median={df_reg['resolution_hours'].median():.2f}h")

    # ---- Feature Engineering ----
    print("\n  [Step 6] Engineering features...")

    categorical_features = ['Borough', 'ChannelType', 'LocationType', 'Status']
    for col in categorical_features:
        if col in df.columns:
            top_vals = df[col].value_counts().head(6).index
            df[col + '_enc'] = df[col].apply(lambda x: x if x in top_vals else 'Other')
            dummies = pd.get_dummies(df[col + '_enc'], prefix=col[:4], drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df_clf = df_clf.reindex(df.index).dropna(subset=['complaint_label']) if complaint_col else df_clf
            if col in df_reg.columns:
                df_reg[col + '_enc'] = df_reg[col].apply(lambda x: x if x in top_vals else 'Other')
                dum = pd.get_dummies(df_reg[col + '_enc'], prefix=col[:4], drop_first=True)
                df_reg = pd.concat([df_reg, dum], axis=1)

    numeric_feature_names = [
        'created_hour', 'created_dayofweek', 'created_month', 'created_year',
        'is_weekend', 'is_business_hours',
        'Latitude', 'Longitude', 'X_Coord', 'Y_Coord',
        'CouncilDistrict', 'PolicePrecinct'
    ]
    numeric_feature_names = [f for f in numeric_feature_names if f in df.columns]

    dummy_col_prefixes = ['Boro', 'Chan', 'Loca', 'Stat']
    dummy_cols = [c for c in df.columns if any(c.startswith(p) for p in dummy_col_prefixes)]

    feature_cols = numeric_feature_names + [c for c in dummy_cols if c in df.columns]
    print(f"    Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

    # ---- Build Classification Arrays ----
    print("\n  [Step 7] Building classification arrays...")
    df_clf_final = df.loc[df.index.isin(df_clf.index) if hasattr(df_clf, 'index') else df.index].copy()
    if complaint_col and complaint_col in df_clf_final.columns:
        df_clf_final = df_clf_final[df_clf_final[complaint_col].isin(top_complaints)]
        df_clf_final['complaint_label'] = df_clf_final[complaint_col].map(complaint_label_map)
    else:
        df_clf_final['complaint_label'] = 0

    valid_clf_features = [c for c in feature_cols if c in df_clf_final.columns]
    df_clf_final = df_clf_final.dropna(subset=['complaint_label'])
    for col in valid_clf_features:
        df_clf_final[col] = pd.to_numeric(df_clf_final[col], errors='coerce')
    df_clf_final[valid_clf_features] = df_clf_final[valid_clf_features].fillna(df_clf_final[valid_clf_features].median())

    X_clf = df_clf_final[valid_clf_features].values.astype(np.float64)
    y_clf = df_clf_final['complaint_label'].values.astype(np.int32)
    X_clf = np.nan_to_num(X_clf, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    Classification: X={X_clf.shape}, classes={len(np.unique(y_clf))}")

    # ---- Build Regression Arrays ----
    print("\n  [Step 8] Building regression arrays...")
    valid_reg_features = [c for c in feature_cols if c in df_reg.columns]
    reg_dummy_cols = [c for c in df_reg.columns if any(c.startswith(p) for p in dummy_col_prefixes)]
    # Order-preserving dedup: list(set(...)) varies per process
    # (PYTHONHASHSEED), so the reported feature list was not reproducible.
    valid_reg_features = _dedup(
        valid_reg_features + [c for c in reg_dummy_cols
                              if c in df_reg.columns])
    for col in valid_reg_features:
        df_reg[col] = pd.to_numeric(df_reg[col], errors='coerce')
    df_reg[valid_reg_features] = df_reg[valid_reg_features].fillna(df_reg[valid_reg_features].median())

    X_reg = df_reg[valid_reg_features].values.astype(np.float64)
    y_reg = df_reg['resolution_hours'].values.astype(np.float64)
    X_reg = np.nan_to_num(X_reg, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    Regression: X={X_reg.shape}, target mean={y_reg.mean():.2f}h")

    print(f"\n  [Summary] NYC 311 Engineering Complete")
    print(f"    Classification samples: {len(y_clf)}, features: {X_clf.shape[1]}")
    print(f"    Regression samples: {len(y_reg)}, features: {X_reg.shape[1]}")

    return {
        'X_reg': X_reg, 'y_reg': y_reg,
        'X_clf': X_clf, 'y_clf': y_clf,
        'feature_names': valid_clf_features,
        'feature_names_reg': valid_reg_features,
        'raw_df': raw_df,
        'clean_df': df,
        'dataset_name': 'NYC_311'
    }
