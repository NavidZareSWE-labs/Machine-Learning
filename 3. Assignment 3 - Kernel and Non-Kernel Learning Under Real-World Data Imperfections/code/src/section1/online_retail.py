import numpy as np
import pandas as pd
import os


def load_and_engineer(data_dir, output_dir, max_rows=None):
    print("\n" + "=" * 80)
    print("DATASET 4: ONLINE RETAIL TRANSACTIONS")
    print("=" * 80)

    xlsx_path = os.path.join(data_dir, "Online Retail.xlsx")
    print(f"  Loading from: {xlsx_path}")
    df = pd.read_excel(xlsx_path, nrows=max_rows)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    raw_df = df.copy()

    # ---- Clean Transactions ----
    print("\n  [Step 1] Cleaning transaction data...")
    before = len(df)

    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    print(f"    Removed {before - len(df)} cancelled transactions")

    before = len(df)
    df = df.dropna(subset=['CustomerID'])
    print(f"    Removed {before - len(df)} rows without CustomerID")

    before = len(df)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print(
        f"    Removed {before - len(df)} rows with non-positive quantity/price")

    df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

    print(f"    Clean transactions: {len(df)}")
    print(f"    Unique customers: {df['CustomerID'].nunique()}")
    print(
        f"    Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    # ---- Aggregate to Customer Level (RFM + Behavioral Features) ----
    print("\n  [Step 2] Aggregating to customer-level features...")
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    customer_df = df.groupby('CustomerID').agg(
        recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        frequency=('InvoiceNo', 'nunique'),
        monetary=('LineTotal', 'sum'),
        avg_order_value=('LineTotal', 'mean'),
        total_quantity=('Quantity', 'sum'),
        avg_quantity_per_order=('Quantity', 'mean'),
        unique_products=('StockCode', 'nunique'),
        avg_unit_price=('UnitPrice', 'mean'),
        max_single_purchase=('LineTotal', 'max'),
        std_order_value=('LineTotal', 'std'),
        n_transactions=('InvoiceNo', 'count'),
    ).reset_index()

    customer_df['std_order_value'] = customer_df['std_order_value'].fillna(0)

    country_counts = df.groupby('CustomerID')[
        'Country'].nunique().reset_index()
    country_counts.columns = ['CustomerID', 'country_diversity']
    customer_df = customer_df.merge(
        country_counts, on='CustomerID', how='left')

    df['dow'] = df['InvoiceDate'].dt.dayofweek
    preferred_dow = df.groupby('CustomerID')['dow'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 0).reset_index()
    preferred_dow.columns = ['CustomerID', 'preferred_dow']
    customer_df = customer_df.merge(preferred_dow, on='CustomerID', how='left')

    df['hour'] = df['InvoiceDate'].dt.hour
    preferred_hour = df.groupby('CustomerID')['hour'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 0).reset_index()
    preferred_hour.columns = ['CustomerID', 'preferred_hour']
    customer_df = customer_df.merge(
        preferred_hour, on='CustomerID', how='left')

    print(f"    Customer-level features: {customer_df.shape}")
    print(f"    Features: {customer_df.columns.tolist()}")

    # ---- Customer Segmentation (Classification Target) ----
    print("\n  [Step 3] Creating customer segments...")
    r_quartile = pd.qcut(customer_df['recency'], q=3, labels=[3, 2, 1])
    f_quartile = pd.qcut(customer_df['frequency'].rank(
        method='first'), q=3, labels=[1, 2, 3])
    m_quartile = pd.qcut(customer_df['monetary'].rank(
        method='first'), q=3, labels=[1, 2, 3])

    customer_df['rfm_score'] = r_quartile.astype(
        int) + f_quartile.astype(int) + m_quartile.astype(int)

    def assign_segment(score):
        if score >= 8:
            return 0  # High-Value
        elif score >= 6:
            return 1  # Loyal
        elif score >= 4:
            return 2  # At-Risk
        else:
            return 3  # Lost

    customer_df['segment'] = customer_df['rfm_score'].apply(assign_segment)
    segment_names = {0: 'High-Value', 1: 'Loyal', 2: 'At-Risk', 3: 'Lost'}
    print(f"    Segment distribution:")
    for seg_id, seg_name in segment_names.items():
        count = (customer_df['segment'] == seg_id).sum()
        print(f"      {seg_name}: {count} ({count/len(customer_df)*100:.1f}%)")

    # ---- Regression Target: Log-Monetary ----
    print("\n  [Step 4] Setting regression target (monetary = total spending)...")
    customer_df['log_monetary'] = np.log1p(customer_df['monetary'])
    print(
        f"    Log-monetary: mean={customer_df['log_monetary'].mean():.2f}, std={customer_df['log_monetary'].std():.2f}")

    # ---- Remove Outliers ----
    print("\n  [Step 5] Removing outliers...")
    before = len(customer_df)
    monetary_99th = customer_df['monetary'].quantile(0.99)
    customer_df = customer_df[customer_df['monetary'] <= monetary_99th]
    print(
        f"    Removed {before - len(customer_df)} extreme spenders (>{monetary_99th:.2f})")

    # ---- Prepare Feature Arrays ----
    print("\n  [Step 6] Preparing feature arrays...")
    feature_cols = [
        'recency', 'frequency', 'avg_order_value', 'total_quantity',
        'avg_quantity_per_order', 'unique_products', 'avg_unit_price',
        'max_single_purchase', 'std_order_value', 'n_transactions',
        'country_diversity', 'preferred_dow', 'preferred_hour'
    ]
    feature_cols = [c for c in feature_cols if c in customer_df.columns]

    for col in feature_cols:
        customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce')
    customer_df[feature_cols] = customer_df[feature_cols].fillna(0)

    X = customer_df[feature_cols].values.astype(np.float64)
    y_clf = customer_df['segment'].values.astype(np.int32)
    y_reg = customer_df['log_monetary'].values.astype(np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Feature matrix: {X.shape}")
    print(
        f"    Classification target (segments): " f"{({int(k): int(v) for k, v in zip(*np.unique(y_clf, return_counts=True))})}")
    print(f"    Regression target (log_monetary): mean={y_reg.mean():.2f}")
    print(f"    Features: {feature_cols}")

    clean_df = customer_df.copy()

    return {
        'X_reg': X.copy(), 'y_reg': y_reg,
        'X_clf': X.copy(), 'y_clf': y_clf,
        'feature_names': feature_cols,
        'raw_df': raw_df,
        'clean_df': clean_df,
        'dataset_name': 'OnlineRetail'
    }
