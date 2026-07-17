import numpy as np
import pandas as pd
import os


# ---- Temporal Split Configuration ----

FEATURE_WINDOW_FRAC = 0.7

# Customers active in the feature window but absent from the target window
# have future spending of exactly zero (they churned).
#   True  -> keep them with target log1p(0) = 0. More honest: churn is a real
#            outcome and a model that anticipates it is doing useful work.
#            Puts a point mass at zero in the target distribution; R^2 falls.
#   False -> drop them. Cleaner distribution, but silently conditions the whole
#            analysis on "customers who returned", which is a selection bias
#            that must then be disclosed in the report.
INCLUDE_CHURNED_CUSTOMERS = True


def _tercile(series, labels):
    """Rank-based tercile binning. Ranking with method='first' breaks ties so
    that qcut cannot fail with non-unique bin edges on heavily tied columns
    (e.g. recency, where many customers share a last-purchase date)."""
    if len(series) < 3 or series.nunique() < 3:
        return pd.Series(np.full(len(series), int(labels[0])),
                         index=series.index, dtype=int)
    ranked = series.rank(method='first')
    return pd.qcut(ranked, q=3, labels=labels).astype(int)


def assign_segment(score):
    if score >= 8:
        return 0  # High-Value
    elif score >= 6:
        return 1  # Loyal
    elif score >= 4:
        return 2  # At-Risk
    else:
        return 3  # Lost


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
    df = df.dropna(subset=['InvoiceDate'])

    print(f"    Clean transactions: {len(df)}")
    print(f"    Unique customers: {df['CustomerID'].nunique()}")
    print(
        f"    Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    # ---- Temporal Split: Feature Window vs Target Window ----
    print("\n  [Step 2] Splitting transaction log on the time axis...")
    date_min = df['InvoiceDate'].min()
    date_max = df['InvoiceDate'].max()
    span = date_max - date_min
    cutoff = date_min + span * FEATURE_WINDOW_FRAC

    df_feat = df[df['InvoiceDate'] < cutoff].copy()
    df_target = df[df['InvoiceDate'] >= cutoff].copy()

    print(f"    Cutoff date: {cutoff}")
    print(f"    Feature window: {date_min} -> {cutoff} "
          f"({len(df_feat)} transactions, "
          f"{df_feat['CustomerID'].nunique()} customers)")
    print(f"    Target window:  {cutoff} -> {date_max} "
          f"({len(df_target)} transactions, "
          f"{df_target['CustomerID'].nunique()} customers)")

    if len(df_feat) == 0 or len(df_target) == 0:
        raise ValueError(
            "Temporal split produced an empty window. Check FEATURE_WINDOW_FRAC "
            "or the max_rows cap, which truncates the date range.")

    # ---- Aggregate Predictors from the FEATURE Window Only ----
    print("\n  [Step 3] Aggregating customer-level features "
          "(feature window only)...")

    # Recency is measured against the cutoff, NOT against date_max. Using
    # date_max would let the feature see how far the target window extends.
    reference_date = cutoff

    customer_df = df_feat.groupby('CustomerID').agg(
        recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        frequency=('InvoiceNo', 'nunique'),
        monetary=('LineTotal', 'sum'),
        # NOTE ON NAMING: LineTotal is per line item, not per invoice. So
        # 'avg_order_value' is really the average LINE value, and
        # 'n_transactions' counts line items, not orders. 'frequency' is the
        # column that counts unique invoices. The names are kept as-is for
        # continuity with the report, but the algebra only lines up if you
        # read them this way.
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

    country_counts = df_feat.groupby('CustomerID')[
        'Country'].nunique().reset_index()
    country_counts.columns = ['CustomerID', 'country_diversity']
    customer_df = customer_df.merge(
        country_counts, on='CustomerID', how='left')

    df_feat['dow'] = df_feat['InvoiceDate'].dt.dayofweek
    preferred_dow = df_feat.groupby('CustomerID')['dow'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 0).reset_index()
    preferred_dow.columns = ['CustomerID', 'preferred_dow']
    customer_df = customer_df.merge(preferred_dow, on='CustomerID', how='left')

    df_feat['hour'] = df_feat['InvoiceDate'].dt.hour
    preferred_hour = df_feat.groupby('CustomerID')['hour'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 0).reset_index()
    preferred_hour.columns = ['CustomerID', 'preferred_hour']
    customer_df = customer_df.merge(
        preferred_hour, on='CustomerID', how='left')

    print(f"    Customer-level features: {customer_df.shape}")
    print(f"    Features: {customer_df.columns.tolist()}")

    # ---- Aggregate Targets from the TARGET Window Only ----
    print("\n  [Step 4] Aggregating future behaviour (target window only)...")
    future_ref_date = date_max + pd.Timedelta(days=1)

    future_df = df_target.groupby('CustomerID').agg(
        future_recency=('InvoiceDate', lambda x: (
            future_ref_date - x.max()).days),
        future_frequency=('InvoiceNo', 'nunique'),
        future_monetary=('LineTotal', 'sum'),
    ).reset_index()

    # Left join: the customer universe is defined by the FEATURE window.
    # Customers who first appear in the target window have no predictors and
    # are correctly excluded. Customers absent from the target window churned.
    customer_df = customer_df.merge(future_df, on='CustomerID', how='left')

    churned_mask = customer_df['future_monetary'].isna()
    n_churned = int(churned_mask.sum())
    print(f"    Customers active in feature window: {len(customer_df)}")
    print(f"    Returned in target window:   {len(customer_df) - n_churned}")
    print(f"    Churned (zero future spend): {n_churned} "
          f"({n_churned / len(customer_df) * 100:.1f}%)")

    customer_df['future_monetary'] = customer_df['future_monetary'].fillna(0.0)
    customer_df['future_frequency'] = customer_df['future_frequency'].fillna(0)
    customer_df['is_churned'] = churned_mask.astype(int)

    if not INCLUDE_CHURNED_CUSTOMERS:
        before = len(customer_df)
        customer_df = customer_df[~churned_mask].reset_index(drop=True)
        print(f"    INCLUDE_CHURNED_CUSTOMERS=False -> dropped "
              f"{before - len(customer_df)} churned customers")
        print(f"    WARNING: analysis is now conditioned on customers who "
              f"returned. This is a selection bias and must be disclosed.")

    # ---- Classification Target: FUTURE Customer Segment ----
    print("\n  [Step 5] Creating customer segments from the target window...")
    # The segment is now built from future-window RFM, so it is NOT a recoding
    # of the predictors. Previously recency/frequency/monetary from the same
    # window fed both X and the label, and the classifier's only job was to
    # rediscover the binning rule written a few lines above it.
    active = customer_df['is_churned'] == 0
    customer_df['rfm_score'] = 3  # churned customers floor at the lowest score

    if int(active.sum()) >= 3:
        sub = customer_df.loc[active]
        r_t = _tercile(sub['future_recency'], labels=[3, 2, 1])
        f_t = _tercile(sub['future_frequency'], labels=[1, 2, 3])
        m_t = _tercile(sub['future_monetary'], labels=[1, 2, 3])
        customer_df.loc[active, 'rfm_score'] = (r_t + f_t + m_t).values

    customer_df['segment'] = customer_df['rfm_score'].apply(assign_segment)
    segment_names = {0: 'High-Value', 1: 'Loyal', 2: 'At-Risk', 3: 'Lost'}
    print(f"    Segment distribution:")
    for seg_id, seg_name in segment_names.items():
        count = int((customer_df['segment'] == seg_id).sum())
        print(f"      {seg_name}: {count} "
              f"({count / len(customer_df) * 100:.1f}%)")
    print(f"    NOTE: 'Lost' now absorbs all churned customers by "
          f"construction.")

    # ---- Regression Target: Log Future Spending ----
    print("\n  [Step 6] Setting regression target (future spending)...")
    customer_df['log_future_monetary'] = np.log1p(
        customer_df['future_monetary'])
    print(f"    Log future monetary: "
          f"mean={customer_df['log_future_monetary'].mean():.2f}, "
          f"std={customer_df['log_future_monetary'].std():.2f}")
    print(f"    Zero-spend point mass: "
          f"{int((customer_df['future_monetary'] == 0).sum())} customers")

    # ---- Remove Outliers ----
    print("\n  [Step 7] Removing outliers...")
    before = len(customer_df)
    monetary_99th = customer_df['future_monetary'].quantile(0.99)
    customer_df = customer_df[customer_df['future_monetary'] <= monetary_99th]
    print(f"    Removed {before - len(customer_df)} extreme future spenders "
          f"(>{monetary_99th:.2f})")

    # ---- Prepare Feature Arrays ----
    print("\n  [Step 8] Preparing feature arrays...")
    # 'monetary' (feature-window spend) is now a legitimate predictor: it is
    # measured strictly before the target window, so including it leaks
    # nothing. It is also the single strongest honest signal for future spend.
    feature_cols = [
        'recency', 'frequency', 'monetary', 'avg_order_value',
        'total_quantity', 'avg_quantity_per_order', 'unique_products',
        'avg_unit_price', 'max_single_purchase', 'std_order_value',
        'n_transactions', 'country_diversity', 'preferred_dow',
        'preferred_hour'
    ]
    feature_cols = [c for c in feature_cols if c in customer_df.columns]

    # Guard: no target-window column may reach the design matrix.
    banned = ('segment', 'rfm_score', 'log_future_monetary', 'is_churned')
    leaked = [c for c in feature_cols
              if c.startswith('future_') or c in banned]
    if leaked:
        raise ValueError(f"Target-window columns in feature set: {leaked}")

    for col in feature_cols:
        customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce')
    customer_df[feature_cols] = customer_df[feature_cols].fillna(0)

    X = customer_df[feature_cols].values.astype(np.float64)
    y_clf = customer_df['segment'].values.astype(np.int32)
    y_reg = customer_df['log_future_monetary'].values.astype(np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Feature matrix: {X.shape}")
    print(
        f"    Classification target (segments): " f"{({int(k): int(v) for k, v in zip(*np.unique(y_clf, return_counts=True))})}")
    print(
        f"    Regression target (log_future_monetary): mean={y_reg.mean():.2f}")
    print(f"    Features: {feature_cols}")
    print(f"    Leakage check: avg_order_value * n_transactions now "
          f"reconstructs FEATURE-window spend, not the target.")

    clean_df = customer_df.copy()

    return {
        'X_reg': X.copy(), 'y_reg': y_reg,
        'X_clf': X.copy(), 'y_clf': y_clf,
        'feature_names': feature_cols,
        'raw_df': raw_df,
        'clean_df': clean_df,
        'dataset_name': 'OnlineRetail'
    }
