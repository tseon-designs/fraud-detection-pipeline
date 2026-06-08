"""
preprocessing.py
----------------
Data cleaning, geolocation merging, feature engineering,
transformation, and class imbalance handling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# 1. DATA CLEANING
# ─────────────────────────────────────────

def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Fraud_Data.csv e-commerce dataset.

    Steps:
    - Parse datetime columns
    - Remove duplicates
    - Drop rows with null target
    - Report missing values
    """
    df = df.copy()

    # Parse datetimes
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Remove duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"[Clean] Removed {n_before - len(df):,} duplicate rows")

    # Drop rows where target is null
    df = df.dropna(subset=['class'])

    # Impute missing numerical values with median (justified: robust to outliers)
    for col in ['age', 'purchase_value']:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[Clean] Imputed '{col}' nulls with median={median_val:.2f}")

    # Impute missing categoricals with mode
    for col in ['source', 'browser', 'sex']:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[Clean] Imputed '{col}' nulls with mode='{mode_val}'")

    print(f"[Clean] Final shape: {df.shape}")
    return df


def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the creditcard.csv bank dataset.

    Steps:
    - Remove duplicates
    - Drop rows with null target
    - No datetime columns present
    """
    df = df.copy()

    n_before = len(df)
    df = df.drop_duplicates()
    print(f"[Clean] Removed {n_before - len(df):,} duplicate rows")

    df = df.dropna(subset=['Class'])

    # Amount and Time: fill any nulls with median
    for col in ['Amount', 'Time']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"[Clean] Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# 2. GEOLOCATION MERGE
# ─────────────────────────────────────────

def merge_ip_to_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge e-commerce data with IP-to-Country lookup using range-based matching.

    Strategy: pandas.merge_asof on lower_bound then filter upper_bound.
    Reference: https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html

    The ip_address column in Fraud_Data.csv is already a float (numeric IP).
    """
    df = fraud_df.copy()
    ip_map = ip_df.copy().sort_values('lower_bound_ip_address')

    # Ensure ip_address is numeric
    df['ip_int'] = df['ip_address'].astype(float)

    df_sorted = df.sort_values('ip_int')

    # Merge: find the largest lower_bound ≤ ip_int
    merged = pd.merge_asof(
        df_sorted,
        ip_map[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )

    # Validate: ip must be ≤ upper_bound
    merged['country'] = np.where(
        merged['ip_int'] <= merged['upper_bound_ip_address'],
        merged['country'],
        'Unknown'
    )
    merged['country'] = merged['country'].fillna('Unknown')

    # Drop helper columns
    merged = merged.drop(columns=['ip_int', 'lower_bound_ip_address', 'upper_bound_ip_address'],
                         errors='ignore')

    n_mapped = (merged['country'] != 'Unknown').sum()
    print(f"[GeoIP] {n_mapped:,} / {len(merged):,} transactions mapped to a country")
    return merged


# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING (E-COMMERCE)
# ─────────────────────────────────────────

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer behavioral, temporal, and velocity features for e-commerce fraud data.

    New columns:
    - time_since_signup: hours between signup and purchase
    - hour_of_day: hour of purchase (0–23)
    - day_of_week: day of purchase (0=Mon, 6=Sun)
    - transaction_count_per_user: total transactions per user_id
    - transaction_velocity: number of transactions per user within 1-hour rolling window
    """
    df = df.copy()

    # Ensure datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])

    # Time since signup (hours)
    df['time_since_signup'] = (
        (df['purchase_time'] - df['signup_time'])
        .dt.total_seconds() / 3600
    )

    # Temporal features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    # Transaction count per user
    df['transaction_count_per_user'] = df.groupby('user_id')['user_id'].transform('count')

    # Transaction velocity: count of transactions per user within 1-hour rolling window
    df = df.sort_values(['user_id', 'purchase_time'])
    df['purchase_ts'] = df['purchase_time'].astype(np.int64) // 10**9  # convert to seconds

    def rolling_count(group):
        timestamps = group['purchase_ts'].values
        counts = []
        for i, t in enumerate(timestamps):
            window_start = t - 3600  # 1 hour = 3600 seconds
            count = np.sum((timestamps >= window_start) & (timestamps <= t))
            counts.append(count)
        return counts

    velocity_list = []
    for uid, grp in df.groupby('user_id'):
        velocity_list.extend(rolling_count(grp))

    df['transaction_velocity'] = velocity_list
    df = df.drop(columns=['purchase_ts'])

    print(f"[FeatureEng] Added: time_since_signup, hour_of_day, day_of_week, "
          f"transaction_count_per_user, transaction_velocity")
    return df


# ─────────────────────────────────────────
# 4. DATA TRANSFORMATION
# ─────────────────────────────────────────

def prepare_fraud_data_for_modeling(df: pd.DataFrame, scaler_type: str = 'standard'):
    """
    Prepare e-commerce data for modeling:
    - One-hot encode categoricals
    - Drop ID/datetime columns
    - Scale numericals
    - Return X, y

    scaler_type: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
    """
    df = df.copy()

    # Drop columns not useful for modeling
    drop_cols = ['user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot encode categoricals
    cat_cols = ['source', 'browser', 'sex', 'country']
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # Separate target
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    print(f"[PrepFraud] Feature matrix shape: {X.shape}")
    print(f"[PrepFraud] Class distribution:\n{y.value_counts(normalize=True).round(4)}")

    return X, y


def prepare_creditcard_for_modeling(df: pd.DataFrame, scaler_type: str = 'standard'):
    """
    Prepare credit card data for modeling:
    - Scale Amount and Time
    - Separate X and y

    V1-V28 are already PCA-scaled so we only scale Amount and Time.
    """
    df = df.copy()

    # Scale Amount and Time (V1-V28 already PCA-normalized)
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

    y = df['Class'].astype(int)
    X = df.drop(columns=['Class'])

    print(f"[PrepCC] Feature matrix shape: {X.shape}")
    print(f"[PrepCC] Class distribution:\n{y.value_counts(normalize=True).round(4)}")

    return X, y, scaler


def split_and_resample(X, y, test_size: float = 0.2, random_state: int = 42,
                        use_smote: bool = True):
    """
    Stratified train-test split followed by SMOTE on training set only.

    JUSTIFICATION for SMOTE over undersampling:
    - Undersampling discards majority class data, losing potentially valuable patterns.
    - SMOTE generates synthetic minority samples via k-nearest neighbors interpolation,
      preserving all majority class data and enriching the minority class representation.
    - Critical: SMOTE is applied ONLY to training data to prevent data leakage.

    Reference: Chawla et al. (2002) https://arxiv.org/abs/1106.1813
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"\n[Split] Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"[Split] Train class distribution (before resampling):")
    print(y_train.value_counts())

    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"\n[SMOTE] Train class distribution (after SMOTE):")
        print(pd.Series(y_train_res).value_counts())
        print(f"[SMOTE] Training set grew from {len(X_train):,} → {len(X_train_res):,} samples")
    else:
        X_train_res, y_train_res = X_train, y_train

    return X_train_res, X_test, y_train_res, y_test
