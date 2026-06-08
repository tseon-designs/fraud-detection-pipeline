"""
data_loader.py
--------------
Utility functions for loading raw datasets and performing initial validation.
"""

import pandas as pd
import numpy as np
import os


DATA_RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def load_fraud_data(path: str = None) -> pd.DataFrame:
    """Load the e-commerce fraud dataset."""
    if path is None:
        path = os.path.join(DATA_RAW_PATH, 'Fraud_Data.csv')
    df = pd.read_csv(path)
    print(f"[Fraud_Data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_ip_country(path: str = None) -> pd.DataFrame:
    """Load the IP-to-Country mapping dataset."""
    if path is None:
        path = os.path.join(DATA_RAW_PATH, 'IpAddress_to_Country.csv')
    df = pd.read_csv(path)
    print(f"[IpAddress_to_Country] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_creditcard(path: str = None) -> pd.DataFrame:
    """Load the bank credit card fraud dataset."""
    if path is None:
        path = os.path.join(DATA_RAW_PATH, 'creditcard.csv')
    df = pd.read_csv(path)
    print(f"[creditcard] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def data_quality_report(df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
    """
    Print a comprehensive data quality summary for a DataFrame.

    Returns a DataFrame with per-column stats:
    - dtype, null count, null %, unique count, sample values
    """
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_count': df.nunique(),
        'sample_values': [df[c].dropna().unique()[:3].tolist() for c in df.columns],
    })
    print(f"\n{'='*60}")
    print(f"Data Quality Report — {name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'='*60}")
    print(report.to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")
    print(f"{'='*60}\n")
    return report
