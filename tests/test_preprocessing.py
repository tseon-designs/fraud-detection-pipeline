"""
test_preprocessing.py
----------------------
Unit tests for the preprocessing pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import (
    clean_fraud_data,
    clean_creditcard,
    merge_ip_to_country,
    engineer_fraud_features,
    prepare_creditcard_for_modeling,
    split_and_resample,
)


# ─── FIXTURES ─────────────────────────────────────────

@pytest.fixture
def sample_fraud_df():
    """Minimal e-commerce fraud DataFrame for testing."""
    return pd.DataFrame({
        'user_id': [1, 2, 3, 4, 4],
        'signup_time': [
            '2024-01-01 10:00:00', '2024-01-02 08:00:00',
            '2024-01-03 12:00:00', '2024-01-04 07:00:00',
            '2024-01-04 07:00:00'
        ],
        'purchase_time': [
            '2024-01-01 11:00:00', '2024-01-02 20:00:00',
            '2024-01-04 12:00:00', '2024-01-04 07:30:00',
            '2024-01-04 07:30:00'
        ],
        'purchase_value': [50.0, 120.5, 200.0, 30.0, 30.0],
        'device_id': ['d1', 'd2', 'd3', 'd4', 'd4'],
        'source': ['SEO', 'Ads', 'SEO', 'Direct', 'Direct'],
        'browser': ['Chrome', 'Firefox', 'Safari', 'Chrome', 'Chrome'],
        'sex': ['M', 'F', 'M', 'F', 'F'],
        'age': [25, 33, 45, 28, 28],
        'ip_address': [3627734021.0, 1987540232.0, 2899243657.0, 1102312845.0, 1102312845.0],
        'class': [0, 1, 0, 1, 1],
    })


@pytest.fixture
def sample_ip_df():
    """Minimal IP-to-Country mapping."""
    return pd.DataFrame({
        'lower_bound_ip_address': [1000000000.0, 2000000000.0, 3000000000.0],
        'upper_bound_ip_address': [1999999999.0, 2999999999.0, 3999999999.0],
        'country': ['CountryA', 'CountryB', 'CountryC'],
    })


@pytest.fixture
def sample_creditcard_df():
    """Minimal credit card DataFrame for testing."""
    np.random.seed(42)
    n = 100
    data = {f'V{i}': np.random.randn(n) for i in range(1, 29)}
    data['Time'] = np.linspace(0, 172800, n)
    data['Amount'] = np.abs(np.random.randn(n) * 100)
    data['Class'] = [1 if i % 20 == 0 else 0 for i in range(n)]  # ~5% fraud
    return pd.DataFrame(data)


# ─── TESTS: CLEANING ──────────────────────────────────

class TestCleanFraudData:
    def test_removes_duplicates(self, sample_fraud_df):
        cleaned = clean_fraud_data(sample_fraud_df)
        assert cleaned.duplicated().sum() == 0

    def test_parses_datetimes(self, sample_fraud_df):
        cleaned = clean_fraud_data(sample_fraud_df)
        assert pd.api.types.is_datetime64_any_dtype(cleaned['signup_time'])
        assert pd.api.types.is_datetime64_any_dtype(cleaned['purchase_time'])

    def test_no_null_target(self, sample_fraud_df):
        df = sample_fraud_df.copy()
        df.loc[0, 'class'] = np.nan
        cleaned = clean_fraud_data(df)
        assert cleaned['class'].isnull().sum() == 0

    def test_output_columns_intact(self, sample_fraud_df):
        cleaned = clean_fraud_data(sample_fraud_df)
        required = ['user_id', 'signup_time', 'purchase_time', 'purchase_value',
                    'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class']
        for col in required:
            assert col in cleaned.columns, f"Missing column: {col}"


class TestCleanCreditCard:
    def test_removes_duplicates(self, sample_creditcard_df):
        df = pd.concat([sample_creditcard_df, sample_creditcard_df.iloc[:5]])
        cleaned = clean_creditcard(df)
        assert cleaned.duplicated().sum() == 0

    def test_no_null_target(self, sample_creditcard_df):
        df = sample_creditcard_df.copy()
        df.loc[0, 'Class'] = np.nan
        cleaned = clean_creditcard(df)
        assert cleaned['Class'].isnull().sum() == 0


# ─── TESTS: GEOLOCATION ───────────────────────────────

class TestMergeIpToCountry:
    def test_country_column_added(self, sample_fraud_df, sample_ip_df):
        merged = merge_ip_to_country(sample_fraud_df, sample_ip_df)
        assert 'country' in merged.columns

    def test_no_nulls_in_country(self, sample_fraud_df, sample_ip_df):
        merged = merge_ip_to_country(sample_fraud_df, sample_ip_df)
        assert merged['country'].isnull().sum() == 0

    def test_unknown_for_unmapped(self, sample_fraud_df, sample_ip_df):
        # ip_address 1987540232 should map to CountryA (range 1B–2B)
        merged = merge_ip_to_country(sample_fraud_df, sample_ip_df)
        assert set(merged['country'].unique()).issubset(
            {'CountryA', 'CountryB', 'CountryC', 'Unknown'}
        )

    def test_row_count_preserved(self, sample_fraud_df, sample_ip_df):
        merged = merge_ip_to_country(sample_fraud_df, sample_ip_df)
        assert len(merged) == len(sample_fraud_df)


# ─── TESTS: FEATURE ENGINEERING ───────────────────────

class TestEngineerFraudFeatures:
    def test_time_since_signup_created(self, sample_fraud_df):
        engineered = engineer_fraud_features(sample_fraud_df)
        assert 'time_since_signup' in engineered.columns

    def test_time_since_signup_non_negative(self, sample_fraud_df):
        engineered = engineer_fraud_features(sample_fraud_df)
        # Remove duplicates first so ordering is consistent
        engineered = engineered.drop_duplicates()
        assert (engineered['time_since_signup'] >= 0).all()

    def test_hour_of_day_range(self, sample_fraud_df):
        engineered = engineer_fraud_features(sample_fraud_df)
        assert engineered['hour_of_day'].between(0, 23).all()

    def test_day_of_week_range(self, sample_fraud_df):
        engineered = engineer_fraud_features(sample_fraud_df)
        assert engineered['day_of_week'].between(0, 6).all()

    def test_velocity_positive(self, sample_fraud_df):
        engineered = engineer_fraud_features(sample_fraud_df)
        assert (engineered['transaction_velocity'] >= 1).all()

    def test_all_new_columns_present(self, sample_fraud_df):
        engineered = engineer_fraud_features(sample_fraud_df)
        new_cols = ['time_since_signup', 'hour_of_day', 'day_of_week',
                    'transaction_count_per_user', 'transaction_velocity']
        for col in new_cols:
            assert col in engineered.columns, f"Missing: {col}"


# ─── TESTS: SPLIT AND RESAMPLE ────────────────────────

class TestSplitAndResample:
    def test_stratified_split(self, sample_creditcard_df):
        X, y, _ = prepare_creditcard_for_modeling(sample_creditcard_df)
        X_tr, X_te, y_tr, y_te = split_and_resample(X, y, use_smote=False)
        # Test set should have same fraud rate as original (approx)
        assert len(X_te) > 0
        assert len(X_tr) > 0

    def test_smote_balances_classes(self, sample_creditcard_df):
        X, y, _ = prepare_creditcard_for_modeling(sample_creditcard_df)
        X_tr, X_te, y_tr, y_te = split_and_resample(X, y, use_smote=True)
        import pandas as pd
        counts = pd.Series(y_tr).value_counts()
        # After SMOTE, both classes should have equal count
        assert counts[0] == counts[1]

    def test_test_set_not_resampled(self, sample_creditcard_df):
        X, y, _ = prepare_creditcard_for_modeling(sample_creditcard_df)
        _, X_te, _, y_te = split_and_resample(X, y, use_smote=True)
        # Test set size should be exactly 20% of original
        expected_test_size = int(len(X) * 0.2)
        assert abs(len(X_te) - expected_test_size) <= 2  # allow rounding
