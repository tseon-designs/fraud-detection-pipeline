"""
run_pipeline.py
---------------
End-to-end fraud detection pipeline runner.

Runs both e-commerce and credit card pipelines:
1. Load raw data
2. Clean and preprocess
3. Feature engineer
4. Stratified split + SMOTE
5. Train Logistic Regression + XGBoost
6. Evaluate and compare models
7. Save best models

Usage:
    python scripts/run_pipeline.py

Prerequisites:
    - Place datasets in data/raw/
    - Install: pip install -r requirements.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_fraud_data, load_ip_country, load_creditcard, data_quality_report
from preprocessing import (
    clean_fraud_data, clean_creditcard,
    merge_ip_to_country, engineer_fraud_features,
    prepare_fraud_data_for_modeling, prepare_creditcard_for_modeling,
    split_and_resample,
)
from modeling import (
    train_logistic_regression, train_xgboost,
    evaluate_model, cross_validate_model,
    save_model, compare_models,
)


def run_fraud_pipeline():
    """Complete pipeline for e-commerce fraud data."""
    print("\n" + "="*65)
    print("  PIPELINE 1: E-COMMERCE FRAUD DETECTION")
    print("="*65)

    # Load
    fraud_raw = load_fraud_data()
    ip_country = load_ip_country()

    # Clean
    fraud_clean = clean_fraud_data(fraud_raw)

    # Geolocation
    fraud_geo = merge_ip_to_country(fraud_clean, ip_country)

    # Feature engineering
    fraud_featured = engineer_fraud_features(fraud_geo)

    # Prepare for modeling
    X, y = prepare_fraud_data_for_modeling(fraud_featured)

    # Split + SMOTE
    X_train, X_test, y_train, y_test = split_and_resample(X, y, use_smote=True)

    # Train models
    print("\n[Training] Logistic Regression...")
    lr = train_logistic_regression(X_train, y_train)

    print("\n[Training] XGBoost (no tuning for speed)...")
    xgb = train_xgboost(X_train, y_train, tune=False)

    # Evaluate
    lr_res = evaluate_model(lr, X_test, y_test, model_name="LR E-Commerce")
    xgb_res = evaluate_model(xgb, X_test, y_test, model_name="XGBoost E-Commerce")

    # Compare
    compare_models([lr_res, xgb_res])

    # Save
    save_model(xgb, 'best_model_fraud.pkl')
    save_model(lr, 'lr_fraud.pkl')

    return xgb, X_test, y_test


def run_creditcard_pipeline():
    """Complete pipeline for credit card fraud data."""
    print("\n" + "="*65)
    print("  PIPELINE 2: CREDIT CARD FRAUD DETECTION")
    print("="*65)

    # Load
    cc_raw = load_creditcard()
    cc_clean = clean_creditcard(cc_raw)

    # Prepare
    X, y, scaler = prepare_creditcard_for_modeling(cc_clean)

    # Split + SMOTE
    X_train, X_test, y_train, y_test = split_and_resample(X, y, use_smote=True)

    # Train
    print("\n[Training] Logistic Regression...")
    lr = train_logistic_regression(X_train, y_train)

    print("\n[Training] XGBoost (no tuning for speed)...")
    xgb = train_xgboost(X_train, y_train, tune=False)

    # Evaluate
    lr_res = evaluate_model(lr, X_test, y_test, model_name="LR Credit Card")
    xgb_res = evaluate_model(xgb, X_test, y_test, model_name="XGBoost Credit Card")

    # Compare
    compare_models([lr_res, xgb_res])

    # Save
    save_model(xgb, 'best_model_creditcard.pkl')
    save_model(lr, 'lr_creditcard.pkl')

    return xgb, X_test, y_test


if __name__ == '__main__':
    print("\n" + "="*65)
    print("  ADEY INNOVATIONS — FRAUD DETECTION PIPELINE")
    print("="*65)

    try:
        run_fraud_pipeline()
    except FileNotFoundError as e:
        print(f"\n[Error] Missing data file: {e}")
        print("  → Place Fraud_Data.csv and IpAddress_to_Country.csv in data/raw/")

    try:
        run_creditcard_pipeline()
    except FileNotFoundError as e:
        print(f"\n[Error] Missing data file: {e}")
        print("  → Place creditcard.csv in data/raw/")

    print("\n" + "="*65)
    print("  ✅ PIPELINE COMPLETE — Models saved to models/")
    print("="*65)
