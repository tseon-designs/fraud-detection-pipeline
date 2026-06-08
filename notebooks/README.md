# Notebooks — Execution Guide

## Overview

Each notebook is self-contained but must be run **in order** since later notebooks depend on outputs (saved CSV files and models) from earlier ones.

## Execution Order

### Step 1 — EDA: E-Commerce Fraud Data
**File**: `eda-fraud-data.ipynb`

**Purpose**: Explore the Fraud_Data.csv dataset.

**Outputs**:
- `data/processed/fraud_data_cleaned.csv`
- `notebooks/plots/class_dist_e-commerce_fraud.png`
- `notebooks/plots/fraud_by_country.png`
- Various EDA plots in `notebooks/plots/`

**Key findings documented**:
- Class imbalance ratio
- Fraud patterns by country, browser, source
- Time-based patterns

---

### Step 2 — EDA: Credit Card Fraud Data
**File**: `eda-creditcard.ipynb`

**Purpose**: Explore the creditcard.csv dataset.

**Outputs**:
- `data/processed/creditcard_cleaned.csv`
- `notebooks/plots/class_dist_credit_card_fraud.png`
- PCA feature analysis plots

**Key findings documented**:
- Extreme imbalance (<0.2% fraud)
- High-signal PCA features (V4, V11, V12, V14, V17)
- Amount patterns by class

---

### Step 3 — Feature Engineering
**File**: `feature-engineering.ipynb`

**Purpose**: Transform raw data into model-ready feature matrices.

**Inputs**: Raw CSV files in `data/raw/`

**Outputs**:
- `data/processed/fraud_train.csv` (SMOTE-balanced)
- `data/processed/fraud_test.csv` (original distribution)
- `data/processed/creditcard_train.csv` (SMOTE-balanced)
- `data/processed/creditcard_test.csv` (original distribution)
- `models/scaler_fraud.pkl`
- `models/scaler_creditcard.pkl`

**Key operations**:
- Geolocation IP merge
- `time_since_signup`, `hour_of_day`, `day_of_week`, velocity features
- One-hot encoding
- StandardScaler (fit on train only)
- SMOTE (applied to training set ONLY)

---

### Step 4 — Model Building & Training
**File**: `modeling.ipynb`

**Purpose**: Train, tune, and compare models for both datasets.

**Inputs**: Processed train/test CSV files from Step 3

**Outputs**:
- `models/best_model_fraud.pkl` (XGBoost)
- `models/best_model_creditcard.pkl` (XGBoost)
- `models/lr_fraud.pkl`
- `models/lr_creditcard.pkl`
- Evaluation plots in `notebooks/plots/`

**Models**:
- Logistic Regression (baseline, `class_weight='balanced'`)
- XGBoost (ensemble, GridSearchCV hyperparameter tuning)
- 5-Fold Stratified Cross-Validation for both

---

### Step 5 — SHAP Explainability
**File**: `shap-explainability.ipynb`

**Purpose**: Interpret XGBoost models and generate business recommendations.

**Inputs**: Saved models and test CSV files

**Outputs**:
- `notebooks/plots/feature_importance_*.png`
- `notebooks/plots/shap_summary_*.png`
- `notebooks/plots/shap_bar_*.png`
- `notebooks/plots/shap_true_positive_*.png`
- `notebooks/plots/shap_false_positive_*.png`
- `notebooks/plots/shap_false_negative_*.png`

**Analyses**:
- Built-in feature importance (top 10)
- SHAP global summary beeswarm plot
- SHAP waterfall plots for TP, FP, FN
- Top 5 fraud drivers interpretation
- 5 actionable business recommendations
