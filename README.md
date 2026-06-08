# Fraud Detection for E-Commerce and Bank Transactions

> **Adey Innovations Inc.** — FinTech Unified Fraud Detection System  
> Built for the 10 Academy Week 5&6 Challenge (Jun 4–16, 2026)

---

## 🎯 Project Overview

A production-quality, end-to-end machine learning system for detecting fraudulent transactions across two distinct data streams:

1. **E-Commerce Transactions** (`Fraud_Data.csv`) — Rich behavioral data including IP addresses, device IDs, browsing context, and signup metadata
2. **Bank Credit Card Transactions** (`creditcard.csv`) — Anonymized PCA-transformed features from a real-world fraud dataset

**Business Goal**: Balance false positives (legitimate transactions incorrectly flagged, damaging customer trust) against false negatives (missed fraud, causing direct financial loss). Standard accuracy is **not used** — we rely on **AUC-PR** and **F1-Score** due to severe class imbalance.

---

## 📁 Repository Structure

```
fraud-detection/
├── .github/workflows/unittests.yml   # CI/CD: runs tests on every push
├── data/
│   ├── raw/                          # Original datasets (gitignored)
│   └── processed/                    # Cleaned & engineered datasets (gitignored)
├── notebooks/
│   ├── eda-fraud-data.ipynb          # Task 1: EDA for e-commerce data
│   ├── eda-creditcard.ipynb          # Task 1: EDA for credit card data
│   ├── feature-engineering.ipynb     # Task 1: Full preprocessing pipelines
│   ├── modeling.ipynb                # Task 2: Model training & evaluation
│   └── shap-explainability.ipynb     # Task 3: SHAP analysis & recommendations
├── src/
│   ├── data_loader.py                # Dataset loading & quality reporting
│   ├── preprocessing.py              # Cleaning, feature engineering, SMOTE
│   ├── modeling.py                   # Model training, evaluation, CV
│   ├── explainability.py             # SHAP analysis utilities
│   └── visualization.py             # EDA plotting utilities
├── tests/
│   └── test_preprocessing.py         # Unit tests for preprocessing pipeline
├── models/                           # Saved model artifacts (gitignored)
├── scripts/
│   └── run_pipeline.py               # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10 or 3.11
- `pip`

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd fraud-detection

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place the raw datasets in `data/raw/`:
```
data/raw/Fraud_Data.csv
data/raw/IpAddress_to_Country.csv
data/raw/creditcard.csv
```

---

## 🚀 Running the Project

### Option 1: Notebooks (Recommended — in order)

```bash
jupyter notebook
```

Run notebooks in this order:
1. `notebooks/eda-fraud-data.ipynb` — E-commerce EDA
2. `notebooks/eda-creditcard.ipynb` — Credit card EDA
3. `notebooks/feature-engineering.ipynb` — Preprocessing + SMOTE
4. `notebooks/modeling.ipynb` — Train and compare models
5. `notebooks/shap-explainability.ipynb` — Interpret and recommend

### Option 2: End-to-End Script

```bash
python scripts/run_pipeline.py
```

### Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=src
```

---

## 📊 Model Performance Summary

*(Fill in after running `modeling.ipynb`)*

### E-Commerce Fraud Dataset

| Model | AUC-PR | F1-Score | ROC-AUC |
|-------|--------|----------|---------|
| Logistic Regression | — | — | — |
| **XGBoost** ✅ (Best) | — | — | — |

### Credit Card Fraud Dataset

| Model | AUC-PR | F1-Score | ROC-AUC |
|-------|--------|----------|---------|
| Logistic Regression | — | — | — |
| **XGBoost** ✅ (Best) | — | — | — |

---

## 🔑 Key Business Recommendations

Based on SHAP analysis:

1. **Flag early-signup transactions** — Transactions within 1 hour of account signup require 2FA
2. **Velocity-based rate limiting** — Block accounts with >3 transactions/hour automatically
3. **Geolocation risk scoring** — Apply risk multipliers for high-fraud-rate countries
4. **High-value transaction review** — Flag purchases >2σ above user's historical average
5. **SHAP for dispute resolution** — Use force plots to explain individual fraud decisions to agents

---

## 📐 Design Principles

- **No accuracy metric** — AUC-PR and F1 only on imbalanced data
- **SMOTE on training set only** — Never applied to test/validation to prevent data leakage
- **Scaler fit on training data only** — Transform test set with training-fitted scaler
- **Reproducibility** — All random seeds fixed at `random_state=42`
- **Two independent pipelines** — E-commerce and credit card are treated as separate problems

---

## 📚 References

- [SMOTE Paper (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [scikit-learn Precision-Recall Curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [pandas.merge_asof](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)

---

## 👤 Author

**Tsion Fikru**  
Data Science Challenge — 10 Academy Week 5&6  
Adey Innovations Inc. Fraud Detection Project
