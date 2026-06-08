"""
modeling.py
-----------
Model training, evaluation, cross-validation, and persistence utilities.

Primary metrics: AUC-PR and F1-Score (NOT accuracy — misleading on imbalanced data).
"""

import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'plots')


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Model",
                   save_plots: bool = True) -> dict:
    """
    Evaluate a trained model with AUC-PR, F1-Score, and Confusion Matrix.

    Returns a dict of metrics.
    """
    ensure_dirs()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*55}")
    print(f"  AUC-PR (Primary):   {auc_pr:.4f}")
    print(f"  F1-Score (weighted):{f1:.4f}")
    print(f"  ROC-AUC:            {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    if save_plots:
        # Confusion Matrix heatmap
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'])
        axes[0].set_title(f'{model_name} — Confusion Matrix', fontweight='bold')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        axes[1].plot(recall, precision, lw=2, color='steelblue',
                     label=f'AUC-PR = {auc_pr:.4f}')
        axes[1].axhline(y=y_test.mean(), color='red', linestyle='--',
                        label=f'Baseline (fraud rate={y_test.mean():.4f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'{model_name} — Precision-Recall Curve', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = model_name.replace(' ', '_').lower()
        plot_path = os.path.join(PLOTS_DIR, f'{safe_name}_evaluation.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[Plot] Saved → {plot_path}")

    return {
        'model_name': model_name,
        'auc_pr': round(auc_pr, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'confusion_matrix': cm,
    }


# ─────────────────────────────────────────
# BASELINE: LOGISTIC REGRESSION
# ─────────────────────────────────────────

def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression baseline.

    class_weight='balanced' compensates for remaining imbalance post-SMOTE.
    """
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[LR] Logistic Regression trained.")
    return model


# ─────────────────────────────────────────
# ENSEMBLE: XGBOOST
# ─────────────────────────────────────────

def train_xgboost(X_train, y_train, tune: bool = True,
                  random_state: int = 42) -> XGBClassifier:
    """
    Train an XGBoost classifier with optional hyperparameter tuning.

    Tuning uses GridSearchCV with StratifiedKFold(3) over a focused grid.
    Primary scoring: average_precision (AUC-PR).
    """
    scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())

    if tune:
        print("[XGB] Running hyperparameter search...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
        }
        base = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            base, param_grid, scoring='average_precision',
            cv=cv, n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        print(f"[XGB] Best params: {search.best_params_}")
        print(f"[XGB] Best CV AUC-PR: {search.best_score_:.4f}")
        model = search.best_estimator_
    else:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)
        print("[XGB] XGBoost trained (no tuning).")

    return model


# ─────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────

def cross_validate_model(model, X, y, n_splits: int = 5,
                          random_state: int = 42) -> dict:
    """
    Stratified K-Fold cross-validation reporting AUC-PR and F1 mean ± std.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        'auc_pr': 'average_precision',
        'f1_weighted': 'f1_weighted',
    }

    results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    summary = {
        'AUC-PR mean': round(results['test_auc_pr'].mean(), 4),
        'AUC-PR std': round(results['test_auc_pr'].std(), 4),
        'F1 mean': round(results['test_f1_weighted'].mean(), 4),
        'F1 std': round(results['test_f1_weighted'].std(), 4),
    }

    print(f"\n[CV] Stratified {n_splits}-Fold Results:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ─────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────

def save_model(model, filename: str):
    """Save a trained model to the models/ directory."""
    ensure_dirs()
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"[Save] Model saved → {path}")


def load_model(filename: str):
    """Load a saved model from the models/ directory."""
    path = os.path.join(MODELS_DIR, filename)
    model = joblib.load(path)
    print(f"[Load] Model loaded ← {path}")
    return model


# ─────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────

def compare_models(results_list: list) -> pd.DataFrame:
    """
    Build a side-by-side comparison table from a list of evaluate_model() dicts.
    """
    rows = []
    for r in results_list:
        rows.append({
            'Model': r['model_name'],
            'AUC-PR': r['auc_pr'],
            'F1-Score': r['f1_score'],
            'ROC-AUC': r['roc_auc'],
        })
    df = pd.DataFrame(rows).set_index('Model')
    print("\n[Comparison] Model Leaderboard:")
    print(df.to_string())
    return df
