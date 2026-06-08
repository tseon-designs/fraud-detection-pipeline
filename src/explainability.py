"""
explainability.py
-----------------
SHAP-based model interpretation: global summary plots, force plots,
and business recommendation generation.

Reference: https://shap.readthedocs.io/en/latest/
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings
warnings.filterwarnings('ignore')

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'plots')


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# SHAP EXPLAINER SETUP
# ─────────────────────────────────────────

def get_shap_explainer(model, X_train_sample=None):
    """
    Create the appropriate SHAP explainer for the given model type.

    - Tree-based models (XGBoost, LightGBM, RandomForest): TreeExplainer (fast, exact)
    - Linear models: LinearExplainer
    - Others: KernelExplainer (slow, model-agnostic)
    """
    model_type = type(model).__name__

    if model_type in ('XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier',
                      'GradientBoostingClassifier'):
        explainer = shap.TreeExplainer(model)
        print(f"[SHAP] Using TreeExplainer for {model_type}")
    elif model_type == 'LogisticRegression':
        explainer = shap.LinearExplainer(model, X_train_sample)
        print(f"[SHAP] Using LinearExplainer for {model_type}")
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_train_sample[:100])
        print(f"[SHAP] Using KernelExplainer for {model_type}")

    return explainer


def compute_shap_values(explainer, X, check_additivity: bool = False):
    """
    Compute SHAP values. Returns raw shap_values array.
    For binary classifiers, returns values for the positive (fraud) class.
    """
    shap_values = explainer.shap_values(X, check_additivity=check_additivity)

    # Tree models return list [class0, class1] — take fraud class
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    print(f"[SHAP] Computed SHAP values: shape={np.array(shap_values).shape}")
    return shap_values


# ─────────────────────────────────────────
# GLOBAL SUMMARY PLOT
# ─────────────────────────────────────────

def plot_shap_summary(shap_values, X, model_name: str = "Model",
                      max_display: int = 20):
    """
    SHAP Summary Plot — shows global feature impact (beeswarm plot).
    Each dot is a sample; color = feature value; x-axis = SHAP impact.
    """
    ensure_dirs()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
        plot_type='dot'
    )
    plt.title(f'SHAP Summary — {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'shap_summary_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[SHAP] Summary plot saved → {path}")


def plot_shap_bar(shap_values, X, model_name: str = "Model", top_n: int = 10):
    """
    SHAP Bar Plot — mean absolute SHAP values (global feature importance).
    """
    ensure_dirs()
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X,
        max_display=top_n,
        plot_type='bar',
        show=False
    )
    plt.title(f'SHAP Global Feature Importance — {model_name}',
              fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'shap_bar_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[SHAP] Bar plot saved → {path}")


# ─────────────────────────────────────────
# INDIVIDUAL FORCE PLOTS
# ─────────────────────────────────────────

def find_sample_indices(y_test, y_pred, y_proba=None):
    """
    Find indices for:
    - True Positive (correctly predicted fraud)
    - False Positive (legitimate flagged as fraud)
    - False Negative (fraud missed by model)
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    # Pick highest-confidence examples if proba available
    if y_proba is not None:
        tp = tp_idx[np.argmax(y_proba[tp_idx])] if len(tp_idx) > 0 else None
        fp = fp_idx[np.argmax(y_proba[fp_idx])] if len(fp_idx) > 0 else None
        fn = fn_idx[np.argmin(y_proba[fn_idx])] if len(fn_idx) > 0 else None
    else:
        tp = tp_idx[0] if len(tp_idx) > 0 else None
        fp = fp_idx[0] if len(fp_idx) > 0 else None
        fn = fn_idx[0] if len(fn_idx) > 0 else None

    print(f"[SHAP] Sample indices — TP: {tp}, FP: {fp}, FN: {fn}")
    return tp, fp, fn


def plot_force_plot(explainer, shap_values, X, idx: int,
                    label: str = "Sample", model_name: str = "Model"):
    """
    Plot and save a SHAP force plot for a single prediction.
    """
    ensure_dirs()

    if idx is None:
        print(f"[SHAP] No {label} sample found — skipping force plot.")
        return

    # Matplotlib waterfall as alternative to JS force plot
    plt.figure(figsize=(12, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=np.array(shap_values)[idx],
            base_values=explainer.expected_value if not isinstance(
                explainer.expected_value, list
            ) else explainer.expected_value[1],
            data=np.array(X)[idx] if not hasattr(X, 'iloc') else X.iloc[idx].values,
            feature_names=list(X.columns) if hasattr(X, 'columns') else None
        ),
        max_display=15,
        show=False
    )
    plt.title(f'SHAP Force Plot — {label} ({model_name})',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    safe_label = label.replace(' ', '_').lower()
    safe_model = model_name.replace(' ', '_').lower()
    path = os.path.join(PLOTS_DIR, f'shap_{safe_label}_{safe_model}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[SHAP] Force plot ({label}) saved → {path}")


# ─────────────────────────────────────────
# BUILT-IN FEATURE IMPORTANCE
# ─────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str = "Model",
                            top_n: int = 10):
    """
    Plot built-in feature importances from tree-based models.
    """
    ensure_dirs()

    if not hasattr(model, 'feature_importances_'):
        print(f"[FeatImp] {type(model).__name__} has no feature_importances_. Skipping.")
        return None

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_feats = feat_imp.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    top_feats.sort_values().plot(kind='barh', ax=ax, color=colors)
    ax.set_title(f'Top {top_n} Built-in Feature Importances — {model_name}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[FeatImp] Plot saved → {path}")
    return feat_imp
