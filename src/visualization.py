"""
visualization.py
----------------
Reusable EDA and reporting visualization utilities.
All plots are saved to notebooks/plots/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'plots')


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ─── STYLE ────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
PALETTE = {'legit': '#2ecc71', 'fraud': '#e74c3c'}


# ─────────────────────────────────────────
# CLASS IMBALANCE
# ─────────────────────────────────────────

def plot_class_distribution(y: pd.Series, dataset_name: str = "Dataset"):
    """Bar chart showing class distribution with counts and percentages."""
    ensure_dirs()

    counts = y.value_counts().sort_index()
    labels = ['Legitimate', 'Fraud']
    colors = [PALETTE['legit'], PALETTE['fraud']]
    total = len(y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='white',
                       linewidth=1.5, width=0.5)
    for bar, cnt in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total * 0.005,
                     f'{cnt:,}\n({cnt/total*100:.2f}%)',
                     ha='center', fontsize=11, fontweight='bold')
    axes[0].set_title(f'{dataset_name} — Class Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Transactions')
    axes[0].set_ylim(0, counts.max() * 1.15)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct='%1.2f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    axes[1].set_title(f'{dataset_name} — Class Ratio', fontsize=13, fontweight='bold')

    plt.suptitle(f'Class Imbalance Analysis — {dataset_name}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'class_dist_{dataset_name.replace(" ", "_").lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────
# UNIVARIATE DISTRIBUTIONS
# ─────────────────────────────────────────

def plot_numerical_distributions(df: pd.DataFrame, columns: list,
                                  dataset_name: str = "Dataset"):
    """Grid of histograms for numerical features."""
    ensure_dirs()

    n = len(columns)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        axes[i].hist(df[col].dropna(), bins=40, color='steelblue',
                     edgecolor='white', linewidth=0.5, alpha=0.85)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{dataset_name} — Numerical Feature Distributions',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'num_dist_{dataset_name.replace(" ", "_").lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {path}")


def plot_categorical_distributions(df: pd.DataFrame, columns: list,
                                    dataset_name: str = "Dataset"):
    """Bar charts for categorical features."""
    ensure_dirs()

    n = len(columns)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        vc = df[col].value_counts().head(15)
        axes[i].barh(vc.index.astype(str), vc.values, color='steelblue',
                     edgecolor='white', linewidth=0.5)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Count')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{dataset_name} — Categorical Feature Distributions',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'cat_dist_{dataset_name.replace(" ", "_").lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────
# BIVARIATE: FEATURE vs. TARGET
# ─────────────────────────────────────────

def plot_bivariate(df: pd.DataFrame, columns: list, target: str,
                   dataset_name: str = "Dataset"):
    """Violin + box plots comparing feature distributions across classes."""
    ensure_dirs()

    n = len(columns)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    colors = [PALETTE['legit'], PALETTE['fraud']]

    for i, col in enumerate(columns):
        data = [df[df[target] == c][col].dropna().values for c in sorted(df[target].unique())]
        parts = axes[i].violinplot(data, positions=[0, 1], showmedians=True)
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.7)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Legitimate', 'Fraud'])
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Value')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{dataset_name} — Feature vs. Class (Bivariate)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'bivariate_{dataset_name.replace(" ", "_").lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────
# GEOLOCATION
# ─────────────────────────────────────────

def plot_fraud_by_country(df: pd.DataFrame, top_n: int = 15):
    """Horizontal bar chart of fraud rate and count by top countries."""
    ensure_dirs()

    country_stats = df.groupby('country').agg(
        total=('class', 'count'),
        fraud=('class', 'sum')
    ).reset_index()
    country_stats['fraud_rate'] = country_stats['fraud'] / country_stats['total'] * 100
    top_by_fraud = country_stats.nlargest(top_n, 'fraud')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Count
    colors_count = plt.cm.Reds(
        np.linspace(0.4, 0.9, len(top_by_fraud))
    )
    axes[0].barh(top_by_fraud['country'], top_by_fraud['fraud'],
                 color=colors_count, edgecolor='white')
    axes[0].set_title(f'Top {top_n} Countries by Fraud Count',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Fraud Transactions')
    axes[0].invert_yaxis()

    # Fraud rate
    top_by_rate = country_stats[country_stats['total'] >= 20].nlargest(top_n, 'fraud_rate')
    colors_rate = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_by_rate)))
    axes[1].barh(top_by_rate['country'], top_by_rate['fraud_rate'],
                 color=colors_rate, edgecolor='white')
    axes[1].set_title(f'Top {top_n} Countries by Fraud Rate % (min 20 txns)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Fraud Rate (%)')
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))

    plt.suptitle('Geolocation Fraud Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, 'fraud_by_country.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {path}")


# ─────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, dataset_name: str = "Dataset",
                              max_cols: int = 30):
    """Correlation heatmap for numerical features."""
    ensure_dirs()

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > max_cols:
        num_df = num_df.iloc[:, :max_cols]

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.7})
    ax.set_title(f'{dataset_name} — Feature Correlation Matrix',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'correlation_{dataset_name.replace(" ", "_").lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[Plot] Saved → {path}")
