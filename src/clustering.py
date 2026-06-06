"""
K-Means Clustering
Discovers natural patient groupings independent of labelled risk categories.

Outputs:
  05_kmeans_elbow_silhouette.png
  06_kmeans_pca.png
  07_cluster_profiles.png
  08_cluster_risk_composition.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils import save_fig

RISK_COLORS    = {'Low': '#2ecc71', 'Moderate': '#f39c12',
                  'High': '#e67e22', 'Very High': '#e74c3c'}
CLUSTER_COLORS = ['#3498db', '#9b59b6', '#1abc9c', '#e74c3c']
ORDER          = ['Low', 'Moderate', 'High', 'Very High']
sns.set_theme(style='whitegrid', font_scale=1.1)

ORDINAL_MAPS = {
    'SmokingStatus':     {'Never': 0, 'Former': 1, 'Light': 2, 'Heavy': 3},
    'AlcoholConsumption':{'NoAlcohol': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3},
    'ExerciseFrequency': {'NoExercise': 0, 'Light': 1, 'Moderate': 2, 'Intensive': 3},
    'DietQuality':       {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3},
    'SleepQuality':      {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3},
    'StressLevel':       {'LowStress': 0, 'ModStress': 1, 'HighStress': 2, 'VHighStress': 3},
    'Gender':            {'Male': 0, 'Female': 1},
    'FamilyHistory':     {'NoFamHist': 0, 'FamHist': 1},
}
FEATURE_COLS = [
    'Age', 'BMI', 'SmokingStatus', 'AlcoholConsumption', 'ExerciseFrequency',
    'DietQuality', 'SleepQuality', 'StressLevel', 'SystolicBP', 'DiastolicBP',
    'Cholesterol', 'BloodSugar', 'ExistingCondition', 'FamilyHistory', 'Gender',
]


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        df_enc[col] = df_enc[col].map(mapping)
    le = LabelEncoder()
    df_enc['ExistingCondition'] = le.fit_transform(df_enc['ExistingCondition'])
    risk_map = {'Low': 0, 'Moderate': 1, 'High': 2, 'Very High': 3}
    df_enc['RiskCategory_num'] = df_enc['RiskCategory'].map(risk_map)
    return df_enc


def run_clustering(df: pd.DataFrame):
    df_enc = _encode(df)
    X = df_enc[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Elbow + Silhouette ────────────────────────────────────────────────────
    inertias, silhouettes = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    print(f"  Silhouette scores (k=2..8): {[round(s, 3) for s in silhouettes]}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(2, 9), inertias, 'o-', color='#3498db', linewidth=2, markersize=8)
    axes[0].axvline(4, color='#e74c3c', linestyle='--', alpha=0.7, label='k=4 selected')
    axes[0].set_title('Elbow Method', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].plot(range(2, 9), silhouettes, 's-', color='#9b59b6', linewidth=2, markersize=8)
    axes[1].axvline(4, color='#e74c3c', linestyle='--', alpha=0.7, label='k=4 selected')
    axes[1].set_title('Silhouette Score vs k', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()
    axes[1].spines[['top', 'right']].set_visible(False)
    plt.suptitle('K-Means: Optimal k Selection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '05_kmeans_elbow_silhouette.png')

    # ── Final clustering k=4 ─────────────────────────────────────────────────
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = km4.fit_predict(X_scaled)
    sil_final = silhouette_score(X_scaled, cluster_labels)
    print(f"  Final silhouette (k=4): {sil_final:.3f}")

    df = df.copy()
    df['Cluster'] = cluster_labels
    df_enc['Cluster'] = cluster_labels

    # ── PCA 2D projection ─────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for c in range(4):
        mask = cluster_labels == c
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=CLUSTER_COLORS[c], label=f'Cluster {c}',
                        alpha=0.6, s=28, edgecolors='none')
    axes[0].set_title(f'K-Means Clusters (k=4) | Sil={sil_final:.3f}', fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca_var[0]*100:.1f}% var)')
    axes[0].set_ylabel(f'PC2 ({pca_var[1]*100:.1f}% var)')
    axes[0].legend(title='Cluster')
    axes[0].spines[['top', 'right']].set_visible(False)

    for r in ORDER:
        mask = df['RiskCategory'] == r
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=RISK_COLORS[r], label=r,
                        alpha=0.6, s=28, edgecolors='none')
    axes[1].set_title('Actual Risk (PCA projection)', fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca_var[0]*100:.1f}% var)')
    axes[1].set_ylabel(f'PC2 ({pca_var[1]*100:.1f}% var)')
    axes[1].legend(title='Risk')
    axes[1].spines[['top', 'right']].set_visible(False)
    plt.suptitle('PCA Projection: K-Means Clusters vs Actual Risk',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '06_kmeans_pca.png')

    # ── Cluster profile heatmap ───────────────────────────────────────────────
    profile_cols = ['Age', 'BMI', 'SystolicBP', 'Cholesterol', 'BloodSugar']
    cp = df.groupby('Cluster')[profile_cols].mean()
    cp_norm = (cp - cp.min()) / (cp.max() - cp.min())

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(cp_norm.T, annot=cp.T.round(1), fmt='.1f',
                cmap='YlOrRd', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Normalised (0–1)'})
    ax.set_title('Cluster Profiles — Mean Clinical Measurements', fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Feature')
    ax.set_xticklabels([f'C{i}' for i in range(4)])
    plt.tight_layout()
    save_fig(fig, '07_cluster_profiles.png')

    # ── Risk composition per cluster ──────────────────────────────────────────
    risk_comp = pd.crosstab(df['Cluster'], df['RiskCategory'], normalize='index') * 100
    print("\n  Cluster → Risk composition (%):")
    print(risk_comp.round(1).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    risk_comp[ORDER].plot(kind='bar', stacked=True, ax=ax,
                          color=[RISK_COLORS[r] for r in ORDER],
                          edgecolor='white', linewidth=0.5, width=0.6)
    ax.set_title('Risk Category Composition per Cluster', fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Percentage (%)')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Risk Category', bbox_to_anchor=(1.01, 1))
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_fig(fig, '08_cluster_risk_composition.png')

    return df, df_enc, X_scaled, X_pca
