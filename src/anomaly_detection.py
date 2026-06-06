"""
Anomaly Detection
Two complementary methods — Isolation Forest and Local Outlier Factor.
Consensus anomalies are flagged by both.

Outputs:
  11_anomaly_pca.png
  12_anomaly_scores.png
  13_anomaly_by_risk.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.utils import save_fig

RISK_COLORS = {'Low': '#2ecc71', 'Moderate': '#f39c12',
               'High': '#e67e22', 'Very High': '#e74c3c'}
ORDER = ['Low', 'Moderate', 'High', 'Very High']


def run_anomaly_detection(df_enc, X_scaled, X_pca, df_raw):
    # ── Isolation Forest ──────────────────────────────────────────────────────
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso_labels = iso.fit_predict(X_scaled)   # -1 = anomaly
    iso_scores = iso.decision_function(X_scaled)

    # ── Local Outlier Factor ──────────────────────────────────────────────────
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_labels = lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_

    n_iso = (iso_labels == -1).sum()
    n_lof = (lof_labels == -1).sum()
    n_consensus = ((iso_labels == -1) & (lof_labels == -1)).sum()
    print(f"  Isolation Forest anomalies : {n_iso} ({n_iso/len(df_enc)*100:.1f}%)")
    print(f"  LOF anomalies              : {n_lof} ({n_lof/len(df_enc)*100:.1f}%)")
    print(f"  Consensus (both methods)   : {n_consensus}")

    df = df_raw.copy()
    df['iso_label']  = iso_labels
    df['lof_label']  = lof_labels
    df['is_anomaly'] = ((iso_labels == -1) & (lof_labels == -1)).astype(int)

    # ── Plot 1: PCA scatter ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_mask, a_mask = iso_labels == 1, iso_labels == -1
    axes[0].scatter(X_pca[n_mask, 0], X_pca[n_mask, 1],
                    c='#3498db', alpha=0.4, s=20, label='Normal', edgecolors='none')
    axes[0].scatter(X_pca[a_mask, 0], X_pca[a_mask, 1],
                    c='#e74c3c', alpha=0.9, s=60, marker='X',
                    label=f'Anomaly (n={n_iso})', zorder=5)
    axes[0].set_title('Isolation Forest — Anomaly Detection', fontweight='bold')
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    axes[0].legend(); axes[0].spines[['top', 'right']].set_visible(False)

    n_lof_m, a_lof_m = lof_labels == 1, lof_labels == -1
    axes[1].scatter(X_pca[n_lof_m, 0], X_pca[n_lof_m, 1],
                    c='#3498db', alpha=0.4, s=20, label='Normal', edgecolors='none')
    axes[1].scatter(X_pca[a_lof_m, 0], X_pca[a_lof_m, 1],
                    c='#e67e22', alpha=0.9, s=60, marker='X',
                    label=f'Anomaly (n={n_lof})', zorder=5)
    axes[1].set_title('Local Outlier Factor — Anomaly Detection', fontweight='bold')
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].legend(); axes[1].spines[['top', 'right']].set_visible(False)

    plt.suptitle('Anomaly Detection: PCA Projection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '11_anomaly_pca.png')

    # ── Plot 2: Score distributions ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(iso_scores[iso_labels == 1],  bins=30, alpha=0.7,
                 color='#3498db', label='Normal')
    axes[0].hist(iso_scores[iso_labels == -1], bins=15, alpha=0.8,
                 color='#e74c3c', label='Anomaly')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1.5,
                    label='Decision boundary')
    axes[0].set_title('Isolation Forest Score Distribution', fontweight='bold')
    axes[0].set_xlabel('Anomaly Score'); axes[0].set_ylabel('Count')
    axes[0].legend(); axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].hist(lof_scores[lof_labels == 1],  bins=30, alpha=0.7,
                 color='#3498db', label='Normal')
    axes[1].hist(lof_scores[lof_labels == -1], bins=15, alpha=0.8,
                 color='#e67e22', label='Anomaly')
    axes[1].set_title('LOF Score Distribution', fontweight='bold')
    axes[1].set_xlabel('LOF Score'); axes[1].set_ylabel('Count')
    axes[1].legend(); axes[1].spines[['top', 'right']].set_visible(False)

    plt.suptitle('Anomaly Score Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '12_anomaly_scores.png')

    # ── Plot 3: Anomaly rate by risk category ─────────────────────────────────
    anom_risk  = df[df['is_anomaly'] == 1]['RiskCategory'].value_counts().reindex(ORDER, fill_value=0)
    total_risk = df['RiskCategory'].value_counts().reindex(ORDER)
    anom_pct   = (anom_risk / total_risk * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ORDER, anom_pct,
                  color=[RISK_COLORS[r] for r in ORDER],
                  edgecolor='white', linewidth=1.5, width=0.55)
    for bar, val, n in zip(bars, anom_pct, anom_risk):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f'{val}%\n(n={n})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Consensus Anomalies by Risk Category\n(% of patients in each category)',
                 fontweight='bold')
    ax.set_xlabel('Risk Category'); ax.set_ylabel('Anomaly Rate (%)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_fig(fig, '13_anomaly_by_risk.png')

    # ── Profile comparison ────────────────────────────────────────────────────
    num_cols = ['Age', 'BMI', 'SystolicBP', 'Cholesterol', 'BloodSugar']
    print("\n  Anomaly vs Normal — mean clinical values:")
    compare = pd.DataFrame({
        'Anomalies': df[df['is_anomaly'] == 1][num_cols].mean().round(1),
        'Normal':    df[df['is_anomaly'] == 0][num_cols].mean().round(1),
    })
    print(compare.to_string())
