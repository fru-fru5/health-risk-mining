"""
Exploratory Data Analysis
Produces 4 figures:
  01_risk_distribution.png
  02_age_by_risk.png
  03_correlation_heatmap.png
  04_lifestyle_vs_risk.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import save_fig

RISK_COLORS = {'Low': '#2ecc71', 'Moderate': '#f39c12',
               'High': '#e67e22', 'Very High': '#e74c3c'}
ORDER = ['Low', 'Moderate', 'High', 'Very High']
sns.set_theme(style='whitegrid', font_scale=1.1)


def run_eda(df: pd.DataFrame):
    _plot_risk_distribution(df)
    _plot_age_by_risk(df)
    _plot_correlation_heatmap(df)
    _plot_lifestyle_vs_risk(df)

    print("\n  Clinical means by risk category:")
    print(df.groupby('RiskCategory')[
        ['Age', 'BMI', 'SystolicBP', 'Cholesterol', 'BloodSugar']
    ].mean().round(1).to_string())


def _plot_risk_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['RiskCategory'].value_counts().reindex(ORDER)
    bars = ax.bar(ORDER, counts,
                  color=[RISK_COLORS[r] for r in ORDER],
                  edgecolor='white', linewidth=1.5, width=0.6)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f'{val}\n({val / len(df) * 100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Risk Category Distribution (N=1,000)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Number of Patients')
    ax.set_xlabel('Risk Category')
    ax.set_ylim(0, max(counts) * 1.22)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_fig(fig, '01_risk_distribution.png')


def _plot_age_by_risk(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df[df['RiskCategory'] == r]['Age'].values for r in ORDER]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color='white', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4, alpha=0.4))
    for patch, r in zip(bp['boxes'], ORDER):
        patch.set_facecolor(RISK_COLORS[r])
        patch.set_alpha(0.85)
    ax.set_xticklabels(ORDER)
    ax.set_title('Age Distribution by Risk Category',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Age (years)')
    ax.set_xlabel('Risk Category')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_fig(fig, '02_age_by_risk.png')


def _plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP',
                'Cholesterol', 'BloodSugar']
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=0, ax=ax, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap — Numerical Features',
                 fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    save_fig(fig, '03_correlation_heatmap.png')


def _plot_lifestyle_vs_risk(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Smoking
    sm_order = ['Never', 'Former', 'Light', 'Heavy']
    sm = pd.crosstab(df['SmokingStatus'], df['RiskCategory']).reindex(sm_order)
    sm_pct = sm.div(sm.sum(axis=1), axis=0) * 100
    sm_pct[ORDER].plot(kind='bar', stacked=True, ax=axes[0],
                       color=[RISK_COLORS[r] for r in ORDER],
                       edgecolor='white', linewidth=0.5)
    axes[0].set_title('Smoking Status vs Risk Category', fontweight='bold')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(title='Risk', bbox_to_anchor=(1.01, 1))
    axes[0].spines[['top', 'right']].set_visible(False)

    # Exercise
    ex_order  = ['NoExercise', 'Light', 'Moderate', 'Intensive']
    ex_labels = ['None', 'Light', 'Moderate', 'Intensive']
    ex = pd.crosstab(df['ExerciseFrequency'], df['RiskCategory']).reindex(ex_order)
    ex_pct = ex.div(ex.sum(axis=1), axis=0) * 100
    ex_pct[ORDER].plot(kind='bar', stacked=True, ax=axes[1],
                       color=[RISK_COLORS[r] for r in ORDER],
                       edgecolor='white', linewidth=0.5)
    axes[1].set_xticklabels(ex_labels, rotation=0)
    axes[1].set_title('Exercise Frequency vs Risk Category', fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_xlabel('')
    axes[1].legend(title='Risk', bbox_to_anchor=(1.01, 1))
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.suptitle('Lifestyle Factors vs Risk Category',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, '04_lifestyle_vs_risk.png')
