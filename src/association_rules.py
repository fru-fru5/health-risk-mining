"""
Association Rule Mining (Apriori)
Discovers co-occurring lifestyle and clinical risk factor patterns.

Outputs:
  09_association_rules_lift.png
  10_rules_support_confidence.png
  association_rules.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from src.utils import save_fig

sns.set_theme(style='whitegrid', font_scale=1.1)


def run_association_mining(df: pd.DataFrame):
    df_t = _discretise(df)
    transactions = _build_transactions(df_t)
    print(f"  Transaction matrix: {transactions.shape}")

    # ── Frequent itemsets ─────────────────────────────────────────────────────
    frequent = apriori(transactions, min_support=0.10,
                       use_colnames=True, max_len=4)
    frequent['length'] = frequent['itemsets'].apply(len)
    print(f"  Frequent itemsets (sup≥0.10): {len(frequent)}")

    # ── Association rules ─────────────────────────────────────────────────────
    rules = association_rules(frequent, metric='lift',
                              min_threshold=1.5)
    rules = rules[rules['confidence'] >= 0.60].sort_values('lift', ascending=False)
    print(f"  Rules (lift≥1.5, conf≥0.60): {len(rules)}")

    # Filter for rules whose consequent is a risk category
    risk_rules = rules[rules['consequents'].apply(
        lambda x: any('RiskCategory' in str(i) for i in x)
    )].copy()
    print(f"  Risk-targeting rules: {len(risk_rules)}")

    # Readable labels
    def fmt(itemset):
        return ' + '.join(sorted([
            str(i).replace('RiskCategory_', 'Risk=')
                  .replace('_', ' ')
                  .replace('NoExercise', 'No Exercise')
                  .replace('VHighStress', 'Very High Stress')
                  .replace('NoAlcohol', 'No Alcohol')
            for i in itemset
        ]))

    risk_rules['antecedent_str'] = risk_rules['antecedents'].apply(fmt)
    risk_rules['consequent_str'] = risk_rules['consequents'].apply(fmt)
    top15 = risk_rules.head(15).copy()

    # ── Plot 1: Lift bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top15)))
    bars = ax.barh(range(len(top15)), top15['lift'],
                   color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels([f'R{i+1}' for i in range(len(top15))], fontsize=10)
    ax.set_xlabel('Lift', fontsize=12)
    ax.set_title('Top 15 Association Rules — Risk Category (by Lift)',
                 fontweight='bold', fontsize=13)
    for bar, row in zip(bars, top15.itertuples()):
        ax.text(bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'Conf={row.confidence:.2f}',
                va='center', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, '09_association_rules_lift.png')

    # ── Plot 2: Support vs Confidence scatter ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(risk_rules['support'], risk_rules['confidence'],
                    c=risk_rules['lift'], cmap='RdYlGn', s=60,
                    alpha=0.75, edgecolors='grey', linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Lift')
    ax.set_xlabel('Support', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Association Rules: Support vs Confidence\n(colour = Lift)',
                 fontweight='bold', fontsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_fig(fig, '10_rules_support_confidence.png')

    # ── Save top rules ────────────────────────────────────────────────────────
    top15[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].to_csv(
        'outputs/association_rules.csv', index=False)
    print("\n  Top 5 risk rules:")
    for i, (_, row) in enumerate(top15.head(5).iterrows()):
        print(f"  R{i+1}: {row['antecedent_str'][:60]}...")
        print(f"      → {row['consequent_str']}  "
              f"[sup={row['support']:.2f}, conf={row['confidence']:.2f}, lift={row['lift']:.2f}]")


def _discretise(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['AgeGroup']  = pd.cut(d['Age'],
                             bins=[0, 40, 55, 70, 100],
                             labels=['Young', 'MiddleAge', 'Senior', 'Elderly'])
    d['BMI_Cat']   = pd.cut(d['BMI'],
                             bins=[0, 18.5, 25, 30, 50],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    d['SBP_Cat']   = pd.cut(d['SystolicBP'],
                             bins=[0, 120, 130, 140, 200],
                             labels=['Normal_BP', 'Elevated_BP', 'Stage1_HTN', 'Stage2_HTN'])
    d['Chol_Cat']  = pd.cut(d['Cholesterol'],
                             bins=[0, 200, 240, 400],
                             labels=['Desirable_Chol', 'Borderline_Chol', 'High_Chol'])
    d['Sugar_Cat'] = pd.cut(d['BloodSugar'],
                             bins=[0, 100, 126, 300],
                             labels=['Normal_Sugar', 'Prediabetic', 'Diabetic_Sugar'])
    return d


def _build_transactions(df_t: pd.DataFrame) -> pd.DataFrame:
    item_cols = [
        'AgeGroup', 'BMI_Cat', 'SBP_Cat', 'Chol_Cat', 'Sugar_Cat',
        'SmokingStatus', 'AlcoholConsumption', 'ExerciseFrequency',
        'DietQuality', 'SleepQuality', 'StressLevel',
        'ExistingCondition', 'FamilyHistory', 'RiskCategory',
    ]
    return pd.get_dummies(df_t[item_cols].astype(str)).astype(bool)
