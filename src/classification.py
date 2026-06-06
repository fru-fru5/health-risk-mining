"""
Classification — Random Forest, Gradient Boosting, Decision Tree
Used as one component of the data mining pipeline, not the sole objective.

Outputs:
  14_model_comparison.png
  15_confusion_matrix.png
  16_feature_importance.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from src.utils import save_fig

ORDER = ['Low', 'Moderate', 'High', 'Very High']

FEATURE_COLS = [
    'Age', 'BMI', 'SmokingStatus', 'AlcoholConsumption', 'ExerciseFrequency',
    'DietQuality', 'SleepQuality', 'StressLevel', 'SystolicBP', 'DiastolicBP',
    'Cholesterol', 'BloodSugar', 'ExistingCondition', 'FamilyHistory', 'Gender',
]

READABLE = {
    'Age': 'Age', 'BMI': 'BMI',
    'SmokingStatus': 'Smoking Status',
    'AlcoholConsumption': 'Alcohol Consumption',
    'ExerciseFrequency': 'Exercise Frequency',
    'DietQuality': 'Diet Quality',
    'SleepQuality': 'Sleep Quality',
    'StressLevel': 'Stress Level',
    'SystolicBP': 'Systolic BP',
    'DiastolicBP': 'Diastolic BP',
    'Cholesterol': 'Cholesterol',
    'BloodSugar': 'Blood Sugar',
    'ExistingCondition': 'Existing Condition',
    'FamilyHistory': 'Family History',
    'Gender': 'Gender',
}


def run_classification(df_enc: pd.DataFrame):
    X = df_enc[FEATURE_COLS].values.astype(float)
    y = df_enc['RiskCategory_num'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest':     RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Decision Tree':     DecisionTreeClassifier(max_depth=8, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        acc       = (y_pred == y_test).mean()
        results[name] = {
            'model': model, 'y_pred': y_pred,
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std':  cv_scores.std(),
        }
        print(f"  {name:<22} acc={acc:.3f}  CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n  Best model: {best_name} ({results[best_name]['accuracy']*100:.1f}%)")

    rf    = results['Random Forest']['model']
    rf_pred = results['Random Forest']['y_pred']

    print("\n  Classification Report (Random Forest):")
    print(classification_report(y_test, rf_pred, target_names=ORDER))

    # ── Plot 1: Model comparison ──────────────────────────────────────────────
    model_names = list(results.keys())
    accs = [results[n]['accuracy'] * 100 for n in model_names]
    cvs  = [results[n]['cv_mean']  * 100 for n in model_names]
    stds = [results[n]['cv_std']   * 100 for n in model_names]
    x = np.arange(len(model_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, accs, w, label='Test Accuracy',
                color=['#3498db', '#9b59b6', '#1abc9c'],
                edgecolor='white', linewidth=1)
    b2 = ax.bar(x + w / 2, cvs, w, label='CV Score (mean)',
                color=['#2980b9', '#8e44ad', '#16a085'],
                edgecolor='white', linewidth=1,
                yerr=stds, capsize=5,
                error_kw=dict(elinewidth=1.5, ecolor='#333'))
    for bar, val in list(zip(b1, accs)) + list(zip(b2, cvs)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(50, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=13)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_fig(fig, '14_model_comparison.png')

    # ── Plot 2: Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, rf_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ORDER).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix — Random Forest', fontweight='bold', fontsize=13)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    save_fig(fig, '15_confusion_matrix.png')

    # ── Plot 3: Feature importance ────────────────────────────────────────────
    feat_df = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': rf.feature_importances_,
        'label':      [READABLE[f] for f in FEATURE_COLS],
    }).sort_values('importance', ascending=True)

    colors = ['#e74c3c' if v > 0.10 else '#e67e22' if v > 0.07 else '#3498db'
              for v in feat_df['importance']]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(feat_df['label'], feat_df['importance'] * 100,
                   color=colors, edgecolor='white', linewidth=0.8, height=0.65)
    for bar, val in zip(bars, feat_df['importance'] * 100):
        ax.text(bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.set_xlabel('Feature Importance (%)', fontsize=12)
    ax.set_title('Feature Importance — Random Forest\n(contribution to risk prediction)',
                 fontweight='bold', fontsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    save_fig(fig, '16_feature_importance.png')

    # Print ranked table
    print("\n  Feature Importance Ranking:")
    for i, (_, row) in enumerate(feat_df.sort_values('importance', ascending=False).iterrows()):
        print(f"  {i+1:2d}. {row['label']:<25} {row['importance']*100:.1f}%")
