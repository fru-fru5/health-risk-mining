"""
Data Generator
Produces a synthetic 1,000-patient health risk dataset
matching the distributions described in the CEC417 project report.
"""

import numpy as np
import pandas as pd


def generate_dataset(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Return a DataFrame of n synthetic patient health assessments."""
    rng = np.random.default_rng(random_state)

    # ── Target distribution ──────────────────────────────────────────────────
    risk_cats = rng.choice(
        ['Low', 'Moderate', 'High', 'Very High'],
        size=n,
        p=[0.287, 0.398, 0.241, 0.074]
    )

    # ── Age (mean shifts with risk) ──────────────────────────────────────────
    age_means = {'Low': 35.2, 'Moderate': 47.8, 'High': 58.6, 'Very High': 65.3}
    age_stds  = {'Low': 10,   'Moderate': 12,   'High': 11,   'Very High': 9}
    age = np.array([
        rng.normal(age_means[r], age_stds[r]) for r in risk_cats
    ]).clip(18, 85).astype(int)

    # ── Smoking ──────────────────────────────────────────────────────────────
    smoking_probs = {
        'Low':      [0.70, 0.20, 0.07, 0.03],
        'Moderate': [0.30, 0.35, 0.25, 0.10],
        'High':     [0.10, 0.20, 0.40, 0.30],
        'Very High':[0.05, 0.10, 0.07, 0.78],
    }
    smoking = [rng.choice(['Never', 'Former', 'Light', 'Heavy'],
                          p=smoking_probs[r]) for r in risk_cats]

    # ── Exercise ─────────────────────────────────────────────────────────────
    exercise_probs = {
        'Low':      [0.05, 0.15, 0.45, 0.35],
        'Moderate': [0.15, 0.30, 0.35, 0.20],
        'High':     [0.35, 0.35, 0.20, 0.10],
        'Very High':[0.45, 0.30, 0.15, 0.10],
    }
    exercise = [rng.choice(['NoExercise', 'Light', 'Moderate', 'Intensive'],
                           p=exercise_probs[r]) for r in risk_cats]

    # ── BMI ──────────────────────────────────────────────────────────────────
    bmi_means = {'Low': 22, 'Moderate': 26, 'High': 30, 'Very High': 34}
    bmi_stds  = {'Low': 3,  'Moderate': 4,  'High': 5,  'Very High': 5}
    bmi = np.array([
        rng.normal(bmi_means[r], bmi_stds[r]) for r in risk_cats
    ]).clip(15, 50).round(1)

    # ── Diet ─────────────────────────────────────────────────────────────────
    diet_probs = {
        'Low':      [0.02, 0.08, 0.30, 0.60],
        'Moderate': [0.10, 0.25, 0.40, 0.25],
        'High':     [0.30, 0.40, 0.20, 0.10],
        'Very High':[0.50, 0.30, 0.15, 0.05],
    }
    diet = [rng.choice(['Poor', 'Fair', 'Good', 'Excellent'],
                       p=diet_probs[r]) for r in risk_cats]

    # ── Sleep ────────────────────────────────────────────────────────────────
    sleep_probs = {
        'Low':      [0.03, 0.12, 0.40, 0.45],
        'Moderate': [0.12, 0.28, 0.38, 0.22],
        'High':     [0.30, 0.38, 0.22, 0.10],
        'Very High':[0.48, 0.30, 0.15, 0.07],
    }
    sleep = [rng.choice(['Poor', 'Fair', 'Good', 'Excellent'],
                        p=sleep_probs[r]) for r in risk_cats]

    # ── Stress ───────────────────────────────────────────────────────────────
    stress_probs = {
        'Low':      [0.45, 0.35, 0.15, 0.05],
        'Moderate': [0.20, 0.35, 0.30, 0.15],
        'High':     [0.05, 0.20, 0.40, 0.35],
        'Very High':[0.03, 0.10, 0.30, 0.57],
    }
    stress = [rng.choice(['LowStress', 'ModStress', 'HighStress', 'VHighStress'],
                         p=stress_probs[r]) for r in risk_cats]

    # ── Alcohol ──────────────────────────────────────────────────────────────
    alcohol_probs = {
        'Low':      [0.40, 0.40, 0.15, 0.05],
        'Moderate': [0.25, 0.35, 0.28, 0.12],
        'High':     [0.10, 0.25, 0.38, 0.27],
        'Very High':[0.05, 0.15, 0.30, 0.50],
    }
    alcohol = [rng.choice(['NoAlcohol', 'Light', 'Moderate', 'Heavy'],
                          p=alcohol_probs[r]) for r in risk_cats]

    # ── Clinical measurements ────────────────────────────────────────────────
    def clinical(means, stds, lo, hi, dtype=int):
        vals = np.array([rng.normal(means[r], stds[r]) for r in risk_cats]).clip(lo, hi)
        return vals.astype(dtype)

    systolic   = clinical({'Low':115,'Moderate':125,'High':140,'Very High':155},
                          {'Low':10, 'Moderate':12, 'High':15, 'Very High':15}, 90, 180)
    diastolic  = clinical({'Low':75, 'Moderate':82, 'High':90, 'Very High':98},
                          {'Low':8,  'Moderate':9,  'High':10, 'Very High':10}, 60, 120)
    cholesterol= clinical({'Low':185,'Moderate':210,'High':240,'Very High':265},
                          {'Low':25, 'Moderate':28, 'High':28, 'Very High':25}, 150, 300)
    blood_sugar= clinical({'Low':88, 'Moderate':100,'High':125,'Very High':155},
                          {'Low':10, 'Moderate':15, 'High':20, 'Very High':25}, 70, 200)

    # ── Demographics & history ───────────────────────────────────────────────
    gender = rng.choice(['Male', 'Female'], size=n)

    conditions    = ['NoCond','Hypertension','Diabetes','HeartDisease','Asthma','Arthritis']
    cond_probs    = {
        'Low':      [0.70, 0.10, 0.08, 0.05, 0.04, 0.03],
        'Moderate': [0.35, 0.20, 0.18, 0.12, 0.08, 0.07],
        'High':     [0.15, 0.25, 0.22, 0.18, 0.12, 0.08],
        'Very High':[0.05, 0.20, 0.25, 0.22, 0.15, 0.13],
    }
    existing = [rng.choice(conditions, p=cond_probs[r]) for r in risk_cats]

    fam_p = {'Low': 0.15, 'Moderate': 0.35, 'High': 0.55, 'Very High': 0.75}
    family = ['FamHist' if rng.random() < fam_p[r] else 'NoFamHist' for r in risk_cats]

    df = pd.DataFrame({
        'Age': age, 'Gender': gender, 'BMI': bmi,
        'SmokingStatus': smoking,
        'AlcoholConsumption': alcohol,
        'ExerciseFrequency': exercise,
        'DietQuality': diet,
        'SleepQuality': sleep,
        'StressLevel': stress,
        'SystolicBP': systolic,
        'DiastolicBP': diastolic,
        'Cholesterol': cholesterol,
        'BloodSugar': blood_sugar,
        'ExistingCondition': existing,
        'FamilyHistory': family,
        'RiskCategory': risk_cats,
    })

    df.to_csv('data/health_data.csv', index=False)
    print(f"  Dataset generated: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Class distribution:\n{df['RiskCategory'].value_counts().to_string()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    return df
