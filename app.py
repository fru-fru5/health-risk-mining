import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Risk Assessment System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
RISK_COLORS   = {'Low':'#2ecc71','Moderate':'#f39c12','High':'#e67e22','Very High':'#e74c3c'}
RISK_BG       = {'Low':'#eafaf1','Moderate':'#fef9e7','High':'#fdf2e9','Very High':'#fdedec'}
CLUSTER_COLORS= ['#3498db','#9b59b6','#1abc9c','#e74c3c']
ORDER         = ['Low','Moderate','High','Very High']

st.markdown("""
<style>
    .risk-box {
        border-radius: 12px; padding: 24px; text-align: center;
        margin: 10px 0; border: 2px solid;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px; text-align: center; margin: 6px 0;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700;
        margin-top: 1.5rem; margin-bottom: 0.5rem;
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_prepare():
    df = pd.read_csv('data/health_risk_assessment.csv')

    # Handle missing values (includes string 'None' in categorical columns)
    for col in ['AlcoholConsumption', 'ExerciseFrequency']:
        df[col] = df[col].replace('None', np.nan).fillna(df[col].mode()[0])
    df['ExistingCondition']  = df['ExistingCondition'].fillna('None')

    # Feature engineering
    df['BP_Category']    = np.where(
        (df['SystolicBP'] > 140) | (df['DiastolicBP'] > 90), 'High_BP', 'Normal_BP')
    df['Chol_Category']  = pd.cut(df['Cholesterol'],
        bins=[0,200,240,400], labels=['Desirable','Borderline','High'])
    df['BMI_Category']   = pd.cut(df['BMI'],
        bins=[0,18.5,25,30,50], labels=['Underweight','Normal','Overweight','Obese'])
    df['AgeGroup']       = pd.cut(df['Age'],
        bins=[0,40,55,70,100], labels=['Young','MiddleAge','Senior','Elderly'])
    df['Sugar_Category'] = pd.cut(df['BloodSugar'],
        bins=[0,100,126,300], labels=['Normal','Prediabetic','Diabetic'])
    return df


@st.cache_resource
def train_models(df):
    ORDINAL_MAPS = {
        'SmokingStatus':     {'Never':0,'Former':1,'Current - Light':2,'Current - Heavy':3},
        'AlcoholConsumption':{'Light':0,'Moderate':1,'Heavy':2},
        'ExerciseFrequency': {'Rare':0,'Moderate':1,'Regular':2,'Intensive':3},
        'DietQuality':       {'Poor':0,'Fair':1,'Good':2,'Excellent':3},
        'SleepQuality':      {'Poor':0,'Fair':1,'Good':2,'Excellent':3},
        'StressLevel':       {'Low':0,'Moderate':1,'High':2,'Very High':3},
        'Gender':            {'Male':0,'Female':1},
        'FamilyHistory':     {'No':0,'Yes':1},
    }
    FEATURE_COLS = ['Age','BMI','SmokingStatus','AlcoholConsumption','ExerciseFrequency',
                    'DietQuality','SleepQuality','StressLevel','SystolicBP','DiastolicBP',
                    'Cholesterol','BloodSugar','ExistingCondition','FamilyHistory','Gender']

    df_enc = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        df_enc[col] = df_enc[col].map(mapping)
    le = LabelEncoder()
    df_enc['ExistingCondition'] = le.fit_transform(df_enc['ExistingCondition'])
    df_enc['RiskCategory_num']  = df_enc['RiskCategory'].map(
        {'Low':0,'Moderate':1,'High':2,'Very High':3})

    X = df_enc[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_scaled, df_enc['RiskCategory_num'].values)

    # K-Means
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_scaled)
    df['Cluster'] = km.labels_

    # Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_scaled)

    # PCA for visualisation
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return rf, km, iso, scaler, le, ORDINAL_MAPS, FEATURE_COLS, df_enc, X_scaled, X_pca, df


@st.cache_data
def mine_rules(df):
    """Custom Apriori — no external library needed."""
    item_cols = ['AgeGroup','BMI_Category','BP_Category','Chol_Category','Sugar_Category',
                 'SmokingStatus','AlcoholConsumption','ExerciseFrequency',
                 'DietQuality','SleepQuality','StressLevel',
                 'ExistingCondition','FamilyHistory','RiskCategory']

    transactions = pd.get_dummies(df[item_cols].astype(str)).astype(bool)
    MIN_SUP, MIN_CONF, MIN_LIFT = 0.10, 0.60, 1.5

    freq = {}
    for col in transactions.columns:
        sup = transactions[col].mean()
        if sup >= MIN_SUP:
            freq[frozenset([col])] = sup

    for c1, c2 in combinations(list(freq.keys()), 2):
        pair = c1 | c2
        if len(pair) == 2:
            sup = transactions[list(pair)].all(axis=1).mean()
            if sup >= MIN_SUP:
                freq[pair] = sup

    rules = []
    for itemset, isup in freq.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for cons_t in combinations(list(itemset), r):
                cons = frozenset(cons_t)
                ant  = itemset - cons
                if not ant:
                    continue
                ant_sup  = freq.get(ant, 0)
                if ant_sup == 0:
                    continue
                conf = isup / ant_sup
                csup = freq.get(cons, transactions[list(cons)].all(axis=1).mean())
                lift = conf / csup if csup > 0 else 0
                if conf >= MIN_CONF and lift >= MIN_LIFT:
                    rules.append({'antecedents': ant, 'consequents': cons,
                                  'support': round(isup,4),
                                  'confidence': round(conf,4),
                                  'lift': round(lift,4)})

    rules_df = pd.DataFrame(rules).sort_values('lift', ascending=False).reset_index(drop=True)
    risk_rules = rules_df[rules_df['consequents'].apply(
        lambda x: any('RiskCategory' in str(i) for i in x))].copy()

    def fmt(s): return ' + '.join(sorted([
        str(i).replace('RiskCategory_','').replace('_',' ') for i in s]))
    risk_rules['ant_str'] = risk_rules['antecedents'].apply(fmt)
    risk_rules['con_str'] = risk_rules['consequents'].apply(fmt)
    return risk_rules


def encode_patient(patient_dict, ordinal_maps, le_classes, feature_cols):
    """Encode a single patient dict into a feature vector."""
    row = patient_dict.copy()
    for col, mapping in ordinal_maps.items():
        if col in row:
            row[col] = mapping.get(row[col], 0)
    # Existing condition label encoding
    cond = row.get('ExistingCondition', 'None')
    if cond in le_classes:
        row['ExistingCondition'] = list(le_classes).index(cond)
    else:
        row['ExistingCondition'] = 0
    return np.array([[row[f] for f in feature_cols]], dtype=float)


def get_matching_rules(patient_dict, risk_rules, df, top_n=5):
    """Find association rules whose antecedents match this patient's profile."""
    bmi   = patient_dict['BMI']
    age   = patient_dict['Age']
    sbp   = patient_dict['SystolicBP']
    dbp   = patient_dict['DiastolicBP']
    chol  = patient_dict['Cholesterol']
    sugar = patient_dict['BloodSugar']

    age_g  = 'Young' if age<=40 else 'MiddleAge' if age<=55 else 'Senior' if age<=70 else 'Elderly'
    bmi_c  = 'Underweight' if bmi<18.5 else 'Normal' if bmi<25 else 'Overweight' if bmi<30 else 'Obese'
    bp_c   = 'High_BP' if (sbp>140 or dbp>90) else 'Normal_BP'
    chol_c = 'Desirable' if chol<200 else 'Borderline' if chol<240 else 'High'
    sug_c  = 'Normal' if sugar<100 else 'Prediabetic' if sugar<126 else 'Diabetic'

    patient_dummy_items = {
        f'AgeGroup_{age_g}', f'BMI_Category_{bmi_c}', f'BP_Category_{bp_c}',
        f'Chol_Category_{chol_c}', f'Sugar_Category_{sug_c}',
        f'SmokingStatus_{patient_dict.get("SmokingStatus","")}',
        f'AlcoholConsumption_{patient_dict.get("AlcoholConsumption","")}',
        f'ExerciseFrequency_{patient_dict.get("ExerciseFrequency","")}',
        f'DietQuality_{patient_dict.get("DietQuality","")}',
        f'SleepQuality_{patient_dict.get("SleepQuality","")}',
        f'StressLevel_{patient_dict.get("StressLevel","")}',
        f'ExistingCondition_{patient_dict.get("ExistingCondition","None")}',
        f'FamilyHistory_{patient_dict.get("FamilyHistory","")}',
    }

    matches = []
    for _, row in risk_rules.iterrows():
        ant_items = set(row['antecedents'])
        overlap = len(ant_items & patient_dummy_items)
        if overlap >= 1:
            matches.append({**row, 'overlap': overlap})

    if not matches:
        return risk_rules.head(top_n)
    return pd.DataFrame(matches).sort_values(['overlap','lift'], ascending=False).head(top_n)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & TRAIN
# ══════════════════════════════════════════════════════════════════════════════

try:
    df = load_and_prepare()
    rf, km, iso, scaler, le, ORDINAL_MAPS, FEATURE_COLS, df_enc, X_scaled, X_pca, df = train_models(df)
    risk_rules = mine_rules(df)
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='background: linear-gradient(135deg, #2c3e50, #3498db);
     padding: 28px 32px; border-radius: 14px; margin-bottom: 24px;'>
    <h1 style='color:white; margin:0; font-size:2rem;'>🏥 Health Risk Assessment System</h1>
    
</div>
""", unsafe_allow_html=True)

if not data_loaded:
    st.error("⚠️ Could not load `data/health_risk_assessment.csv`. "
             "Please make sure the file is in the `data/` folder.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Patient Assessment",
    "📊 Dashboard",
    "🗄️ Dataset",
    "ℹ️ About"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PATIENT ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Enter Patient Details")
    st.markdown("Fill in the patient information below and click **Assess Risk** to get the prediction.")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            age    = st.slider("Age", 18, 85, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi    = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)
            family = st.selectbox("Family History of Disease", ["No", "Yes"])

        with col2:
            st.markdown("**🚬 Lifestyle**")
            smoking  = st.selectbox("Smoking Status",
                ["Never", "Former", "Current - Light", "Current - Heavy"])
            alcohol  = st.selectbox("Alcohol Consumption",
                ["Light", "Moderate", "Heavy"])
            exercise = st.selectbox("Exercise Frequency",
                ["Rare", "Moderate", "Regular", "Intensive"])
            diet     = st.selectbox("Diet Quality",
                ["Poor", "Fair", "Good", "Excellent"])
            sleep    = st.selectbox("Sleep Quality",
                ["Poor", "Fair", "Good", "Excellent"])
            stress   = st.selectbox("Stress Level",
                ["Low", "Moderate", "High", "Very High"])

        with col3:
            st.markdown("**🩺 Clinical Measurements**")
            systolic    = st.slider("Systolic BP (mmHg)", 90, 180, 120)
            diastolic   = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
            cholesterol = st.slider("Cholesterol (mg/dL)", 150, 300, 200)
            blood_sugar = st.slider("Blood Sugar (mg/dL)", 70, 200, 90)
            condition   = st.selectbox("Existing Condition",
                ["None", "Hypertension", "Diabetes", "Heart Disease", "Arthritis", "Asthma"])

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if submitted:
        patient = {
            'Age': age, 'Gender': gender, 'BMI': bmi,
            'SmokingStatus': smoking, 'AlcoholConsumption': alcohol,
            'ExerciseFrequency': exercise, 'DietQuality': diet,
            'SleepQuality': sleep, 'StressLevel': stress,
            'SystolicBP': systolic, 'DiastolicBP': diastolic,
            'Cholesterol': cholesterol, 'BloodSugar': blood_sugar,
            'ExistingCondition': condition, 'FamilyHistory': family,
        }

        # Encode & predict
        X_pat = encode_patient(patient, ORDINAL_MAPS, le.classes_, FEATURE_COLS)
        X_pat_scaled = scaler.transform(X_pat)

        risk_num   = rf.predict(X_pat_scaled)[0]
        risk_proba = rf.predict_proba(X_pat_scaled)[0]
        risk_label = ORDER[risk_num]
        cluster    = km.predict(X_pat_scaled)[0]
        iso_score  = iso.decision_function(X_pat_scaled)[0]
        is_anomaly = iso.predict(X_pat_scaled)[0] == -1

        st.markdown("---")
        st.markdown("## 📋 Assessment Results")

        # ── Risk result box ───────────────────────────────────────────────────
        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            color = RISK_COLORS[risk_label]
            bg    = RISK_BG[risk_label]
            icons = {'Low':'✅','Moderate':'⚠️','High':'🔶','Very High':'🚨'}
            msgs  = {
                'Low':       'This patient shows a low health risk profile. Continue healthy habits.',
                'Moderate':  'Moderate risk detected. Lifestyle improvements recommended.',
                'High':      'High risk identified. Medical consultation advised.',
                'Very High': 'Very high risk. Immediate medical attention recommended.',
            }
            st.markdown(f"""
            <div class="risk-box" style="background:{bg}; border-color:{color};">
                <div style="font-size:3rem;">{icons[risk_label]}</div>
                <div style="font-size:2rem; font-weight:800; color:{color};">{risk_label} Risk</div>
                <div style="color:#555; margin-top:8px; font-size:0.95rem;">{msgs[risk_label]}</div>
            </div>
            """, unsafe_allow_html=True)

            if is_anomaly:
                st.warning("⚡ **Anomalous Profile Detected** — This patient's combination of "
                           "features is unusual compared to the general population. "
                           "Extra clinical scrutiny recommended.")

        with col_gauge:
            # Probability bars
            st.markdown("**Prediction Confidence**")
            for i, (cat, prob) in enumerate(zip(ORDER, risk_proba)):
                c = RISK_COLORS[cat]
                st.markdown(f"""
                <div style="margin:6px 0;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color:{c}; font-weight:600;">{cat}</span>
                        <span>{prob*100:.1f}%</span>
                    </div>
                    <div style="background:#eee; border-radius:6px; height:12px;">
                        <div style="background:{c}; width:{prob*100:.1f}%; height:12px;
                             border-radius:6px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Cluster & Key metrics ─────────────────────────────────────────────
        col_cl, col_m1, col_m2, col_m3, col_m4 = st.columns(5)

        cluster_names = {0:'Moderate Risk Group', 1:'Moderate Risk Group',
                         2:'Critical Risk Group', 3:'Healthy Group'}
        cluster_desc  = {
            0: 'Middle-aged patients with moderate lifestyle and clinical metrics.',
            1: 'Middle-aged patients with slightly elevated metrics.',
            2: 'Older patients with elevated BP, BMI, and cholesterol.',
            3: 'Younger patients with healthy lifestyle and normal metrics.',
        }

        with col_cl:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem;">🔵</div>
                <div style="font-weight:700; font-size:1rem;">Cluster {cluster}</div>
                <div style="font-size:0.8rem; color:#555;">{cluster_names[cluster]}</div>
            </div>
            """, unsafe_allow_html=True)

        bp_status = "⚠️ Elevated" if (systolic > 140 or diastolic > 90) else "✅ Normal"
        ch_status = "⚠️ High" if cholesterol >= 240 else "🟡 Borderline" if cholesterol >= 200 else "✅ OK"
        sg_status = "⚠️ Diabetic" if blood_sugar >= 126 else "🟡 Pre-diabetic" if blood_sugar >= 100 else "✅ Normal"
        bm_status = "⚠️ Obese" if bmi >= 30 else "🟡 Overweight" if bmi >= 25 else "✅ Normal"

        for col, label, val, status in [
            (col_m1, "Blood Pressure", f"{systolic}/{diastolic}", bp_status),
            (col_m2, "Cholesterol",    f"{cholesterol} mg/dL",   ch_status),
            (col_m3, "Blood Sugar",    f"{blood_sugar} mg/dL",   sg_status),
            (col_m4, "BMI",            f"{bmi:.1f}",             bm_status),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.75rem; color:#888;">{label}</div>
                    <div style="font-weight:700; font-size:1.1rem;">{val}</div>
                    <div style="font-size:0.8rem;">{status}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Cluster info ──────────────────────────────────────────────────────
        st.info(f"**Cluster {cluster} — {cluster_names[cluster]}:** {cluster_desc[cluster]}")

        # ── Matching association rules ─────────────────────────────────────────
        st.markdown("### 🔗 Matching Risk Patterns (Association Rules)")
        st.markdown("Rules from the dataset that match this patient's profile:")

        matched = get_matching_rules(patient, risk_rules, df)
        if len(matched) > 0:
            for i, (_, row) in enumerate(matched.iterrows()):
                conf_color = '#e74c3c' if row['confidence'] > 0.80 else \
                             '#e67e22' if row['confidence'] > 0.70 else '#f39c12'
                st.markdown(f"""
                <div style="background:#f8f9fa; border-left:4px solid {conf_color};
                     padding:12px 16px; border-radius:0 8px 8px 0; margin:6px 0;">
                    <span style="font-weight:600;">IF</span> {row['ant_str']}
                    &nbsp;→&nbsp;
                    <span style="font-weight:600; color:{conf_color};">{row['con_str']}</span>
                    &nbsp;&nbsp;
                    <span style="font-size:0.8rem; color:#888;">
                    [Support={row['support']:.2f} | Confidence={row['confidence']:.0%} | Lift={row['lift']:.2f}]
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strong matching rules found for this specific profile.")

        # ── Recommendations ───────────────────────────────────────────────────
        st.markdown("### 💡 Personalised Recommendations")
        recs = []
        if smoking in ['Current - Light', 'Current - Heavy']:
            recs.append("🚭 **Smoking Cessation** — Smoking is the #2 risk predictor. "
                        "Cessation programmes reduce CVD risk by up to 50%.")
        if bmi >= 25:
            recs.append(f"⚖️ **Weight Management** — Your BMI ({bmi:.1f}) is above normal. "
                        "Target BMI 18.5–24.9 through diet and exercise.")
        if exercise in ['Rare']:
            recs.append("🏃 **Increase Exercise** — Regular physical activity reduces mortality risk by 30%. "
                        "Aim for 150 mins/week of moderate activity.")
        if systolic > 130 or diastolic > 85:
            recs.append(f"💊 **Monitor Blood Pressure** — Your BP ({systolic}/{diastolic} mmHg) "
                        "is elevated. Regular monitoring and low-sodium diet recommended.")
        if cholesterol >= 200:
            recs.append(f"🥗 **Cholesterol Management** — Cholesterol {cholesterol} mg/dL. "
                        "Reduce saturated fats and increase fibre intake.")
        if blood_sugar >= 100:
            recs.append(f"🍬 **Blood Sugar Control** — Blood sugar {blood_sugar} mg/dL. "
                        "Reduce refined carbohydrates and monitor regularly.")
        if stress in ['High', 'Very High']:
            recs.append("🧘 **Stress Reduction** — Chronic stress doubles heart disease risk. "
                        "Consider mindfulness, exercise, or counselling.")
        if sleep == 'Poor':
            recs.append("😴 **Improve Sleep** — Poor sleep increases obesity and diabetes risk. "
                        "Aim for 7–9 hours of quality sleep per night.")

        if not recs:
            recs.append("✅ **Keep it up!** Your lifestyle indicators look good. "
                        "Continue regular health check-ups and maintain current habits.")

        for rec in recs:
            st.markdown(f"- {rec}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Dataset Overview & Mining Results")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, val in [
        (k1, "Total Patients", f"{len(df):,}"),
        (k2, "Features",       "15"),
        (k3, "High Risk",      f"{(df['RiskCategory'].isin(['High','Very High'])).sum():,}"),
        (k4, "Clusters Found", "4"),
        (k5, "Avg Age",        f"{df['Age'].mean():.0f} yrs"),
    ]:
        with col:
            st.metric(label, val)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Risk distribution
    with col_a:
        st.markdown("**Risk Category Distribution**")
        fig, ax = plt.subplots(figsize=(6,4))
        counts = df['RiskCategory'].value_counts().reindex(ORDER)
        bars = ax.bar(ORDER, counts, color=[RISK_COLORS[r] for r in ORDER],
                      edgecolor='white', linewidth=1.5, width=0.6)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count'); ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Age by risk
    with col_b:
        st.markdown("**Age Distribution by Risk Category**")
        fig, ax = plt.subplots(figsize=(6,4))
        data = [df[df['RiskCategory']==r]['Age'].values for r in ORDER]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(linewidth=1.3),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))
        for patch, r in zip(bp['boxes'], ORDER):
            patch.set_facecolor(RISK_COLORS[r]); patch.set_alpha(0.85)
        ax.set_xticklabels(ORDER, fontsize=9)
        ax.set_ylabel('Age'); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col_c, col_d = st.columns(2)

    # Feature importance
    with col_c:
        st.markdown("**Feature Importance (Random Forest)**")
        READABLE = {
            'Age':'Age','BMI':'BMI','SmokingStatus':'Smoking',
            'AlcoholConsumption':'Alcohol','ExerciseFrequency':'Exercise',
            'DietQuality':'Diet','SleepQuality':'Sleep','StressLevel':'Stress',
            'SystolicBP':'Systolic BP','DiastolicBP':'Diastolic BP',
            'Cholesterol':'Cholesterol','BloodSugar':'Blood Sugar',
            'ExistingCondition':'Existing Cond.','FamilyHistory':'Family History','Gender':'Gender'
        }
        feat_df = pd.DataFrame({
            'label': [READABLE[f] for f in FEATURE_COLS],
            'imp':   rf.feature_importances_
        }).sort_values('imp', ascending=True)
        colors = ['#e74c3c' if v>0.10 else '#e67e22' if v>0.07 else '#3498db'
                  for v in feat_df['imp']]
        fig, ax = plt.subplots(figsize=(6,5))
        ax.barh(feat_df['label'], feat_df['imp']*100, color=colors,
                edgecolor='white', height=0.65)
        ax.set_xlabel('Importance (%)'); ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Cluster PCA
    with col_d:
        st.markdown("**K-Means Clusters (PCA Projection)**")
        fig, ax = plt.subplots(figsize=(6,5))
        for c in range(4):
            mask = km.labels_ == c
            ax.scatter(X_pca[mask,0], X_pca[mask,1],
                       c=CLUSTER_COLORS[c], label=f'C{c}',
                       alpha=0.5, s=20, edgecolors='none')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(title='Cluster', fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Correlation heatmap
    st.markdown("**Correlation Heatmap — Clinical Measurements**")
    fig, ax = plt.subplots(figsize=(9,5))
    num_cols = ['Age','BMI','SystolicBP','DiastolicBP','Cholesterol','BloodSugar','RiskScore']
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
                center=0, ax=ax, linewidths=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Top association rules table
    st.markdown("**Top 10 Risk Association Rules**")
    display_rules = risk_rules.head(10)[['ant_str','con_str','support','confidence','lift']].copy()
    display_rules.columns = ['If (Antecedent)','Then (Consequent)','Support','Confidence','Lift']
    display_rules['Support']    = display_rules['Support'].map('{:.2f}'.format)
    display_rules['Confidence'] = display_rules['Confidence'].map('{:.0%}'.format)
    display_rules['Lift']       = display_rules['Lift'].map('{:.2f}'.format)
    st.dataframe(display_rules, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DATASET
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🗄️ Patient Dataset")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.multiselect("Filter by Risk Category", ORDER, default=ORDER)
    with col_f2:
        gender_filter = st.multiselect("Filter by Gender", ["Male","Female"], default=["Male","Female"])
    with col_f3:
        age_range = st.slider("Age Range", 18, 85, (18, 85))

    filtered = df[
        (df['RiskCategory'].isin(risk_filter)) &
        (df['Gender'].isin(gender_filter)) &
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1])
    ]

    st.markdown(f"Showing **{len(filtered):,}** of {len(df):,} patients")

    show_cols = ['PatientID','Age','Gender','BMI','SmokingStatus','ExerciseFrequency',
                 'SystolicBP','Cholesterol','BloodSugar','ExistingCondition',
                 'FamilyHistory','RiskScore','RiskCategory']
    st.dataframe(filtered[show_cols].reset_index(drop=True),
                 use_container_width=True, height=450)

    # Summary stats
    st.markdown("**Summary Statistics**")
    st.dataframe(filtered[['Age','BMI','SystolicBP','Cholesterol','BloodSugar','RiskScore']
                          ].describe().round(2), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────