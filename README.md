

## Mining Techniques
| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | EDA | Understand distributions and relationships |
| 2 | K-Means Clustering | Discover natural patient groupings (unsupervised) |
| 3 | Apriori Association Rules | Find co-occurring risk factor combinations |
| 4 | Isolation Forest + LOF | Detect anomalous patient profiles |
| 5 | Random Forest Classification | Predict risk category, rank feature importance |

## Dataset
- **1,000** synthetic patient health assessments
- **15 features**: demographic, lifestyle, and clinical measurements
- **Target**: RiskCategory (Low / Moderate / High / Very High)
- Distribution: Low 28.7% | Moderate 39.8% | High 24.1% | Very High 7.4%

## Project Structure
```
health_risk_mining/
├── main.py               # Pipeline entry point
├── requirements.txt
├── data/
│   └── health_data.csv   # Generated dataset
├── outputs/              # All figures and CSVs (auto-generated)
└── src/
    ├── data_generator.py  # Synthetic dataset generation
    ├── eda.py             # Exploratory data analysis
    ├── clustering.py      # K-Means clustering
    ├── association_rules.py  # Apriori rule mining
    ├── anomaly_detection.py  # Isolation Forest + LOF
    ├── classification.py  # RF / GB / DT + feature importance
    └── utils.py           # Shared helpers
```

## Setup & Run
```bash
pip install -r requirements.txt
python main.py
```

## Key Results
### Clustering (K-Means, k=4)
- Cluster 3: Young, low BMI, normal BP → 86% Low risk ("Healthy Young")
- Cluster 0/1: Middle-aged, moderate metrics → predominantly Moderate risk
- Cluster 2: Older, high BMI, elevated BP → 66% High + 33% Very High ("Critical")

### Association Rules (Apriori)
- Young + Normal BP + Never Smoked → **Low Risk** [conf=0.91, lift=2.96]
- Obese + Heavy Smoker + No Exercise → **Very High Risk** [conf≥0.85]
- 172 risk-targeting rules discovered (lift≥1.5, confidence≥0.60)

### Anomaly Detection
- 50 anomalies each by Isolation Forest and LOF (5% contamination rate)
- 11 consensus anomalies flagged by both methods
- Anomalies show elevated Age (57.2), BMI (31.9), and BP (142.3)

### Classification (Random Forest)
- Test accuracy: **88.0%** | CV: 85.4% ±1.3%
- Top predictors: Blood Sugar (15.5%), BMI (12.4%), Systolic BP (12.3%)
