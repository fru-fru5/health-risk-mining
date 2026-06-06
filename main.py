
import warnings
warnings.filterwarnings('ignore')

from src.data_generator import generate_dataset
from src.eda import run_eda
from src.clustering import run_clustering
from src.association_rules import run_association_mining
from src.anomaly_detection import run_anomaly_detection
from src.classification import run_classification
from src.utils import print_section

if __name__ == "__main__":
    print_section("HEALTH RISK DATA MINING SYSTEM")

    # Step 1 – Generate & explore data
    print_section("STEP 1: DATA GENERATION & EDA")
    df = generate_dataset(n=1000, random_state=42)
    run_eda(df)

    # Step 2 – Clustering
    print_section("STEP 2: K-MEANS CLUSTERING")
    df, df_enc, X_scaled, pca_coords = run_clustering(df)

    # Step 3 – Association rules
    print_section("STEP 3: ASSOCIATION RULE MINING")
    run_association_mining(df)

    # Step 4 – Anomaly detection
    print_section("STEP 4: ANOMALY DETECTION")
    run_anomaly_detection(df_enc, X_scaled, pca_coords, df)

    # Step 5 – Classification
    print_section("STEP 5: CLASSIFICATION")
    run_classification(df_enc)

    print_section("PIPELINE COMPLETE")
    print("All outputs saved to: outputs/")
