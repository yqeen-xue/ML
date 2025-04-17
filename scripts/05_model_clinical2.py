import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest # type: ignore
from sksurv.metrics import concordance_index_censored # type: ignore
import os

# Ensure output folder exists
os.makedirs("results", exist_ok=True)

def load_data(filepath, clinical_only=False):
    df = pd.read_csv(filepath)

    # Ensure PatientID is not in the feature set
    if "PatientID" in df.columns:
        df = df.drop(columns=["PatientID"])

    # Outcome definition
    y = df[["status", "time"]]
    y["status"] = y["status"].astype(bool)
    y_structured = np.array(list(y.itertuples(index=False)), dtype=[("event", bool), ("time", float)])

    # Feature selection
    if clinical_only:
        X = df.loc[:, ["age", "gender", "smoking", "histology", "EGFR", "KRAS", "ALK"]]  # select only clinical
    else:
        X = df.drop(columns=["status", "time"])

    return X, y_structured

def train_and_evaluate(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=10,
                                  max_features="sqrt", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    c_index = concordance_index_censored(y_test["event"], y_test["time"], model.predict(X_test))[0]
    print(f"{model_name} C-index: {c_index:.4f}")

    return {"model": model_name, "c_index": round(c_index, 4)}

def main():
    results = []

    # Model 1: Clinical only
    X1, y1 = load_data("data/clinical2_with_rna.csv", clinical_only=True)
    results.append(train_and_evaluate(X1, y1, "clinical_only"))

    # Model 2: Clinical + RNA
    X2, y2 = load_data("data/clinical2_rna_merged.csv", clinical_only=False)
    results.append(train_and_evaluate(X2, y2, "clinical_rna"))

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/clinical_model_results.csv", index=False)
    print("Results saved to: results/clinical_model_results.csv")

if __name__ == "__main__":
    main()
