# scripts/D1_modela_clinical1.py (patched for clinical1 columns)

import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# -------------------------
# Load cleaned clinical1.csv
# -------------------------
def load_data(filepath):
    df = pd.read_csv(filepath)

    # Patch: convert "deadstatus.event" to status if needed
    if "status" not in df.columns and "deadstatus.event" in df.columns:
        df["status"] = df["deadstatus.event"]

    df = df.dropna(subset=["status"])

    y = df["status"].astype(int)
    X = df.drop(columns=["PatientID", "status"])
    X = X.select_dtypes(include=[np.number]).dropna(axis=1, how="any")

    return X, y

# -------------------------
# Stratified split
# -------------------------
def stratified_split(X, y, test_frac=0.1):
    return train_test_split(X, y, test_size=test_frac, random_state=42, stratify=y)

# -------------------------
# Main script
# -------------------------
def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/cleaned_clinical1.csv"

    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    print("Train class distribution:", np.bincount(y_train))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    pipe.fit(X_train, y_train)

    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"[✓] Model A (D1) Results — AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

    pd.DataFrame([{
        "model": "ModelA_D1_Clinical1_LogReg",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "classifier": "LogisticRegression"
    }]).to_csv("results/modelA_clinical1_results.csv", index=False)

if __name__ == "__main__":
    main()