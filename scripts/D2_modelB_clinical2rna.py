import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load dataset with clinical + RNA features
# -------------------------
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df["time"] >= 0]
    df = df.dropna(subset=["status"])

    y = df["status"].astype(int)
    X = df.drop(columns=["PatientID", "status", "time"])
    X = X.select_dtypes(include=[np.number]).dropna(axis=1, how="any")

    return X, y

# -------------------------
# Manual stratified split to ensure both classes exist
# -------------------------
def manual_balanced_split(X, y, test_frac=0.1):
    X0 = X[y == 0]
    y0 = y[y == 0]
    X1 = X[y == 1]
    y1 = y[y == 1]

    X0, y0 = X0.sample(frac=1, random_state=42), y0.sample(frac=1, random_state=42)
    X1, y1 = X1.sample(frac=1, random_state=42), y1.sample(frac=1, random_state=42)

    n0 = int(len(X0) * (1 - test_frac))
    n1 = int(len(X1) * (1 - test_frac))

    X_train = pd.concat([X0[:n0], X1[:n1]])
    y_train = pd.concat([y0[:n0], y1[:n1]])
    X_test = pd.concat([X0[n0:], X1[n1:]])
    y_test = pd.concat([y0[n0:], y1[n1:]])

    return X_train, X_test, y_train, y_test

# -------------------------
# Main script
# -------------------------
def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"

    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = manual_balanced_split(X, y, test_frac=0.1)

    print("Train class distribution:", np.bincount(y_train))

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    pipe.fit(X_train, y_train)

    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"AUC: {auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

    pd.DataFrame([{
        "model": "ModelB_Clinical2RNA_RF",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 5
    }]).to_csv("results/modelB_clinical2rna_results.csv", index=False)

if __name__ == "__main__":
    main()
