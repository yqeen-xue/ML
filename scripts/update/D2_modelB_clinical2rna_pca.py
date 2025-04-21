
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------
# Load and prepare data (RNA + clinical)
# -------------------------
def load_data(filepath, pca_components=50):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.dropna(subset=["status"])

    y = df["status"].astype(int)
    X = df.drop(columns=["PatientID", "status"])
    X = X.select_dtypes(include=[np.number]).dropna(axis=1, how="any")

    # Apply PCA to RNA features
    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, y

# -------------------------
# Manual stratified split
# -------------------------
def manual_balanced_split(X, y, test_frac=0.1):
    X0 = X[y == 0]
    y0 = y[y == 0]
    X1 = X[y == 1]
    y1 = y[y == 1]

    X0, y0 = X0[np.random.permutation(len(X0))], y0.sample(frac=1, random_state=42)
    X1, y1 = X1[np.random.permutation(len(X1))], y1.sample(frac=1, random_state=42)

    n0 = int(len(X0) * (1 - test_frac))
    n1 = int(len(X1) * (1 - test_frac))

    X_train = np.vstack([X0[:n0], X1[:n1]])
    y_train = pd.concat([y0[:n0], y1[:n1]])
    X_test = np.vstack([X0[n0:], X1[n1:]])
    y_test = pd.concat([y0[n0:], y1[n1:]])

    return X_train, X_test, y_train, y_test

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"
    X, y = load_data(filepath, pca_components=50)
    X_train, X_test, y_train, y_test = manual_balanced_split(X, y, test_frac=0.1)

    print("Train class distribution:", np.bincount(y_train))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    pipe.fit(X_train, y_train)

    if len(pipe.classes_) < 2:
        print("[!] Model trained with only one class. Skipping evaluation.")
        return

    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"[✓] Optimized Model B Results — AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

    pd.DataFrame([{
        "model": "ModelB_v2_Clinical2RNA_PCA_LogReg",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "pca_components": 50,
        "classifier": "LogisticRegression"
    }]).to_csv("results/modelB_v2_clinical2rna_pca_results.csv", index=False)

if __name__ == "__main__":
    main()
