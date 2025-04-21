import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------
# Load and prepare data
# -------------------------
def load_data(filepath, pca_components=50):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.dropna(subset=["status"])
    y = df["status"].astype(int)

    # Remove identifier
    df = df.drop(columns=["PatientID", "status"], errors="ignore")

    # Only keep numeric columns
    df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="any")

    print(f"[i] All features after cleaning: {df.shape}")

    # Define known clinical columns — adjust as needed
    clinical_cols = [col for col in df.columns if col.lower() in [
        "age", "egfr", "kras", "alk", "survival.time", "time"
    ]]
    rna_cols = [col for col in df.columns if col not in clinical_cols]

    # Split
    X_clin = df[clinical_cols] if clinical_cols else pd.DataFrame(index=df.index)
    X_rna = df[rna_cols]

    print(f"[i] RNA shape before PCA: {X_rna.shape}, Clinical shape: {X_clin.shape}")

    # PCA on RNA only
    pca = PCA(n_components=min(pca_components, X_rna.shape[1]), random_state=42)
    X_rna_pca = pca.fit_transform(X_rna)

    # Combine
    if not X_clin.empty:
        X_all = np.hstack([X_clin.values, X_rna_pca])
    else:
        X_all = X_rna_pca

    print(f"[i] Final feature shape after PCA fusion: {X_all.shape}")

    return X_all, y

# -------------------------
# Manual stratified split
# -------------------------
def manual_balanced_split(X, y, test_frac=0.1):
    X0 = X[y == 0]
    y0 = y[y == 0]
    X1 = X[y == 1]
    y1 = y[y == 1]

    # Shuffle
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

    print(f"[✓] Fixed PCA Model B Results — AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

    pd.DataFrame([{
        "model": "ModelB_v3_Clinical2RNA_PCA_Fixed",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "pca_components": 50,
        "classifier": "LogisticRegression"
    }]).to_csv("results/modelB_v3_clinical2rna_pca_fixed.csv", index=False)

if __name__ == "__main__":
    main()
