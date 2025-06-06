import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# -------------------------
# Load clinical1 and image features
# -------------------------
def load_data(clinical_path, image_path):
    df_clin = pd.read_csv(clinical_path)
    df_img = pd.read_csv(image_path)

    # Map status if not present
    if "status" not in df_clin and "deadstatus.event" in df_clin.columns:
        df_clin["status"] = df_clin["deadstatus.event"]

    # Merge
    df = pd.merge(df_clin, df_img, left_on="PatientID", right_on="patient_id", how="inner")

    # Drop non-numeric / ID columns
    y = df["status"].astype(int)
    X = df.drop(columns=["PatientID", "patient_id", "status"])
    X = X.drop(columns=["deadstatus.event", "Survival.time"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).dropna(axis=1, how="any")

    return X, y

# -------------------------
# Stratified split
# -------------------------
def stratified_split(X, y, test_frac=0.1):
    return train_test_split(X, y, test_size=test_frac, random_state=42, stratify=y)

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs("results", exist_ok=True)

    clinical_path = "data/cleaned_clinical1.csv"
    image_path = "outputs/image_features_dataset1.csv"

    X, y = load_data(clinical_path, image_path)
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

    print(f"[✓] Model B (D1) Results — AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

    pd.DataFrame([{
        "model": "ModelB_D1_Clinical1Image_LogReg",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "classifier": "LogisticRegression"
    }]).to_csv("results/modelB_clinical1image_results.csv", index=False)

if __name__ == "__main__":
    main()