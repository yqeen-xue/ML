# scripts/D2_modelB_clinical2rna.py

import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
# Main script
# -------------------------
def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"

    X, y = load_data(filepath)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Train class distribution:", np.bincount(y_train))

    if len(np.unique(y_train)) < 2:
        print("[!] Training set only contains one class. Skipping model.")
        return

    # Pipeline with scaler and classifier
    clf = RandomForestClassifier(random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__min_samples_leaf": [3, 5]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        error_score=np.nan
    )

    gs.fit(X_train, y_train)

    if len(gs.classes_) < 2:
        print("[!] Trained model only predicts one class. Skipping evaluation.")
        return

    y_pred_proba = gs.predict_proba(X_test)[:, 1]
    y_pred = gs.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Best Params:", gs.best_params_)
    print(f"AUC: {auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

    pd.DataFrame([{
        "model": "ModelB_Clinical2RNA_RF",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        **gs.best_params_
    }]).to_csv("results/modelB_clinical2rna_results.csv", index=False)

if __name__ == "__main__":
    main()