import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Load clinical2 dataset with positive and negative classes
def load_data(filepath):
    df = pd.read_csv(filepath)

    # Binary classification target
    y = df["status"].astype(int)

    # Safely drop non-feature columns (e.g., ID, label, survival time)
    drop_cols = [col for col in ["PatientID", "status", "time"] if col in df.columns]
    X = df.drop(columns=drop_cols)

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    # Debug print
    print(f"[i] Feature shape: {X.shape}, Label balance: {np.bincount(y)}")

    return X, y

def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_with_rna.csv"

    # Load dataset
    X, y = load_data(filepath)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define classifier pipeline with scaler
    clf = RandomForestClassifier(random_state=42, class_weight="balanced")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    # Parameter grid for tuning
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__min_samples_leaf": [3, 5]
    }

    # Grid search
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)

    # Evaluation
    y_pred_proba = gs.predict_proba(X_test)[:, 1]
    y_pred = gs.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Best Params:", gs.best_params_)
    print(f"[✓] Model A (balanced) Results — AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

    # Save result
    pd.DataFrame([{
        "model": "ModelA_Clinical2_RF_Balanced",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        **gs.best_params_
    }]).to_csv("results/modelA_clinical2_balanced_results.csv", index=False)

if __name__ == "__main__":
    main()
