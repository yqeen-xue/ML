import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Load data
def load_data(clinical_rna_path, image_path):
    # Load and clean RNA + clinical
    df_rna = pd.read_csv(clinical_rna_path, low_memory=False)
    df_rna = df_rna[df_rna["status"].isin([0, 1])]

    # Load image features
    df_img = pd.read_csv(image_path)

    # Merge
    df = pd.merge(df_rna, df_img, left_on="PatientID", right_on="patient_id", how="inner")

    # Drop irrelevant columns
    df.drop(columns=["PatientID", "patient_id"], inplace=True)

    # Split target and features
    y = df["status"].astype(int)
    X = df.drop(columns=["status"])

    return X, y


def main():
    os.makedirs("results", exist_ok=True)

    # File paths
    clinical_rna_path = "data/clinical2_rna_merged.csv"
    image_path = "outputs/image_features_dataset2.csv"

    # Load merged data
    X, y = load_data(clinical_rna_path, image_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train class distribution:", np.bincount(y_train))

    # Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=50, random_state=42)),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    # Grid
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [5, 7],
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

    print(f"[\u2713] Optimized Model D Results — AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

    # Save result
    pd.DataFrame([{
        "model": "ModelD_Clinical2RNAImage_RF_PCA",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        **gs.best_params_
    }]).to_csv("results/modelD_clinical2rnaimage_pca_results.csv", index=False)

if __name__ == "__main__":
    main()
