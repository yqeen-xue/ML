import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Load data (clinical2 without RNA)
def load_clinical(filepath):
    df = pd.read_csv(filepath)
    df = df[df["time"] >= 0]
    df = df.dropna(subset=["status"])

    y = df["status"].astype(int)
    X = df.drop(columns=["PatientID", "status", "time"])

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X, y

# Load image features
def load_image_features(image_feat_path):
    image_feat = pd.read_csv(image_feat_path)
    image_feat.reset_index(inplace=True)
    image_feat.rename(columns={"index": "PatientID"}, inplace=True)
    image_feat["PatientID"] = image_feat["PatientID"].astype(str).str.extract(r"(R01-\d+|AMC-\d+)")

    return image_feat

def main():
    os.makedirs("results", exist_ok=True)
    clinical_path = "data/clinical2_norna.csv"
    image_feat_path = "data/features_segm_tumour.csv"

    # Load data
    clinical = pd.read_csv(clinical_path)
    image = load_image_features(image_feat_path)

    # Merge clinical with image features
    merged = pd.merge(clinical, image, on="PatientID")
    print("Merged shape:", merged.shape)

    y = merged["status"].astype(int)
    X = merged.drop(columns=["PatientID", "status", "time"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline
    clf = RandomForestClassifier(random_state=42)
    pipe = Pipeline([("clf", clf)])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__min_samples_leaf": [3, 5]
    }

    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)

    y_pred_proba = gs.predict_proba(X_test)[:, 1]
    y_pred = gs.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Best Params:", gs.best_params_)
    print(f"AUC: {auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

    pd.DataFrame([{
        "model": "ModelC_Clinical2_Image",
        "auc": round(auc, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        **gs.best_params_
    }]).to_csv("results/modelC_clinical2_image_results.csv", index=False)

if __name__ == "__main__":
    main()
