import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Load clinical + RNA data (already merged and cleaned)
df_rna = pd.read_csv("data/clinical2_rna.csv")

# Load extracted image features for the same patients
features_segm = pd.read_csv("data/features_segm_tumour.csv")
features_gray = pd.read_csv("data/features_greyscale_tumour.csv")
image_features = pd.concat([features_segm, features_gray], axis=1)

# Align samples present in both datasets
common_ids = set(df_rna["PatientID"]).intersection(set(image_features.index))
df_rna = df_rna[df_rna["PatientID"].isin(common_ids)].copy()
image_features = image_features.loc[df_rna["PatientID"]].reset_index(drop=True)

# Prepare labels and feature matrix
y = df_rna["status"].astype(int)
X_clinical = df_rna.drop(columns=["PatientID", "status", "time"])
X = pd.concat([X_clinical.reset_index(drop=True), image_features], axis=1)

# Define classifier and pipeline
clf = RandomForestClassifier(random_state=42)
pipe = Pipeline([("clf", clf)])

# Parameter grid
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, 7],
    "clf__min_samples_leaf": [3, 5]
}

# Grid search
model = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
model.fit(X, y)

# Predict and evaluate on training set
pred_proba = model.predict_proba(X)[:, 1]
pred = model.predict(X)
auc = roc_auc_score(y, pred_proba)
f1 = f1_score(y, pred)
acc = accuracy_score(y, pred)

print("Best Params:", model.best_params_)
print(f"AUC: {auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

# Save results
os.makedirs("results", exist_ok=True)
pd.DataFrame([{
    "model": "ModelD_Clinical2RNA_Image_RF",
    "auc": round(auc, 4),
    "f1_score": round(f1, 4),
    "accuracy": round(acc, 4),
    **model.best_params_
}]).to_csv("results/modelD_clinical2rna_image_results.csv", index=False)
