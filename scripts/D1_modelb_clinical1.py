import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, f1_score

# -------------------------
# Load clinical and image features
# -------------------------
clinical_df = pd.read_csv("data/clinical1.csv")
image_df = pd.read_csv("outputs/image_features_dataset1.csv")

# -------------------------
# Merge features on patient_id
# -------------------------
merged_df = clinical_df.merge(image_df, how="inner", left_on="PatientID", right_on="patient_id")

# -------------------------
# Prepare features and labels
# -------------------------
X = merged_df.drop(columns=["PatientID", "patient_id", "Survival.time", "deadstatus.event"])
y = merged_df["deadstatus.event"].astype(int)

# Keep only numeric columns
X = X.select_dtypes(include=[np.number]).dropna(axis=1, how='any')

# -------------------------
# Define pipeline
# -------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])

# -------------------------
# Cross-validation
# -------------------------
auc_scores = cross_val_score(
    pipeline, X, y, cv=5,
    scoring=make_scorer(roc_auc_score)
)

f1_scores = cross_val_score(
    pipeline, X, y, cv=5,
    scoring=make_scorer(f1_score)
)

print("[✓] Model B (clinical1 + segmentation features) Results")
print(f"AUC (mean ± std): {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
print(f"F1  (mean ± std): {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
