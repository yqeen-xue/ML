import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, f1_score

# -------------------------
# Load clinical1 data
# -------------------------
df = pd.read_csv("data/clinical1.csv")

# -------------------------
# Prepare features and labels
# -------------------------
X = df.drop(columns=["PatientID", "Survival.time", "deadstatus.event"])
y = df["deadstatus.event"].astype(int)  # binary outcome (0 = alive, 1 = dead)

# Drop non-numeric or empty columns if necessary
X = X.select_dtypes(include=[np.number]).dropna(axis=1, how='any')

# -------------------------
# Define pipeline: scaler + classifier
# -------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])

# -------------------------
# Cross-validate model
# -------------------------
auc_scores = cross_val_score(
    pipeline, X, y, cv=5,
    scoring=make_scorer(roc_auc_score)
)

f1_scores = cross_val_score(
    pipeline, X, y, cv=5,
    scoring=make_scorer(f1_score)
)

print("[✓] Model A (clinical1-only) Results")
print(f"AUC (mean ± std): {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
print(f"F1  (mean ± std): {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
