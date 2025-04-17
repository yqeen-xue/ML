import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import os

def load_data(filepath, clinical_only=False):
    df = pd.read_csv(filepath)

    if "PatientID" in df.columns:
        df = df.drop(columns=["PatientID"])

    # Drop rows with negative or missing survival info
    df = df[df["time"] >= 0].dropna(subset=["time", "status"])

    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Extract y
    y = df[["status", "time"]].copy()
    y["status"] = y["status"].astype(bool)
    y_struct = np.array(list(y.itertuples(index=False)), dtype=[("event", bool), ("time", float)])

    # Select features
    if clinical_only:
        clinical_cols = ["age", "gender", "smoking", "histology", "EGFR", "KRAS", "ALK"]
        X = df[clinical_cols]
    else:
        X = df.drop(columns=["status", "time"])

    return X, y_struct

def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"

    X, y = load_data(filepath, clinical_only=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rsf = RandomSurvivalForest(random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("rsf", rsf)])

    param_grid = {
        "rsf__n_estimators": [100, 200],
        "rsf__max_depth": [3, 5, 7],
        "rsf__min_samples_leaf": [3, 5],
    }

    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    c_index = concordance_index_censored(
        y_test["event"], y_test["time"], gs.predict(X_test)
    )[0]

    print("Best Params:", gs.best_params_)
    print("C-index on test set:", round(c_index, 4))

    pd.DataFrame([{
        "model": "RandomSurvivalForest_pipeline",
        "c_index": round(c_index, 4),
        **gs.best_params_
    }]).to_csv("results/clinical_model_results.csv", index=False)

if __name__ == "__main__":
    main()
