# scripts/06_model_clinical1.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import os

def load_data(filepath):
    df = pd.read_csv(filepath)

    if "PatientID" in df.columns:
        df = df.drop(columns=["PatientID"])

    # Drop invalid survival times
    df = df[df["Survival.time"] >= 0].dropna(subset=["Survival.time", "deadstatus.event"])

    # Prepare survival label
    y = df[["deadstatus.event", "Survival.time"]].copy()
    y["deadstatus.event"] = y["deadstatus.event"].astype(bool)
    y_struct = np.array(list(y.itertuples(index=False)), dtype=[("event", bool), ("time", float)])

    # Select features
    X = df.drop(columns=["deadstatus.event", "Survival.time"])

    return X, y_struct

def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/cleaned_clinical1.csv"

    # Load data
    X, y = load_data(filepath)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline
    rsf = RandomSurvivalForest(random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("rsf", rsf)])

    # Grid search parameters
    param_grid = {
        "rsf__n_estimators": [100, 200],
        "rsf__max_depth": [3, 5, 7],
        "rsf__min_samples_leaf": [3, 5],
    }

    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    # Evaluate
    c_index = concordance_index_censored(
        y_test["event"], y_test["time"], gs.predict(X_test)
    )[0]

    print("Best Params:", gs.best_params_)
    print("C-index on test set:", round(c_index, 4))

    # Save results
    pd.DataFrame([{
        "model": "RSF_clinical1",
        "c_index": round(c_index, 4),
        **gs.best_params_
    }]).to_csv("results/clinical1_model_results.csv", index=False)

if __name__ == "__main__":
    main()
