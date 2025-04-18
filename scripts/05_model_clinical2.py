import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

def load_data(filepath):
    """
    Load and preprocess the merged clinical + RNA dataset.
    Returns:
        X (features), y (structured survival data)
    """
    df = pd.read_csv(filepath)

    # Drop non-feature columns if present
    if "PatientID" in df.columns:
        df = df.drop(columns=["PatientID"])

    # Ensure time and status are valid
    df = df[df["time"] >= 0]
    df = df.dropna(subset=["time", "status"])

    # Make sure all inputs are numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()  # Drop rows with any remaining NaNs

    # Extract y (structured array)
    y_df = df[["status", "time"]].copy()
    y_df["status"] = y_df["status"].astype(bool)
    y_struct = np.array(list(y_df.itertuples(index=False)), dtype=[("event", bool), ("time", float)])

    # Extract X (features)
    X = df.drop(columns=["status", "time"])

    return X, y_struct

def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"
    X, y = load_data(filepath)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define model and pipeline
    rsf = RandomSurvivalForest(random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rsf", rsf)
    ])

    # Define hyperparameter grid
    param_grid = {
        "rsf__n_estimators": [100, 200],
        "rsf__max_depth": [3, 5, 7],
        "rsf__min_samples_leaf": [3, 5]
    }

    # Run grid search with 3-fold CV
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    # Evaluate on test set
    c_index = concordance_index_censored(
        y_test["event"], y_test["time"], gs.predict(X_test))[0]

    print("Best Params:", gs.best_params_)
    print("C-index on test set:", round(c_index, 4))

    # Save results
    pd.DataFrame([{
        "model": "RandomSurvivalForest_pipeline",
        "c_index": round(c_index, 4),
        **gs.best_params_
    }]).to_csv("results/clinical_model_results.csv", index=False)

if __name__ == "__main__":
    main()
