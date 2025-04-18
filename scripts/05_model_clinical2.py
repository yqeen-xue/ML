import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import os

# Function to load and prepare data
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Drop ID column if present
    if "PatientID" in df.columns:
        df = df.drop(columns=["PatientID"])

    # Drop invalid or missing outcome values
    df = df[df["time"] >= 0].dropna(subset=["status", "time"])

    # Create structured y array for survival analysis
    y_df = df[["status", "time"]].copy()
    y_df["status"] = y_df["status"].astype(bool)
    y = np.array(list(y_df.itertuples(index=False)), dtype=[("event", bool), ("time", float)])

    # Prepare features
    X = df.drop(columns=["status", "time"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())  # Fill remaining NaNs with column means

    return X, y

# Main function to train model and evaluate performance
def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"

    # Load data
    X, y = load_and_prepare_data(filepath)

    if len(X) == 0:
        print("❗ No valid samples available for training.")
        return

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define pipeline with scaling and RSF model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rsf", RandomSurvivalForest(random_state=42))
    ])

    # Define grid search parameters
    param_grid = {
        "rsf__n_estimators": [100, 200],
        "rsf__max_depth": [3, 5],
        "rsf__min_samples_leaf": [3, 5],
    }

    # Run grid search
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    # Evaluate model performance
    c_index = concordance_index_censored(
        y_test["event"], y_test["time"], gs.predict(X_test)
    )[0]

    print("Best Params:", gs.best_params_)
    print("C-index on test set:", round(c_index, 4))

    # Save results to CSV
    pd.DataFrame([{
        "model": "RandomSurvivalForest_pipeline",
        "c_index": round(c_index, 4),
        **gs.best_params_
    }]).to_csv("results/clinical_model_results.csv", index=False)

if __name__ == "__main__":
    main()
