import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import os

# Load and prepare the data
def load_data(filepath):
    df = pd.read_csv(filepath)

    if "PatientID" in df.columns:
        df = df.drop(columns=["PatientID"])

    # Drop negative or missing survival times
    df = df[df["time"] >= 0].dropna(subset=["time", "status"])

    # Convert all to numeric (some RNA columns may be non-numeric)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with any remaining missing values
    df = df.dropna()

    # Extract y (status + time)
    y = df[["status", "time"]].copy()
    y["status"] = y["status"].astype(bool)
    y_struct = np.array(list(y.itertuples(index=False)), dtype=[("event", bool), ("time", float)])

    # Features = everything else
    X = df.drop(columns=["status", "time"])

    return X, y_struct

# Main execution
def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"

    # Load and clean data
    X, y = load_data(filepath)

    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=50)  # or adjust n_components if needed
    X_pca = pca.fit_transform(X_scaled)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Pipeline (only the model, since data is already scaled + reduced)
    rsf = RandomSurvivalForest(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [3, 5],
    }

    gs = GridSearchCV(rsf, param_grid=param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    # Evaluate
    c_index = concordance_index_censored(y_test["event"], y_test["time"], gs.predict(X_test))[0]

    print("Best Params:", gs.best_params_)
    print("C-index on test set:", round(c_index, 4))

    # Save results
    pd.DataFrame([{
        "model": "RandomSurvivalForest_PCA",
        "c_index": round(c_index, 4),
        **gs.best_params_
    }]).to_csv("results/clinical_model_results_pca.csv", index=False)

if __name__ == "__main__":
    main()
