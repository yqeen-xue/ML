import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import numpy as np
import os

# Load and prepare the data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df["time"] >= 0]  # Remove negative survival times
    df = df.dropna(subset=["status", "time"])  # Remove rows missing y
    y = df[["status", "time"]].copy()
    y["status"] = y["status"].astype(bool)
    y_struct = np.array(list(y.itertuples(index=False)), dtype=[("event", bool), ("time", float)])
    X = df.drop(columns=["PatientID", "status", "time"])
    return X, y_struct

# Main execution
def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"
    X, y = load_data(filepath)

    # KNN imputation + scaling
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Reduce RNA dimensionality with PCA
    pca = PCA(n_components=0.95)  # retain 95% variance
    X_pca = pca.fit_transform(X_scaled)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Define model and pipeline
    rsf = RandomSurvivalForest(random_state=42)
    pipe = Pipeline([("rsf", rsf)])

    # Grid search with 3-fold CV
    param_grid = {
        "rsf__n_estimators": [100, 200],
        "rsf__max_depth": [3, 5],
        "rsf__min_samples_leaf": [3, 5],
    }
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    # Evaluate
    c_index = concordance_index_censored(y_test["event"], y_test["time"], gs.predict(X_test))[0]

    print("Best Params:", gs.best_params_)
    print("C-index on test set:", round(c_index, 4))

    # Save results
    pd.DataFrame([{
        "model": "RandomSurvivalForest_PCA_KNN",
        "c_index": round(c_index, 4),
        **gs.best_params_
    }]).to_csv("results/clinical_model_results_pca.csv", index=False)

if __name__ == "__main__":
    main()
