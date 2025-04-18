import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# Load and preprocess the data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df["time"] >= 0]
    df = df.dropna(subset=["status", "time"])

    # Extract survival information
    y = df[["status", "time"]].copy()
    y["status"] = y["status"].astype(bool)
    y_struct = np.array(list(y.itertuples(index=False)), dtype=[("event", bool), ("time", float)])

    # Extract feature matrix and drop ID/status/time columns
    X = df.drop(columns=["PatientID", "status", "time"])

    # Convert non-numeric (object) columns to numeric using label encoding
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return X, y_struct

# Main execution function
def main():
    os.makedirs("results", exist_ok=True)
    filepath = "data/clinical2_rna_merged.csv"
    X, y = load_data(filepath)

    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Apply PCA (keep 95% variance)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Define the Random Survival Forest pipeline
    rsf = RandomSurvivalForest(random_state=42)
    pipe = Pipeline([("rsf", rsf)])

    # Define parameter grid for grid search
    param_grid = {
        "rsf__n_estimators": [100, 200],
        "rsf__max_depth": [3, 5],
        "rsf__min_samples_leaf": [3, 5],
    }

    # Run grid search with 3-fold cross-validation
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    # Evaluate the model
    c_index = concordance_index_censored(y_test["event"], y_test["time"], gs.predict(X_test))[0]
    print("Best Params:", gs.best_params_)
    print("C-index on test set:", round(c_index, 4))

    # Save the result
    pd.DataFrame([{
        "model": "RandomSurvivalForest_PCA_KNN",
        "c_index": round(c_index, 4),
        **gs.best_params_
    }]).to_csv("results/clinical_model_results_pca.csv", index=False)

if __name__ == "__main__":
    main()