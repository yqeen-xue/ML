import argparse
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Argument parser
parser = argparse.ArgumentParser(description="Clean clinical2 dataset")
parser.add_argument('--input', type=str, required=True, help='Path to clinical2.csv')
parser.add_argument('--output', type=str, required=True, help='Path to save cleaned data')
args = parser.parse_args()

# Read raw clinical2 data
df = pd.read_csv(args.input)

print("Original shape:", df.shape)

# Replace known missing values with np.nan
df = df.replace(["Not Collected", "N/A", "Not collected", "Not Assessed", "Unchecked", "Not Recorded In Database"], np.nan)

# Rename columns for convenience
df.rename(columns={
    "Case ID": "PatientID",
    "Age at Histological Diagnosis": "age",
    "Gender": "gender",
    "Smoking status": "smoking",
    "Histology ": "histology",
    "EGFR mutation status": "EGFR",
    "KRAS mutation status": "KRAS",
    "ALK translocation status": "ALK",
    "Survival Status": "status",
    "Time to Death (days)": "time"
}, inplace=True)

# Select variables of interest
columns_to_use = ["PatientID", "age", "gender", "smoking", "histology", "EGFR", "KRAS", "ALK", "status", "time"]
df = df[columns_to_use]

# Drop rows without survival time or status
df = df.dropna(subset=["status", "time"])

# Create binary status variable (1 = dead, 0 = alive)
df["status"] = df["status"].apply(lambda x: 1 if str(x).lower().strip() == "dead" else 0)

# Encode categorical variables
categorical = ["gender", "smoking", "histology", "EGFR", "KRAS", "ALK"]
for col in categorical:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Apply KNN imputation on all columns except ID
df_impute = df.drop(columns=["PatientID"])
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=df_impute.columns)

# Standardize numerical features
scaler = StandardScaler()
numerical = ["age", "time"]
df_imputed[numerical] = scaler.fit_transform(df_imputed[numerical])

# Merge PatientID back
df_imputed["PatientID"] = df["PatientID"].values
df_final = df_imputed[["PatientID", "status", "age", "gender", "smoking", "histology", "EGFR", "KRAS", "ALK", "time"]]

# Save cleaned data
df_final.to_csv(args.output, index=False)
print("Cleaned data is in:", args.output)
print("Shape:", df_final.shape)
