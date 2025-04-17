import pandas as pd
import numpy as np
import argparse
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='Preprocess clinical1.csv')
parser.add_argument('--input', type=str, required=True, help='Path to clinical1.csv')
parser.add_argument('--output', type=str, required=True, help='Path to save cleaned output')
args = parser.parse_args()


df = pd.read_csv(args.input)

print(f"data: {df.shape}")
print(f"list: {df.columns.tolist()}")

numeric_cols = ['age', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 'Survival.time']
id_col = 'PatientID'
target_col = 'deadstatus.event'

df_numeric = df[numeric_cols]

# KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)

# Standard
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=numeric_cols)

df_clean = pd.concat([df[[id_col, target_col]].reset_index(drop=True), df_scaled], axis=1)

df_clean.to_csv(args.output, index=False)
print(f"clean data saved in: {args.output}")
