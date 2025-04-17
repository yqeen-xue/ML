import pandas as pd

# Load full clinical2.csv (not the cleaned one)
df_raw = pd.read_csv("/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/clinical2.csv")
df_raw = df_raw.replace(["Not Collected", "N/A", "Not collected", "Not Assessed", "Unchecked", "Not Recorded In Database"], pd.NA)
df_raw["status"] = df_raw["Survival Status"].map({"Dead": 1, "Alive": 0})

# Load RNA sample IDs
rna = pd.read_csv("/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/rnaseq.txt", sep="\t", nrows=1)
rna_samples = list(rna.columns[1:])

# Filter clinical2 based on RNA samples
df_rna = df_raw[df_raw["Case ID"].isin(rna_samples)]

# Optional: select variables you're interested in
columns = ["Case ID", "status", "Age at Histological Diagnosis", "Gender", "Smoking status",
           "Histology ", "EGFR mutation status", "KRAS mutation status", "ALK translocation status", "Time to Death (days)"]
df_rna = df_rna[columns].copy()
df_rna.rename(columns={
    "Case ID": "PatientID",
    "Age at Histological Diagnosis": "age",
    "Gender": "gender",
    "Smoking status": "smoking",
    "Histology ": "histology",
    "EGFR mutation status": "EGFR",
    "KRAS mutation status": "KRAS",
    "ALK translocation status": "ALK",
    "Time to Death (days)": "time"
}, inplace=True)

# Save
df_rna.to_csv("data/clinical2_with_rna.csv", index=False)
print("Done. Saved full clinical2 with RNA coverage to data/clinical2_with_rna.csv")
print("Shape:", df_rna.shape)
print("Status breakdown:\n", df_rna["status"].value_counts())
