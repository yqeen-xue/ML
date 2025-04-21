
import pandas as pd
import numpy as np
import os

# -------------------------
# Load full clinical2.csv
# -------------------------
clinical_path = "/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/clinical2.csv"
clinical = pd.read_csv(clinical_path)
clinical = clinical.replace([
    "Not Collected", "N/A", "Not collected",
    "Not Assessed", "Unchecked", "Not Recorded In Database"
], pd.NA)

# Add binary status
clinical["status"] = clinical["Survival Status"].map({"Dead": 1, "Alive": 0})
clinical["Case ID"] = clinical["Case ID"].str.strip()

# -------------------------
# Load RNA expression matrix
# -------------------------
rna_path = "/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/rnaseq.txt"
rna = pd.read_csv(rna_path, sep="\t")
rna = rna.rename(columns={rna.columns[0]: "Gene"})

# Extract list of RNA patient IDs (sample IDs)
rna_sample_ids = rna.columns[1:]  # skip 'Gene'

# -------------------------
# Filter clinical samples with RNA
# -------------------------
clinical_rna = clinical[clinical["Case ID"].isin(rna_sample_ids)].copy()

# Keep selected clinical columns
columns = [
    "Case ID", "status", "Age at Histological Diagnosis", "Gender",
    "Smoking status", "Histology ", "EGFR mutation status",
    "KRAS mutation status", "ALK translocation status", "Time to Death (days)"
]
clinical_rna = clinical_rna[columns].copy()
clinical_rna.rename(columns={
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

# -------------------------
# Transpose RNA matrix to rows = patients, columns = genes
# -------------------------
rna_filtered = rna[["Gene"] + [pid for pid in clinical_rna["PatientID"] if pid in rna.columns]]
rna_transposed = rna_filtered.set_index("Gene").T
rna_transposed.index.name = "PatientID"
rna_transposed.reset_index(inplace=True)

# -------------------------
# Merge clinical + RNA features
# -------------------------
merged = pd.merge(clinical_rna, rna_transposed, on="PatientID", how="inner")
merged = merged.dropna(subset=["status", "time"])

# -------------------------
# Save result
# -------------------------
os.makedirs("data", exist_ok=True)
merged.to_csv("data/clinical2_rna_merged.csv", index=False)

print("[✓] Saved merged clinical + RNA to data/clinical2_rna_merged.csv")
print("Shape:", merged.shape)
print("Status breakdown:\n", merged["status"].value_counts())