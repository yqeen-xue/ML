import pandas as pd

# Read clinical data and pre
clinical_path = "/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/clinical2.csv"
df_raw = pd.read_csv(clinical_path)

df_raw = df_raw.replace([
    "Not Collected", "N/A", "Not collected", "Not Assessed",
    "Unchecked", "Not Recorded In Database"
], pd.NA)

# Add status（1 = Dead, 0 = Alive）
df_raw["status"] = df_raw["Survival Status"].map({"Dead": 1, "Alive": 0})
df_raw["Case ID"] = df_raw["Case ID"].str.strip()  

# filter RNA data with clinical ID
rna = pd.read_csv("/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/rnaseq.txt", sep="\t", nrows=1)
rna_sample_ids = list(rna.columns[1:])
df_rna_clinical = df_raw[df_raw["Case ID"].isin(rna_sample_ids)].copy()
df_rna_clinical.rename(columns={"Case ID": "PatientID"}, inplace=True)

df_rna_clinical.to_csv("data/clinical2_rna_clinical_only.csv", index=False)

# load RNA 
rna_full = pd.read_csv("/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/rnaseq.txt", sep="\t")
rna_full = rna_full.rename(columns={rna_full.columns[0]: "Gene"})
rna_filtered = rna_full[["Gene"] + rna_sample_ids]  

rna_transposed = rna_filtered.set_index("Gene").T
rna_transposed.index.name = "PatientID"
rna_transposed.reset_index(inplace=True)

# merge RNA with clinical
merged = pd.merge(df_rna_clinical, rna_transposed, on="PatientID", how="inner")
merged.to_csv("data/clinical2_rna_merged.csv", index=False)

print("[✓] Saved merged clinical + RNA to data/clinical2_rna_merged.csv")
print("Shape:", merged.shape)
print("Status breakdown:\n", merged["status"].value_counts())
