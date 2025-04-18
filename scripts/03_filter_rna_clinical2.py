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


df_rna.to_csv("data/clinical2_with_rna.csv", index=False)
print("Done. Saved full clinical2 with RNA coverage to data/clinical2_with_rna.csv")
print("Shape:", df_rna.shape)
print("Status breakdown:\n", df_rna["status"].value_counts())
