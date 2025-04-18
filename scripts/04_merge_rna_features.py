import pandas as pd

# Load clinical data with RNA
clinical = pd.read_csv("data/clinical2_with_rna.csv")
sample_ids = clinical["PatientID"].tolist()

# Load full cleaned clinical2 to extract no-RNA samples
full_clinical2 = pd.read_csv("data/cleaned_clinical2.csv")

# Filter out patients that do NOT have RNA data
no_rna_clinical = full_clinical2[~full_clinical2["PatientID"].isin(sample_ids)]
no_rna_clinical.to_csv("data/clinical2_norna.csv", index=False)
print("Saved clinical2 samples WITHOUT RNA to: data/clinical2_norna.csv")
print("Shape:", no_rna_clinical.shape)

# Load RNA expression data
rna_path = "/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/rnaseq.txt"
rna = pd.read_csv(rna_path, sep="\t")

# Rename gene column
rna = rna.rename(columns={rna.columns[0]: "Gene"})

# Filter RNA columns to match clinical RNA samples
rna_filtered = rna[["Gene"] + [s for s in sample_ids if s in rna.columns]]

# Transpose RNA: rows = samples, columns = genes
rna_transposed = rna_filtered.set_index("Gene").T
rna_transposed.index.name = "PatientID"
rna_transposed.reset_index(inplace=True)

# Merge RNA with clinical data
merged = pd.merge(clinical, rna_transposed, on="PatientID", how="inner")
merged.to_csv("data/clinical2_rna.csv", index=False)

print("Merged clinical + RNA saved to: data/clinical2_rna.csv")
print("Final shape:", merged.shape)
