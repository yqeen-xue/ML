import pandas as pd

# Load clinical data with RNA
clinical = pd.read_csv("data/clinical2_with_rna.csv")
sample_ids = clinical["PatientID"].tolist()

# Load RNA expression data
rna_path = "/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/rnaseq.txt"
rna = pd.read_csv(rna_path, sep="\t")

# Rename gene column
rna = rna.rename(columns={rna.columns[0]: "Gene"})

# Filter columns: keep genes + samples with RNA
rna_filtered = rna[["Gene"] + [s for s in sample_ids if s in rna.columns]]

# Transpose: rows = samples, columns = genes
rna_transposed = rna_filtered.set_index("Gene").T
rna_transposed.index.name = "PatientID"
rna_transposed.reset_index(inplace=True)

# Merge with clinical
merged = pd.merge(clinical, rna_transposed, on="PatientID", how="inner")

# Save merged file
merged.to_csv("data/clinical2_rna_merged.csv", index=False)
print("Merged clinical + RNA saved to: data/clinical2_rna_merged.csv")
print("Final shape:", merged.shape)
