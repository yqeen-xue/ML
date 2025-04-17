import pandas as pd

# Load cleaned clinical2 data
df = pd.read_csv("data/cleaned_clinical2.csv")

# Load RNA sample IDs from rnaseq.txt
rna = pd.read_csv("dataset2/rnaseq.txt", sep="\t", nrows=1)
rna_samples = list(rna.columns[1:])  # first column = gene names

# Filter clinical2 to only samples with RNA
df_rna = df[df["PatientID"].isin(rna_samples)]

print("Shape after RNA filter:", df_rna.shape)
df_rna.to_csv("data/clinical2_with_rna.csv", index=False)
