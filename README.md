
# README

This repository contains all scripts and resources used in the survival classification study on two lung cancer datasets (Dataset1 and Dataset2), integrating clinical, image, and transcriptomic features.

---

## 1. Environment Setup

Before running any script, create a conda environment:

```bash
conda env create -f env.yml
conda activate ml-env
```

---

## 2. Reproducible Pipeline Execution

For full pipeline execution from data preprocessing to model evaluation:

```bash
chmod +x runall.sh  # (only once)
./runall.sh
```

This script sequentially runs:

- Data preprocessing for clinical1 and clinical2
- RNA and image feature merging
- Image feature extraction
- Model training for both Dataset1 and Dataset2

All results will be saved in the `results/` directory.

---

## 3. Step-by-step Manual Execution (If runall.sh fails)

Each step can be run individually as follows:

### (1) Clinical Data Preprocessing

```bash
python scripts/01_pre_clinical1.py --input /user/home/ms13525/scratch/mshds-ml-data-2025/dataset1/clinical1.csv --output data/cleaned_clinical1.csv

python scripts/02_pre_clinical2.py --input /user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/clinical2.csv --output data/cleaned_clinical2.csv
```

### (2) Merge RNA with Clinical2

```bash
python scripts/03_merge_rna_clinical2.py
```

### (3) Image Feature Extraction

```bash
python scripts/04_extract_image_features_dataset1.py
python scripts/05_extract_image_features_dataset2.py
```

### (4) Train Models on Dataset 1

```bash
python scripts/D1_modela_clinical1.py
python scripts/D1_modelb_clinical1.py
```

### (5) Train Models on Dataset 2

```bash
python scripts/D2_modelA_clinical2.py
python scripts/D2_modelB_clinical2rna.py
python scripts/D2_modelC_clinical2image.py
python scripts/D2_modelD_clinical2rnaimage.py
```

---

## 📂 4. Directory Structure

```
ML/
├── data/                      # Raw and cleaned data files
├── outputs/                   # Image features
├── results/                   # Model outputs
├── scripts/                   # All Python scripts
│   ├── 01_pre_clinical1.py
│   ├── 02_pre_clinical2.py
│   ├── 03_merge_rna_clinical2.py
|   ├── 04_extract_image_features_dataset1.py
|   ├── 05_extract_image_features_dataset2.py
|   ├── D1_modela_clinical1.py
|   ├── D1_modelb_clinical1image.py
|   ├── D2_modelA_clinical2.py
|   ├── D2_modelB_clinical2rna.py
|   ├── D2_modelC_clinical2image.py
│   └── D2_modelD_clinical2rnaimage.py
├── runall.sh                  # Unified execution script
├── env.yml                    # Conda environment
└── README.md
```

---

## Notes

- All classifiers use stratified splitting and class weighting to mitigate imbalances.
- All intermediate CSV outputs are reused across steps.
- Results include AUC, F1, and Accuracy metrics per model.
