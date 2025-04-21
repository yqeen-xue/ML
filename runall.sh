#!/bin/bash
set -e  

echo "STEP 1: Data Cleaning"
python scripts/01_pre_clinical1.py  \
--input /user/home/ms13525/scratch/mshds-ml-data-2025/dataset1/clinical1.csv \
--output data/cleaned_clinical1.csv

python scripts/02_pre_clinical2.py \
--input /user/home/ms13525/scratch/mshds-ml-data-2025/dataset2/clinical2.csv \
--output data/cleaned_clinical2.csv


echo ""
echo "STEP 2: Merge RNA with clinical2"
python scripts/03_merge_rna_clinical2.py

echo ""
echo "STEP 3: Image Feature Extraction"
python scripts/04_extract_image_features_dataset1.py
python scripts/05_extract_image_features_dataset2.py

echo ""
echo "STEP 5: Model Training — Dataset 1"
python scripts/D1_modela_clinical1.py
python scripts/D1_modelb_clinical1.py

echo ""
echo "STEP 4: Model Training — Dataset 2"
python scripts/D2_modelA_clinical2.py
python scripts/D2_modelB_clinical2rna.py
python scripts/D2_modelC_clinical2image.py
python scripts/D2_modelD_clinical2rnaimage.py


echo ""
echo "ALL STEPS COMPLETED SUCCESSFULLY"
echo "Results saved in: results/"
