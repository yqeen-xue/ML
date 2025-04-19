# scripts/05_extract_image_features_dataset2.py

import os
import numpy as np
import pandas as pd
import pydicom
from skimage.feature import graycomatrix, greycoprops
from tqdm import tqdm

# -------------------------
# Load a DICOM series into a 3D volume
# -------------------------
def load_dicom_volume(dcm_dir):
    dcm_files = sorted([
        os.path.join(dcm_dir, f) for f in os.listdir(dcm_dir) if f.endswith('.dcm')
    ])
    slices = [pydicom.dcmread(f).pixel_array for f in dcm_files]
    volume = np.stack(slices, axis=0)
    return volume

# -------------------------
# Extract intensity and texture features from volume
# -------------------------
def extract_features(volume):
    features = {}

    # Compute mean projection along axial slices
    mean_img = np.mean(volume, axis=0)

    # Normalize to 0-255 for texture analysis
    scaled_img = ((mean_img - mean_img.min()) / (mean_img.max() - mean_img.min()) * 255).astype(np.uint8)

    # Basic intensity statistics
    features['mean_intensity'] = np.mean(mean_img)
    features['std_intensity'] = np.std(mean_img)
    features['min_intensity'] = np.min(mean_img)
    features['max_intensity'] = np.max(mean_img)

    # Texture features from GLCM
    glcm = graycomatrix(scaled_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = greycoprops(glcm, 'contrast')[0, 0]
    features['glcm_homogeneity'] = greycoprops(glcm, 'homogeneity')[0, 0]
    features['glcm_energy'] = greycoprops(glcm, 'energy')[0, 0]
    features['glcm_correlation'] = greycoprops(glcm, 'correlation')[0, 0]

    return features

# -------------------------
# Main function to iterate over AMC and R01 patients
# -------------------------
def main():
    root = "/user/home/ms13525/scratch/mshds-ml-data-2025/dataset2"
    output_csv = "outputs/image_features_dataset2.csv"
    patient_ids = [d for d in os.listdir(root) if d.startswith("AMC") or d.startswith("R01")]

    results = []

    for pid in tqdm(patient_ids):
        patient_dir = os.path.join(root, pid)
        subdirs = [os.path.join(patient_dir, s) for s in os.listdir(patient_dir)]

        # Collect candidate subfolders with 'CT' in name
        candidate_ct_dirs = []
        for sub in subdirs:
            if os.path.isdir(sub):
                inner_dirs = [os.path.join(sub, name) for name in os.listdir(sub)
                              if os.path.isdir(os.path.join(sub, name)) and 'CT' in name.upper()]
                candidate_ct_dirs.extend(inner_dirs)

        if not candidate_ct_dirs:
            print(f"[!] No CT folder found for {pid}")
            continue

        # Choose the CT folder with the most DICOM files
        best_ct_dir = max(candidate_ct_dirs, key=lambda d: len([f for f in os.listdir(d) if f.endswith(".dcm")]))
        try:
            volume = load_dicom_volume(best_ct_dir)
            feats = extract_features(volume)
            feats['patient_id'] = pid
            results.append(feats)
        except Exception as e:
            print(f"[!] Skipped {pid} due to error: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Features saved to {output_csv}")

if __name__ == "__main__":
    main()
