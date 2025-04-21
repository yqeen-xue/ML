
import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

# -------------------------
# Custom extraction function (based on features_extraction_binarySegm_intensity_update.py)
# -------------------------
def extract_all_features(seg_folder):
    features = {}
    try:
        for file in os.listdir(seg_folder):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                filepath = os.path.join(seg_folder, file)
                img = nib.load(filepath)
                data = img.get_fdata()

                mask = data > 0
                if np.sum(mask) == 0:
                    continue

                values = data[mask]

                features[f"{file}_mean"] = np.mean(values)
                features[f"{file}_std"] = np.std(values)
                features[f"{file}_min"] = np.min(values)
                features[f"{file}_max"] = np.max(values)
                features[f"{file}_volume"] = np.sum(mask)
    except Exception as e:
        raise RuntimeError(f"Failed to extract from {seg_folder}: {e}")

    return features

# -------------------------
# Constants and paths
# -------------------------
DATASET1_PATH = "/user/home/ms13525/scratch/mshds-ml-data-2025/dataset1"
OUTPUT_PATH = "outputs/image_features_dataset1.csv"

# -------------------------
# Get all LUNG1 patient directories
# -------------------------
def get_patient_dirs():
    return [d for d in os.listdir(DATASET1_PATH) if d.startswith("LUNG1-")]

# -------------------------
# Locate segmentation path per patient
# -------------------------
def get_segmentation_path(patient_dir):
    base_path = os.path.join(DATASET1_PATH, patient_dir)
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if "Segmentation" in d:
                return os.path.join(root, d)
    return None

# -------------------------
# Main routine
# -------------------------
def main():
    patient_dirs = get_patient_dirs()
    features_list = []

    for pid in tqdm(patient_dirs):
        seg_path = get_segmentation_path(pid)
        if seg_path is None:
            print(f"[!] No segmentation found for {pid}, skipped.")
            continue

        try:
            feats = extract_all_features(seg_path)
            feats['patient_id'] = pid
            features_list.append(feats)
        except Exception as e:
            print(f"[!] Error processing {pid}: {e}")
            continue

    df = pd.DataFrame(features_list)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Features saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
