# scripts/05_extract_image_features.py

import os
import numpy as np
import pandas as pd
import nibabel as nib
import skimage.measure
from scipy.spatial.distance import pdist

# ========== Feature Extraction Functions ==========

def tumour_features(tumour_array, voxel_size):
    if np.sum(tumour_array) > 0:
        verts, faces, _, _ = skimage.measure.marching_cubes(tumour_array, 0.5, spacing=voxel_size)
        area = skimage.measure.mesh_surface_area(verts, faces)
        volume = np.sum(tumour_array > 0.1) * np.prod(voxel_size)
        radius = (3.0 / (4.0 * np.pi) * volume) ** (1.0 / 3)
        distance = pdist(verts)
        features = {
            "maximum_diameter": np.amax(distance),
            "surface_area": area,
            "surface_to_volume_ratio": area / volume,
            "volume": volume
        }
        return features
    else:
        return {
            "maximum_diameter": 0,
            "surface_area": 0,
            "surface_to_volume_ratio": 0,
            "volume": 0
        }

def save_features_csv(patient_ids, features_list, feature_names, output_path):
    df = pd.DataFrame(features_list, columns=feature_names)
    df['PatientID'] = patient_ids
    df.set_index('PatientID', inplace=True)
    df.to_csv(output_path)

# ========== Main ==========

def main():
    print("Extracting image features from segmentation masks...")

    seg_path = "/user/home/ms13525/scratch/hackathon-2025/data/automated_segm"
    patient_ids = sorted([f[:-29] for f in os.listdir(seg_path) if f.endswith("_automated_approx_segm.nii.gz")])

    features = []
    for pid in patient_ids:
        mask_path = os.path.join(seg_path, pid + "_automated_approx_segm.nii.gz")
        if not os.path.exists(mask_path):
            continue

        nii = nib.load(mask_path)
        mask_data = nii.get_fdata()
        voxel_size = nii.header.get_zooms()

        binary_mask = mask_data > 0
        feat_dict = tumour_features(binary_mask, voxel_size)
        features.append(list(feat_dict.values()))
        print(f"Processed: {pid}")

    feature_names = ["maximum_diameter", "surface_area", "surface_to_volume_ratio", "volume"]
    save_features_csv(patient_ids, features, feature_names, "data/features_segm_tumour.csv")
    print("Saved: data/features_segm_tumour.csv")

if __name__ == "__main__":
    main()
