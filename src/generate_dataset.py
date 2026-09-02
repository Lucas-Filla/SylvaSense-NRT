import os
import numpy as np
import ee
from dotenv import load_dotenv
from data_pipeline import get_aoi, fetch, prepare_image, get_labels, download_patch

def slice_into_patches(image_array, label_array, patch_size=256):
    """Takes big 3d raster array and slices it into list of 256x256 patches"""
    channels, height, width = image_array.shape
    patches_x = []
    patches_y = []

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            #SLICE 6-channel image
            img_patch = image_array[:, y:y+patch_size, x:x+patch_size]
            #SLICE label mask
            lbl_patch = label_array[:, y:y+patch_size, x:x+patch_size]

            patches_x.append(img_patch)
            patches_y.append(lbl_patch)

    return np.array(patches_x), np.array(patches_y)

if __name__ == "__main__":
    load_dotenv()
    project_id = os.getenv("EE_PROJECT_ID")
    ee.Initialize(project=project_id)
    #Using 4.5km buffer to cut into dozens of clean patches
    #changed from 10 to 4.5 due to request size limits -> will have to write a wrapper loop to query multiple coordinates at 4.5km each
    aoi = get_aoi(-63.90, -8.76, 4.5)

    print("Fetching multi-spectral imagery and forest loss labels from GEE...")
    raw_image = fetch(aoi, '2025-01-01', '2026-05-12')
    formatted_image = prepare_image(raw_image, aoi)
    label_image = get_labels(aoi)

    print("Downloading regional image composite...")
    image_array = download_patch(formatted_image, aoi)

    print("Downloading regional label mask...")
    label_array = download_patch(label_image, aoi)

    print("SLICING!!")
    X_patches, y_patches = slice_into_patches(image_array, label_array)
    print(f"Generated {X_patches.shape[0]} new training patches")

    os.makedirs("data", exist_ok=True)
    np.save("data/X_gee_train.npy", X_patches)
    np.save("data/y_gee_train.npy", y_patches)
    print("Saved to data/X_gee_train.npy and data/y_gee_train.npy")



