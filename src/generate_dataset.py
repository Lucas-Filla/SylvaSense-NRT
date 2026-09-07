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

    #Coordinates for active deforestation zones in Brazil
    hotspots = [
        (-63.90, -8.76),
        (-63.85, -8.80),
        (-63.80, -8.85),
        (-63.75, -8.90),
        (-63.70, -8.95)
    ]

    all_X, all_y = [], []

    #For all the hostpots!!
    for lon, lat in hotspots:
        print(f"Processing hostpot: {lon}, {lat}")
        #Using 4.5km buffer to cut into dozens of clean patches
        #changed from 10 to 4.5 due to request size limits -> will have to write a wrapper loop to query multiple coordinates at 4.5km each
        aoi = get_aoi(lon, lat, 4.5)

        raw_image = fetch(aoi, '2025-01-01', '2026-05-12')
        formatted_image = prepare_image(raw_image, aoi)
        label_image = get_labels(aoi)

        image_array = download_patch(formatted_image, aoi)
        label_array = download_patch(label_image, aoi)

        X_patches, y_patches = slice_into_patches(image_array, label_array)

        all_X.append(X_patches)
        all_y.append(y_patches)

    final_X = np.vstack(all_X)
    final_y = np.vstack(all_y)

    print(f"Generated {final_X.shape[0]} training patches")

    os.makedirs("data", exist_ok=True)
    np.save("data/X_gee_train.npy", final_X)
    np.save("data/y_gee_train.npy", final_y)
    print("Saved to data/X_gee_train.npy and data/y_gee_train.npy")



