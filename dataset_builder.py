import rasterio
import numpy as np

def build_dataset(filepath, patch_size=256):
    print("Loading GeoTIFF into memory...")
    with rasterio.open(filepath) as src:
        # rasterio reads the arrays in the shape
        master_tensor = src.read()

    print(f"Master Tensor Shape: {master_tensor.shape}")
    # Features (X)
    # Selects layers 0-5 
    X = master_tensor[0:6, :, :]
    # Labels (Y)
    # Layer 6
    y = master_tensor[6, :, :]
    #normalize features
    X = np.clip(X, 0, 10000) / 10000.0

    _, height, width = X.shape
    X_patches = []
    y_patches = []

    print(f"Slicing into {patch_size}x{patch_size} patches...")

    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            #cutting each 256x256 square
            X_patch = X[:, i:i+patch_size, j:j+patch_size]
            y_patch = y[i:i+patch_size, j:j+patch_size]

            X_patches.append(X_patch)
            y_patches.append(y_patch)

    #converting lists into final arrays
    X_train = np.array(X_patches)
    y_train = np.array(y_patches)

    print(f"Final input tensor shape: {X_train.shape}")
    print(f"Final input label shape: {y_train.shape}")
    
    #saving processed arrays to data folder
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', np.expand_dims(y_train, axis=1))
    print("Training dataset saved successfully.")

if __name__ == "__main__":
    build_dataset("data/SylvaSense_Training_PortoVelho.tif")


