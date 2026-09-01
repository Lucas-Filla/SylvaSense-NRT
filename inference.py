import torch
import numpy as np
import matplotlib.pyplot as plt

#Imports UNet architecture from /src/
from src.models.unet import UNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Inferring on device: {device}")

    model = UNet(in_channels=6, out_channels=1).to(device)
    #load weights from trained models
    model.load_state_dict(torch.load("unet_weights.pth", weights_only=True))
    #locks model for testing
    model.eval()

    print("\nLoading test sample...")
    #Load first 6-channel image and corresponsding ground truth
    X_sample = np.load('data/X_train.npy')[0]
    y_sample = np.load('data/y_train.npy')[0]
    #Conver to PyTorch tensor
    #Unsqueeze fakes a batch size of 1, turning shape (6, 256, 256) into (1, 6, 256, 256)
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0).to(device)

    print("Inferring...")
    with torch.no_grad():
        raw_logits = model(X_tensor)
        #squash raw logits into 0.0 to 1.0 percentages
        probabilities = torch.sigmoid(raw_logits)
        #Force anything above 50%/0.5 certainty to 1 (deforestation), else 0
        prediction_mask = (probabilities > 0.5).float()

    print("\nGenerating visualization...")
    #prepare prediction for visualization
    pred_vis = prediction_mask.squeeze().cpu().numpy()
    #prepare ground truth
    gt_vis = y_sample.squeeze()
    #satellite image
    #first 3 channels for visual approx. and transpose to Height, Width, Channels
    sat_vis = X_sample[:3, :, :].transpose(1, 2, 0)
    #normalize raw array to clean 0.0-1.0 scale so Matplotlib renders it correctly
    sat_vis = (sat_vis - sat_vis.min()) / (sat_vis.max() - sat_vis.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].imshow(sat_vis)
    axes[0].set_title("Satellite Input (approx.)")
    axes[0].axis("off")

    axes[1].imshow(gt_vis, cmap="gray")
    axes[1].set_title("Hansen Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_vis, cmap="gray")
    axes[2].set_title("U-Net Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    #Save directly to hard drive
    plt.savefig("results/inference_result.png")
    print("Visualization saced to results/inference_result.png")