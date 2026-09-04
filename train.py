import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

#Imports UNet architecture from /src/
from src.models.unet import UNet

def load_data(batch_size=3):
    print("Loading data...")

    #Loads the NumPy arrays from data
    X_train = np.load('data/X_gee_train.npy')
    y_train = np.load('data/y_gee_train.npy')

    #Converts raw NumPy arrays into PyTorch Tensors (FLoat32 is used for weights)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    #package the features and labels into one dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Total patches loaded: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)} (Batch size: {batch_size})")
    return loader


if __name__ == "__main__":
    #Switch to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader = load_data()

    model = UNet(in_channels=6, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss() #Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3
    print("\nTraining loop go!")
    for epoch in range(epochs):
        #Will track total loss for a later avg
        epoch_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            #clear old gradients
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        #Prints out avg loss for each epoch, showing whether the training is working
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

    print("\nTraining complete!")
    torch.save(model.state_dict(), "weights/unet_gee_test.pth")
    print("Model weights saved to weights/unet_gee_test.pth")
    
