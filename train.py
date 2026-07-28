import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.unet import UNet

if __name__ == "__main__":
    model = UNet(in_channels=6, out_channels=1)
    
