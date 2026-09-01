from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => batch normalization => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__() #initiliaze nn.Module class so PyTorch can track block weights
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), #first convolution 
            nn.BatchNorm2d(out_channels), #normalization
            nn.ReLU(inplace=True), #inplace=True modifies input tensor in memory (negs to 0)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1 ): #takes the 6 sentinel/NDVI bands and outputs 1 mask
        super().__init__() #to track the network's parameters

        #Encoding (contracting path)
        self.inc = DoubleConv(in_channels, 64) #scales the 6 channels to 64 feature maps
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128)) #1/2 spacial size, 2x channels
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) #again 
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512)) #again 

        #bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024)) #again 

        #decoder (expanding path)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) #First up convolution, d2x spatial size, 1/2 channels
        self.conv_up1 = DoubleConv(1024, 512) #merges up-sampled features w skip connection features

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        #Final output layer --- 64 feature maps -> 1 channel target  mask
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x): #defines path image takes through network
        #encoder
        x1 = self.inc(x) #initial DoubleConv block 256x256 64 channels
        x2 = self.down1(x1) #pooled and convolved 128x128 128
        x3 = self.down2(x2) #64x64 256
        x4 = self.down3(x3) #32x32 512
        x5 = self.down4(x4) #16x16 1024

        #decoder
        u1 = self.up1(x5) #up-samples bottleneck tensor 32x32 512 channels
        u1 = torch.cat([x4, u1], dim=1) #skip connection 2, glues x2 and u3 together
        c1 = self.conv_up1(u1) #smooths glued tensors with doubleconv

        u2 = self.up2(c1)
        u2 = torch.cat([x3, u2], dim=1) 
        c2 = self.conv_up2(u2)

        u3 = self.up3(c2)
        u3 = torch.cat([x2, u3], dim=1) 
        c3 = self.conv_up3(u3)

        u4 = self.up4(c3)
        u4 = torch.cat([x1, u4], dim=1) 
        c4 = self.conv_up4(u4)

        #output mapping
        logits = self.outc(c4) #flattens 64 channels -> 1 w raw prediction numbers
        return logits

if __name__ == "__main__":
    model = UNet(in_channels=6, out_channels=1)
    #simulate
    dummy_input = torch.randn(1, 6, 256, 256)
    output = model(dummy_input)

    print(f"Incoming tensor shape: {dummy_input.shape}")
    print(f"prediction shape: {output.shape}")
    print("\nU-Net architecture compiles!")

    
