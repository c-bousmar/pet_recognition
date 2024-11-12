import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_channels = param["MODEL"]["NB_CHANNEL"]
        self.n_classes  = param["MODEL"]["NB_CLASSES"]


        self.dble_conv_down1 = (Downscaling(self.n_channels, 64))
        self.dble_conv_down2 = (Downscaling(64, 128))
        self.dble_conv_down3 = (Downscaling(128, 256))
        self.dble_conv_down4 = (Downscaling(256, 512))
        self.bottel_neck     = (DoubleConvolution(512, 1024))
        self.up_dble_conv1   = (Upscaling(1024, 512))
        self.up_dble_conv2   = (Upscaling(512, 256))
        self.up_dble_conv3   = (Upscaling(256, 128))
        self.up_dble_conv4   = (Upscaling(128, 64))
        self.out             = nn.Conv2d(in_channels=64, out_channels=self.n_classes, kernel_size=1)

    def forward(self, x):
        x1  = self.dble_conv_down1(x)
        x2  = self.dble_conv_down2(x1[1])
        x3  = self.dble_conv_down3(x2[1])
        x4  = self.dble_conv_down4(x3[1])
        b   = self.bottel_neck(x4[1])
        x5  = self.up_dble_conv1(b, x4[0])
        x6  = self.up_dble_conv2(x5, x3[0])
        x7  = self.up_dble_conv3(x6, x2[0])
        x8  = self.up_dble_conv4(x7, x1[0])
        out = self.out(x8)
        return out
    
    
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Downscaling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvolution(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_a = self.conv(x)
        x_b = self.maxpool(x_a)
        return x_a, x_b
    

class Upscaling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x_a, x_b):
        x_a = self.up(x_a)
        x = torch.cat([x_a, x_b], 1)
        return self.conv(x)
