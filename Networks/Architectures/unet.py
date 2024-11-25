import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self,param):
        super().__init__()
        self.n_channels = param["MODEL"]["NB_CHANNEL"]
        self.n_classes  = param["MODEL"]["NB_CLASSES"]

        self.enc1 = EncoderBlock(self.n_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Expansive path (Decoder)
        self.dec1 = DecoderBlock(1024, 512, 512)
        self.dec2 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec4 = DecoderBlock(128, 64, 64)

        self.output_conv = nn.Conv2d(64, self.n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        # Contracting path
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b1 = F.relu(self.bottleneck_conv1(s4))
        b1 = F.relu(self.bottleneck_conv2(b1))

        # Expansive path
        d1 = self.dec1(b1, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        # Output
        outputs = torch.sigmoid(self.output_conv(d4))
        return outputs

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Max pooling with 2x2 filter
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # First convolution with ReLU activation
        x = F.relu(self.conv1(x))
        # Second convolution with ReLU activation
        x = F.relu(self.conv2(x))
        # Max pooling
        x = self.pool(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_features):
        # Upsample the input
        x = self.upsample(x)

        # Resize skip features to match x's shape
        skip_features = F.interpolate(skip_features, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        # Concatenate upsampled input with skip features
        x = torch.cat([x, skip_features], dim=1)

        # Apply two convolutions with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x

