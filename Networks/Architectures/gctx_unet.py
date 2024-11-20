import torch
import torch.nn as nn
from ..utils.gc_vit import gc_vit_tiny


class SE(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, int(in_channels * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(in_channels * expansion), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusedMBConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.gelu = nn.GELU()
        self.squeeze_excitation = SE(in_channels, in_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x_residual = x
        x = self.depthwise_conv(x)
        x = self.gelu(x)
        x = self.squeeze_excitation(x)
        return self.downsample(x) + x_residual
    

class Encoder(nn.Module):
    def __init__(self, encoder_stages, gc_vit_block):
        super(Encoder, self).__init__()
        self.stages = nn.ModuleList()
        in_channels = encoder_stages[0]
        for out_channels in encoder_stages[1:]:
            stage = nn.Sequential(
                gc_vit_block,
                gc_vit_block,
                FusedMBConv(in_channels, out_channels)  # Downsample
            )
            self.stages.append(stage)
            in_channels = out_channels

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return x, skips


class Bottleneck(nn.Module):
    def __init__(self, gc_vit_block):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            gc_vit_block,
            gc_vit_block
        )

    def forward(self, x):
        return self.bottleneck(x)
    

class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampler, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upsample(x)
    

class Decoder(nn.Module):
    def __init__(self, decoder_stages, bottleneck_channels, gc_vit_block):
        super(Decoder, self).__init__()
        self.stages = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        in_channels = bottleneck_channels
        for out_channels in decoder_stages:
            self.skip_connections.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))
            stage = nn.Sequential(
                Upsampler(in_channels, out_channels),
                gc_vit_block,
                gc_vit_block
            )
            self.stages.append(stage)
            in_channels = out_channels

    def forward(self, x, skips):
        for skip, stage in zip(reversed(skips), self.stages):
            x = stage(x)
            x = torch.cat([x, skip], dim=1)
        return x


class GCTx_UNet(nn.Module):
    def __init__(self, params):
        super(GCTx_UNet, self).__init__()

        input_channels = params["NB_CHANNEL"]
        num_classes = params["NB_CLASSES"]
        encoder_stages = params["ENCODER_STAGES"]
        decoder_stages = params["DECODER_STAGES"]
        self.gc_vit = gc_vit_tiny(embed_dim=64, num_heads=[2, 4, 8, 16], depths=[2, 2, 6, 2])

        self.patchify = nn.Sequential(
            nn.Conv2d(input_channels, encoder_stages[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(encoder_stages[0], encoder_stages[0], kernel_size=3, padding=1)
        )
        self.encoder = Encoder(encoder_stages, self.gc_vit_block)
        self.bottleneck = Bottleneck(encoder_stages[-1], self.gc_vit_block)
        self.decoder = Decoder(decoder_stages, encoder_stages[-1], self.gc_vit_block)
        self.final_upsample = nn.ConvTranspose2d(decoder_stages[-1], decoder_stages[-1], kernel_size=2, stride=2)
        self.output_layer = nn.Conv2d(decoder_stages[-1], num_classes, kernel_size=1)


    def forward(self, x):
        x = self.patchify(x) # Stem

        x, skips = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x, skips)

        x = self.final_upsample(x)
        return self.output_layer(x)
    