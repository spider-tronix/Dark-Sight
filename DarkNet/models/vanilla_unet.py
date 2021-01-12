import torch.nn as nn

from results.configs import *
from utils.module import model_summary


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        """
        Basic single downConv block
        :param in_channels: No of Input Channels
        :param out_channels: No of Output Channels
        :param pooling: Flag to toggle the pooling
        """
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        self.act = nn.LeakyReLU(0.2)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))

        prior = x

        if self.pooling:
            x = self.pool(x)

        return x, prior


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Basic single upConv block
        :param in_channels: No of input channels
        :param out_channels: No of output channels
        """
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convT = nn.ConvTranspose2d(
            self.in_channels, self.out_channels, kernel_size=2, stride=2
        )

        self.conv1 = nn.Conv2d(
            self.out_channels * 2, self.out_channels, kernel_size=3, stride=1, padding=1
        )

        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, prior):
        x = self.convT(x)  # Expansion path

        cat = torch.cat((x, prior), dim=1)  # The Skip Connection!

        out = self.act(self.conv1(cat))
        out = self.act(self.conv2(out))

        return out


class VanillaUNet(nn.Module):
    def __init__(self, channels_1=32):
        super(VanillaUNet, self).__init__()
        conv_sizes = [1, 2, 4, 8, 16]
        channels = [4] + [
            channels_1 * i for i in conv_sizes
        ]  # [4, 32, 64, 128, 256, 512]

        self.pools = [True] * 4 + [False]
        self.upsample_temps = nn.UpsamplingBilinear2d(size=SIZE)

        self.encoder = nn.ModuleList(
            [
                DownConv(in_channels, out_channels, pool)
                for in_channels, out_channels, pool in zip(
                    channels, channels[1:], self.pools
                )
            ]
        )

        self.decoder = nn.ModuleList(
            [
                UpConv(in_channels, out_channels)
                for in_channels, out_channels in zip(
                    channels[::-1], channels[::-1][1:-1]
                )
            ]
        )

        self.lastconv = nn.Conv2d(
            in_channels=channels_1, out_channels=3, kernel_size=1, stride=1, padding=0
        )

        # self.upscale = torch.nn.PixelShuffle(2)  # UP Sampling (Inspired fom SuperResolution)
        # TODO: Use upscale in Testing Phase

        self.trace = []

    def forward(self, x_1, x_2):  # input of shape (N, C, H, W)
        self.trace = []
        x_2 = self.upsample_temps(x_2)
        x = torch.cat([x_1, x_2], dim=1)

        for down_layer in self.encoder:
            x, prior = down_layer(x)
            self.trace.append(prior)
        self.trace.pop(-1)

        for up_layer, feature_map in zip(self.decoder, self.trace[::-1]):
            x = up_layer(x, feature_map)

        x = self.lastconv(x)  # no activation
        # x = self.upscale(x)
        return x


if __name__ == "__main__":
    net = VanillaUNet().cuda()
    model_summary(net)
    # low = torch.randint(2 ** 8, (1, 3, SIZE[0], SIZE[1])).float().cuda()
    # temps = torch.randint(40, (1, 1, 32, 24)).float().cuda()

    # TODO: Upsample vs interpolate (also bilinear from usage in papers)

    # output = net(low, temps)
    # print(outpu0t)
