import torch
import torch.nn as nn


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

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=3, stride=1, padding=1)
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

        self.convT = nn.ConvTranspose2d(self.in_channels, self.out_channels,
                                        kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(self.out_channels * 2, self.out_channels,
                               kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=3, stride=1, padding=1)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, prior):
        x = self.convT(x)  # Expansion path

        cat = torch.cat((x, prior), dim=1)  # The Skip Connection!

        out = self.act(self.conv1(cat))
        out = self.act(self.conv2(out))

        return out
