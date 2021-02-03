# OM NAMO NARARAYNA

# https://github.com/lavi135246/pytorch-Learning-to-See-in-the-Dark/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


class laviUnet(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def __init__(self, num_classes=10, inc_therm=True, raw_format=True):
        super(laviUnet, self).__init__()
        self.raw_format = raw_format
        if inc_therm:
            fchannel = 5
        else:
            fchannel = 4
        if not raw_format:
            fchannel = fchannel - 1
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = nn.Conv2d(fchannel, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        if raw_format:
            self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
        else:
            self.conv10_1 = nn.Conv2d(32, 3, kernel_size=1, stride=1)

        self._initialize_weights()

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        if self.raw_format:
            out = nn.functional.pixel_shuffle(conv10, 2)
        else:
            out = conv10
        # print(out.shape)
        out -= out.min(1, keepdim=True)[0]
        out /= out.max(1, keepdim=True)[0]
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt


if __name__ == "__main__":
    img_dir = "/home/arvinth/Pictures/dp.jpg"
    img = plt.imread(img_dir)
    model = laviUnet(10, inc_therm=False, raw_format=False)
    W = img.shape[1]
    H = img.shape[0]
    # xx = np.random.randint(0, )
    # yy = np.random.randint(0, H)
    xx = 0
    yy = 0
    ps = 512
    img = img[yy : yy + ps, xx : xx + ps, :]
    img = np.float32(np.expand_dims(img, axis=0) / 255.0)
    img = torch.tensor(img)
    img = torch.transpose(img, 1, 3)
    print("input shape", img.shape)
    output = model(img)
    print(output.shape)
    f, (ax1, ax2) = plt.subplots(2, 3)
    ax1[0].imshow(torch.transpose(img, 1, 3).detach().numpy()[0])
    ax1[1].imshow(torch.transpose(output, 1, 3).detach().numpy()[0])
    ax2[0].imshow(
        torch.transpose(output, 1, 3).detach().numpy()[0][:, :, 0], cmap="gray"
    )
    ax2[1].imshow(
        torch.transpose(output, 1, 3).detach().numpy()[0][:, :, 1], cmap="gray"
    )
    ax2[2].imshow(
        torch.transpose(output, 1, 3).detach().numpy()[0][:, :, 2], cmap="gray"
    )

    print(output)
    plt.show()
