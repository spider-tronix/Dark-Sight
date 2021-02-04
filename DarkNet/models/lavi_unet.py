# OM NAMO NARARAYNA

# https://github.com/lavi135246/pytorch-Learning-to-See-in-the-Dark/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import errno


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

        self.bn1 = torch.nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.bn2 = torch.nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bn3 = torch.nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bn4 = torch.nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.bn5 = torch.nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.bn6 = torch.nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.bn7 = torch.nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.bn8 = torch.nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn9 = torch.nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        if raw_format:
            self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
        else:
            self.conv10_1 = nn.Conv2d(32, 3, kernel_size=1, stride=1)

        self._initialize_weights()

    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        bn1 = self.bn1(pool1)

        conv2 = self.lrelu(self.conv2_1(bn1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        bn2 = self.bn2(pool2)

        conv3 = self.lrelu(self.conv3_1(bn2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        bn3 = self.bn3(pool3)

        conv4 = self.lrelu(self.conv4_1(bn3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        bn4 = self.bn4(pool4)

        conv5 = self.lrelu(self.conv5_1(bn4))
        conv5 = self.lrelu(self.conv5_2(conv5))
        bn5 = self.bn5(conv5)

        up6 = self.upv6(bn5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        bn6 = self.bn6(conv6)

        up7 = self.upv7(bn6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        bn7 = self.bn7(conv7)

        up8 = self.upv8(bn7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))
        bn8 = self.bn8(conv8)

        up9 = self.upv9(bn8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))
        bn9 = self.bn9(conv9)

        conv10 = self.conv10_1(bn9)
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

    save_dir = "./results/lavi_unet_results/driver_results/"
    os.chdir("./")
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(save_dir + "driver_img.png")

    print(output)
    plt.show()
