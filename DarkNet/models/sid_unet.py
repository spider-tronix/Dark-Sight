# OM NAMO NARAYANA
import numpy as np
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt

# Ignore warnings
import warnings


class sidUnet(nn.Module):
    def __init__(self, inc_therm=True, raw_format=True):
        super(sidUnet, self).__init__()


        self.raw_format = raw_format
        if inc_therm:
            fchannel = 5
        else:
            fchannel = 4
        if not raw_format:
            fchannel = fchannel - 1
        
        self.conv1_1 = nn.Conv2d(fchannel, 32, 3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))

        self.pool = nn.MaxPool2d(1)

        self.up_6 = nn.ConvTranspose2d(512, 256, 2, stride=(2, 2))
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=(1, 1))
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))

        self.up_7 = nn.ConvTranspose2d(256, 128, 2, stride=(2, 2))
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=(1, 1))
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))

        self.up_8 = nn.ConvTranspose2d(128, 64, 2, stride=(2, 2))
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=(1, 1))
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))

        self.up_9 = nn.ConvTranspose2d(64, 32, 2, stride=(2, 2))
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=(1, 1))
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))

        self.up_10 = nn.ConvTranspose2d(32, 5, 2, stride=(2, 2))
        self.conv10_1 = nn.Conv2d(10, 10, 3, padding=(1, 1))
        self.conv10_1 = nn.Conv2d(10, 10, 3, padding=(1, 1))

        self.up_11 = nn.ConvTranspose2d(10, 3, 2, stride=(2, 2))
        self.conv_11_1 = nn.Conv2d(3, 3, 3, padding=(1, 1))

    def forward(self, x):

        fig, ax = plt.subplots(1, 11, figsize=(20, 20))

        conv1 = self.pool(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
        conv2 = self.pool(F.relu(self.conv2_2(F.relu(self.conv2_1(conv1)))))

        conv3 = self.pool(F.relu(self.conv3_2(F.relu(self.conv3_1(conv2)))))
        conv4 = self.pool(F.relu(self.conv4_2(F.relu(self.conv4_1(conv3)))))
        conv5 = self.pool(F.relu(self.conv5_2(F.relu(self.conv5_1(conv4)))))

        up6 = torch.cat((self.up_6(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_2(F.relu(self.conv6_1(up6))))
        up7 = torch.cat((self.up_7(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_2(F.relu(self.conv7_1(up7))))
        up8 = torch.cat((self.up_8(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_2(F.relu(self.conv8_1(up8))))
        up9 = torch.cat((self.up_9(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_2(F.relu(self.conv9_1(up9))))
        up10 = torch.cat((self.up_10(conv9), x), dim=1)
        conv10 = F.relu(self.conv10_1(up10))
        up11 = self.up_11(conv10)
        conv11 = self.conv_11_1(up11)

        out = conv11
        out = torch.transpose(out, 1, 3)  # 0, 2, 1, 3
        # out = torch.transpose(out, 2, 3)  # 0, 2, 3, 1
        # out = transforms.Normalize(0, 1)(out)
        out -= out.min(1, keepdim=True)[0]
        out /= out.max(1, keepdim=True)[0]

        ax[0].imshow(conv1.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[1].imshow(conv2.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[2].imshow(conv3.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[3].imshow(conv4.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[4].imshow(conv5.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[5].imshow(conv6.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[6].imshow(conv7.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[7].imshow(conv8.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[8].imshow(conv9.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[9].imshow(conv10.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax[10].imshow(conv11.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        plt.show()

        return out


if __name__ == "__main__":
    pass
    """trial run"""
    model = sidUnet()
    for i in range(1):
        # tin = torch.rand(1, 5, 512, 512)/1e5
        tin = torch.ones_like(torch.empty(1, 5, 512, 512))
        output = model.forward(tin)
        print(output.shape)
        fig, [ax1, ax2] = plt.subplots(1, 2)
        ax1.imshow(tin.detach().numpy()[0][:3, :, :].transpose(1, 2, 0))
        ax2.imshow(output.detach().numpy()[0])
        # plt.savefig("./results/black_noise/noise{}.png".format(i + 1))
        plt.show()
