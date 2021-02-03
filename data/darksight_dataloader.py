# OM NAMO NARAYANA

import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from torch.nn import ReplicationPad2d
import tensorflow as tf
import sys

sys.path.insert(1, "./")

from data.darksight_dataset import DarkSightDataset


class MatchSize(object):
    """
    __init__(self, cam_shape, therm_shape):
        Args:
            cam_shape: shape of short exposure camera image (matplotlib format)
            therm_shape: shape of thermal image (PIL fomrat)
        Returns:
            NULL
    __call__(self, sample):
        Args:
            {long_exposure, short_exposure, thermal_response}
        Returns:
            {long_exposure, short_exposure, thermal_response}: thermal_response in PIL format
    """

    def __init__(self, cam_shape, therm_shape):
        self.ratio = min(cam_shape[0] / therm_shape[1], cam_shape[1] / therm_shape[0])

        # resizing preserving aspect ratio
        self.therm_shape = (
            int(therm_shape[0] * self.ratio),
            int(therm_shape[1] * self.ratio),
        )

        lpad = int((cam_shape[0] - self.therm_shape[1]) / 2)
        rpad = cam_shape[0] - lpad - self.therm_shape[1]
        upad = int((cam_shape[1] - self.therm_shape[0]) / 2)
        dpad = cam_shape[1] - upad - self.therm_shape[0]
        self.padding = (upad, dpad, lpad, rpad)  # for RepllicationPad2d
        self.shape = cam_shape

    def __call__(self, sample):

        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )

        therm = therm.resize(self.therm_shape)
        m = ReplicationPad2d(self.padding)
        therm_tensor = transforms.ToTensor()(therm).unsqueeze_(0)
        therm_tensor = m(therm_tensor)

        # input and output are PIL image
        therm = transforms.ToPILImage()(therm_tensor.squeeze_(0))

        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class RandomCrop(object):
    """
    __init__(self, ps, hbuf, wbuf):
        Args:
            ps: patch size
            hbuf: minimum starting y coordinate
            wbuf: minimum sarting x coordinate
        Returns:
            NULL
    __call__(self, sample, color_percent=50):
       Args:
            {long_exposure, short_exposure, thermal_response}:thermal_response in PIL format
            color_percent[optional]: pecentage of pixels above the fixed threshold
        Returns:
            {long_exposure, short_exposure, thermal_response}:thermal_response in PIL formatat
    """

    def __init__(self, ps=512, hbuf=550, wbuf=550):
        self.ps = ps  # patch size
        self.hbuf = hbuf
        self.wbuf = wbuf

    def __call__(self, sample, color_percent=50):

        iter_times = 0
        while True:
            iter_times += 1
            long_exp, short_exp, therm = (
                sample["long_exposure"],
                sample["short_exposure"],
                sample["thermal_response"],
            )
            W = short_exp.shape[1]
            H = short_exp.shape[0]
            ps = self.ps
            xx = np.random.randint(self.wbuf, W - ps - self.wbuf)
            yy = np.random.randint(self.hbuf, H - ps - self.hbuf)
            short_exp = short_exp[yy : yy + ps, xx : xx + ps, :]
            therm = therm.crop((xx, yy, xx + ps, yy + ps))
            long_exp = long_exp[yy * 2 : yy * 2 + ps * 2, xx * 2 : xx * 2 + ps * 2, :]
            cnt = 0
            for i in range(0, 512):
                for j in range(0, 512):
                    r = long_exp[i][j][0] * 255
                    g = long_exp[i][j][1] * 255
                    b = long_exp[i][j][2] * 255
                    if r < 40 and g < 40 and b < 40:
                        cnt += 1
            # color_precent added
            if cnt < ((512 * 512) * color_percent / 100) or iter_times > 10:
                print("sampled on {} iteration".format(iter_times))
                break
        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class RandomFlip(object):

    """
    __call__(self, sample):
        Args:
            {long_exposure, short_exposure, thermal_response}:thermal_response in PIL format
        Returns:
            {long_exposure, short_exposure, thermal_response}:thermal_response in numpy format
    """

    def __call__(self, sample):

        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )
        therm = np.array(therm)
        if np.random.randint(2, size=1)[0] == 1:  # random flip
            short_exp = np.flip(short_exp / 255.0, axis=0)
            therm = np.flip(therm, axis=0)
            long_exp = np.flip(long_exp, axis=0)
        if np.random.randint(2, size=1)[0] == 1:
            short_exp = np.flip(short_exp, axis=1)
            therm = np.flip(therm, axis=1)
            long_exp = np.flip(long_exp, axis=1)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            short_exp = np.transpose(short_exp, (1, 0, 2))
            therm = np.transpose(therm, (1, 0))
            long_exp = np.transpose(long_exp, (1, 0, 2))
        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class ConcatTherm(object):

    """
    __call__(self, sample):
        Args:
            {long_exposure, short_exposure, thermal_response}:thermal_response in numpy format
    Returns:
            {input_sample, output_sample}:input_sample is concatenation of short_exposure and thermal response
    """

    def __call__(self, sample):

        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )
        input_sample = np.transpose(short_exp, (2, 0, 1))
        input_sample = np.append(input_sample, np.expand_dims(therm, axis=0), axis=0)
        return {"input_sample": input_sample, "output_sample": long_exp}


def my_transform(train=True, cam_shape=(2010, 3012), therm_shape=(32, 24)):
    transform = []
    transform.append(MatchSize(cam_shape, therm_shape))
    transform.append(RandomCrop())
    transform.append(RandomFlip())
    transform.append(ConcatTherm())
    return transform


class DarkSighDataLoader:
    """
    load(self, batch_size=1, shuffle=True):
        Returns:
            dataloader 
    """
    def __init__(self):
        self.dataset_dir = "./dataset/"
        self.transformed_dataset = DarkSightDataset(
            self.dataset_dir, transform=my_transform(True)
        )
        self.data = list(self.transformed_dataset)

    def load(self, batch_size=1, shuffle=True):
        dataloader = DataLoader(self.data, batch_size, shuffle=shuffle)
        return dataloader


if __name__ == "__main__":
    data = DarkSighDataLoader().load()
    data = iter(data)
    print(next(data))
