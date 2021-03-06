# OM NAMO NARAYANA

import glob
import rawpy
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from torch.nn import ReplicationPad2d
import tensorflow as tf
import sys
import torch

sys.path.insert(1, "./")

from DarkNet.models.sid_unet import sidUnet

# Ignore warnings
import warnings

from torchvision.transforms.functional import pil_to_tensor

warnings.filterwarnings("ignore")


def pack_raw(raw, blevel=512):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    # subtract the black level
    im = np.maximum(im - blevel, 0) / (16383 - blevel)
    # im = im / 16383

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )
    return out


class PreprocessRaw(object):
    def __init__(self, raw_format):
        self.raw_format = raw_format

    def __call__(self, sample):
        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )
        if self.raw_format:
            long_exp = long_exp.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
            )


            long_exp = np.float32(long_exp / 65535.0)
            short_exp = pack_raw(short_exp)
        else:
            long_exp = np.float32(long_exp / 255.0)
            short_exp = np.float32(short_exp / 255.0)


        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class DarkSightDataset(Dataset):
    """DarkSight dataset."""

    def __init__(self, root_dir, transform=None, raw_format=True):
        """
        Args:
            root_dir (string): Directory with all the datapoints.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.raw_format = raw_format
        self.transform = []
        self.transform.append(PreprocessRaw(raw_format=raw_format))
        if transform:
            for t in transform:
                self.transform.append(t)
        self.transform = transforms.Compose(self.transform)

        if raw_format:
            extension = ".CR3"
        else:
            extension = ".JPG"

        self.root_dir = root_dir
        self.long_exp_list = glob.glob(
            root_dir + "**/*raw*long*" + extension, recursive=True
        )
        self.short_exp_list = glob.glob(
            root_dir + "**/*raw*short*" + extension, recursive=True
        )
        self.therm_list = glob.glob(root_dir + "**/temp.jpg", recursive=True)
        print("no. of datapoints", len(self.long_exp_list))
        assert len(self.long_exp_list) == len(self.short_exp_list) and len(
            self.long_exp_list
        ) == len(self.therm_list)

    def __len__(self):
        return len(self.long_exp_list)

    def __getitem__(self, idx):
        print("data index", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.raw_format:
            long_exp = rawpy.imread(self.long_exp_list[idx])
            short_exp = rawpy.imread(self.short_exp_list[idx])
        else:
            long_exp = plt.imread(self.long_exp_list[idx])
            short_exp = plt.imread(self.short_exp_list[idx])
        therm = Image.open(self.therm_list[idx])
        therm = ImageOps.grayscale(therm)
        sample = {
            "long_exposure": long_exp,
            "short_exposure": short_exp,  # change to long_exp for debugging augmentation
            "thermal_response": therm,
        }

        sample = self.transform(sample)

        try:
            return [
                torch.tensor(sample["input_sample"].copy()),
                torch.tensor(sample["output_sample"].copy()),
            ]

        except:
            return sample


if __name__ == "__main__":

    # drive code
    dataset_dir = "./dataset/"
    transformed_dataset = DarkSightDataset(dataset_dir, raw_format=False)
    data = iter(transformed_dataset)
    next(data)
    # debugging
    # print(data[0])

