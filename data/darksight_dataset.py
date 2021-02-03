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
    def __call__(self, sample):
        long_exp, short_exp, therm = (
            sample["long_exposure"],
            sample["short_exposure"],
            sample["thermal_response"],
        )
        long_exp = long_exp.postprocess(
            use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16
        )
        long_exp = np.float32(long_exp / 65535.0)
        short_exp = pack_raw(short_exp)
        return {
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "thermal_response": therm,
        }


class DarkSightDataset(Dataset):
    """DarkSight dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the datapoints.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = [PreprocessRaw()]
        if transform:
            for t in transform:
                self.transform.append(t)
        self.transform = transforms.Compose(self.transform)

        self.root_dir = root_dir
        self.long_exp_list = glob.glob(root_dir + "**/*long*.CR3", recursive=True)
        self.short_exp_list = glob.glob(root_dir + "**/*short*.CR3", recursive=True)
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
        long_exp = rawpy.imread(self.long_exp_list[idx])
        short_exp = rawpy.imread(self.short_exp_list[idx])
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
    transformed_dataset = DarkSightDataset(dataset_dir)
    data = list(transformed_dataset)
    print(data[0])
    # debugging

    # long_shot_cam = glob.glob(long_shots_dir + "**/*long*.CR3", recursive=True)
    # short_shot_cam = glob.glob(long_shots_dir + "**/*short*.CR3", recursive=True)
    # therm = glob.glob(long_shots_dir + "**/*temp.jpg", recursive=True)
    """dictionary format

    print('dataset1: ', data[0]['short_exposure'].shape,
        data[0]['thermal_response'].shape)
    # print('dataset2: ', data[1]['short_exposure'].shape,
    #       data[1]['thermal_response'].shape)
    # data[0]['thermal_response'].show()
    print(np.max(data[0]['long_exposure']))
    plt.figure()
    plt.imshow(data[0]['long_exposure'])
    plt.figure()
    plt.imshow(data[0]['thermal_response'])
    plt.figure()
    plt.imshow(data[0]['short_exposure'][:, :, :3])
    print(data[0]['short_exposure'][:, :, 1:4].shape)
    plt.show()
    print(data[0]['thermal_response'][0][20:30])

    """

    """tensor format"""
